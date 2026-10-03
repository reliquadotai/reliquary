"""End-to-end run of an agentic corpus job on the test boxes: a real miner
(Qwen3.8-27B, prefix caching), a real corpus validator and real grade
executors, through submission, TOPLOC verdict, grade, replay and export, with
two forgeries (a forged final_diff, forged observations) that must end as
confirmed audit failures.

The 27B cannot be served twice on one H100: the validator runs intake-only
(no model) while the miner holds the card, then restarts with the model to
audit. Storage is a throwaway MinIO; the production bucket is refused.
Runbook: docs/runbooks/agentic-corpus-swe.md, section "End-to-end run".

Grade executors are started with the real CLI (`reliquary corpus
register-grade-executor` / `grade-executor`), not by this script. A replay
failure sanctions only when executors of TWO DISTINCT PROVIDERS agree (ruling
P17), so the run needs two executors with distinct `--provider-id`, one per
Docker host.

Secrets never touch a file: R2_ACCESS_KEY_ID and R2_SECRET_ACCESS_KEY come from
the environment of every subcommand (`--start-minio` starts MinIO with them);
the state directory keeps only the bucket name, endpoint and region. The role
hotkeys are generated in memory by `mine`; only their public addresses are
written.

    export R2_BUCKET_ID=reliquary-agentic-e2e R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=...
    python scripts/agentic_corpus_e2e.py prepare --state /opt/agentic-e2e --env-commit SHA --start-minio
    python scripts/agentic_corpus_e2e.py validator --state /opt/agentic-e2e --intake-only
    python scripts/agentic_corpus_e2e.py mine --state /opt/agentic-e2e
    python scripts/agentic_corpus_e2e.py wait-grades --state /opt/agentic-e2e
    python scripts/agentic_corpus_e2e.py validator --state /opt/agentic-e2e        # full: audits
    python scripts/agentic_corpus_e2e.py wait-verdicts --state /opt/agentic-e2e
    python scripts/agentic_corpus_e2e.py check --state /opt/agentic-e2e --out summary.json

`check` exits 0 on success, 1 on a failure, 2 when inconclusive (a forgery did
not exercise its path: rerun with a new --state and another --prompt-start).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

JOB_ID = "agentic-e2e"
MODEL = "Qwen/Qwen3.8-27B"
DEFAULT_BUCKET = "reliquary-agentic-e2e"
ROLES = ("honest", "forge_diff", "forge_obs")
FORGERS = ("forge_diff", "forge_obs")
# Two executors must agree, from two providers (rulings P16/P17).
SANCTION_EXECUTORS = 2
# Appended to the gold patch: the forged diff still passes grading (the fix is
# in it, the new file touches no test) but is not what the actions produce.
FORGED_HUNK = ("diff --git a/FORGED.txt b/FORGED.txt\nnew file mode 100644\n--- /dev/null\n"
               "+++ b/FORGED.txt\n@@ -0,0 +1 @@\n+forged\n")
# Every bash command of the forging box sources /etc/hostname (present in every
# container, its content the container id) and prints "/etc/hostname: line 1:
# <id>: command not found": the model sees it, the replay box does not, so
# every bash observation differs. A missing file would not do: bash ignores a
# BASH_ENV that does not exist (measured 2026-10-03, a first run's forged
# episode replayed with 0 mismatches). Edit observations are untouched (the
# edit tool runs no bash).
FORGED_BASH_ENV = {"BASH_ENV": "/etc/hostname"}
# Fresh keys tried per forger before giving up on a free prompt.
FORGER_KEY_ATTEMPTS = 2000
# Environment variables never written to the state directory.
_SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD")
# Never the single-turn e2e's "corpus-e2e-minio": starting this run must not
# remove a MinIO another run is using.
MINIO_CONTAINER = "agentic-e2e-minio"


# --------------------------------------------------------------------------
# The verdict of the run (pure)
# --------------------------------------------------------------------------

def _allowed(replay: Mapping) -> int:
    allowed = replay.get("allowed")
    if allowed is not None:
        return int(allowed)
    from reliquary.corpus.replay_compare import allowed_mismatches

    return allowed_mismatches(int(replay.get("observations_compared") or 0))


def _judge_forgery(role: str, s: Mapping, failures: list[str],
                   inconclusive: list[str]) -> bool:
    """Judge one forged submission; False when it did not exercise its path."""
    replay = s.get("replay") or {}
    if not s["verdict_passed"]:
        inconclusive.append(f"{role}: caught by TOPLOC, not by the replay")
        return False
    if role == "forge_diff" and not s["graded_success"]:
        inconclusive.append("forge_diff: the forged diff did not pass grading")
        return False
    if role == "forge_obs":
        # Ruling P7: only bash observations are forged, so only they can
        # exceed the replay's tolerance.
        bash = s.get("bash_observations")
        if bash is None:
            inconclusive.append("forge_obs: its bash observations could not be counted")
            return False
        if bash <= _allowed(replay):
            inconclusive.append(f"forge_obs: {bash} bash observations cannot exceed the "
                                f"tolerance of {_allowed(replay)}")
            return False
    if replay.get("status") != "ok":
        failures.append(f"{role}: the replay ended {replay.get('status')!r}, not a "
                        f"confirmed failure ({replay})")
    elif not replay.get("failed"):
        failures.append(f"{role}: the replay did not fail ({replay})")
    elif len(set(replay.get("graded_by") or ())) < SANCTION_EXECUTORS:
        failures.append(f"{role}: a failure not confirmed by two executors")
    elif len(set(replay.get("providers") or ())) < SANCTION_EXECUTORS:
        failures.append(f"{role}: a failure not confirmed by two providers")
    if not s["voided"]:
        failures.append(f"{role}: not voided")
    elif s.get("void_reason") != "replay_failed":
        failures.append(f"{role}: voided as {s.get('void_reason')!r}, not replay_failed")
    if role == "forge_diff" and replay.get("replay_diff_equal") is not False:
        failures.append("forge_diff: the replayed diff equalled the forged one")
    return True


def evaluate(summary: dict) -> tuple[list[str], list[str]]:
    """(failures, inconclusive) of a run, from ``check``'s summary."""
    failures: list[str] = []
    inconclusive: list[str] = []
    roles = summary["roles"]

    honest = roles["honest"]
    subs = honest["submissions"]
    if not subs:
        failures.append("honest: no accepted submission")
    if any(not s["verdict_passed"] for s in subs):
        failures.append("honest: a TOPLOC verdict failed")
    if any(s["voided"] for s in subs) or honest["state"]["confirmed_failures"]:
        failures.append("honest: voided or charged a confirmed failure")
    if not any(s["graded_success"] and s["replay_certified"] for s in subs):
        failures.append("honest: no certified success")

    for role in FORGERS:
        forger = roles[role]
        if not forger["submissions"]:
            failures.append(f"{role}: no accepted submission to judge")
            continue
        judged = [_judge_forgery(role, s, failures, inconclusive) for s in forger["submissions"]]
        if any(judged) and forger["state"]["confirmed_failures"] < 1:
            failures.append(f"{role}: no confirmed failure on the miner")

    export = summary["export"]
    if export["sft_rows"] < 1:
        failures.append("export: no certified row")
    forgers = {roles[role]["hotkey"] for role in FORGERS}
    if forgers & set(export["sft_hotkeys"]):
        failures.append("export: a forger's row reached the SFT set")
    return failures, inconclusive


def forgeable_bash_observations(actions: Iterable) -> int:
    """The observations ``FORGED_BASH_ENV`` changes: answered bash calls whose
    arguments are a JSON object (anything else the harness answers with an
    error of its own, without running bash)."""
    count = 0
    for action in actions:
        if action.tool != "bash" or action.observation is None:
            continue
        try:
            arguments = json.loads(action.arguments or "{}")
        except (TypeError, ValueError):
            continue
        count += isinstance(arguments, dict)
    return count


def pick_forger_keys(job, honest_hotkey: str, *, honest_episodes: int, make_key) -> dict:
    """A key per forger whose first prompt no other role visits.

    The walk draws prompts with replacement, so a forger's single episode can
    land on a prompt the honest miner (or the other forger) fills first and be
    refused ``prompt_full`` (a 2026-10-03 run lost its forged diff that way).
    Keys are free: draw until each forger's cursor 0 is a prompt nobody else's
    walk visits."""
    from reliquary.corpus.walk import job_walk_index

    taken = {job_walk_index(job, honest_hotkey, c) for c in range(honest_episodes)}
    keys = {}
    for role in FORGERS:
        for _ in range(FORGER_KEY_ATTEMPTS):
            key = make_key()
            prompt = job_walk_index(job, key.ss58_address, 0)
            if prompt not in taken:
                keys[role] = key
                taken.add(prompt)
                break
        else:
            raise SystemExit(f"no free prompt for {role}: the honest walk and the other "
                             f"forger cover every prompt; raise --prompt-count")
    return keys


def forged_diff(gold_patch: str) -> str:
    return gold_patch.rstrip("\n") + "\n" + FORGED_HUNK


def wait_complete(submissions: set, done: set) -> bool:
    return bool(submissions) and submissions <= done


def public_r2_settings(env: Mapping[str, str]) -> dict[str, str]:
    """The R2_* settings the state directory may keep: no credential."""
    return {k: v for k, v in env.items()
            if k.startswith("R2_") and not any(m in k for m in _SECRET_MARKERS)}


# --------------------------------------------------------------------------
# Shared state
# --------------------------------------------------------------------------

def _require_credentials() -> None:
    missing = [k for k in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY") if not os.environ.get(k)]
    if missing:
        raise SystemExit(f"set {' and '.join(missing)} in the environment (never in a file)")


def _load_env(state: Path) -> None:
    os.environ.update(json.loads((state / "env.json").read_text()))
    _require_credentials()
    from scripts.corpus_e2e import _refuse_production_bucket

    _refuse_production_bucket()
    os.environ["RELIQUARY_TASK_CONTRACT"] = str(state / "contract.json")
    os.environ["RELIQUARY_TASK_ID"] = JOB_ID
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")


# --------------------------------------------------------------------------
# Subcommands
# --------------------------------------------------------------------------

def prepare(args) -> None:
    state = Path(args.state)
    os.environ.setdefault("R2_BUCKET_ID", DEFAULT_BUCKET)
    _require_credentials()
    from scripts.corpus_e2e import (
        _refuse_production_bucket, conditional_put_preflight, ensure_bucket, start_minio,
    )

    _refuse_production_bucket()
    if (state / "job.json").exists():
        raise SystemExit(f"{state} already holds a run; name a new --state")
    from reliquary.corpus.job import parse_job
    from reliquary.environment.agentic_swe import (
        SUPPORTED_RENDERER, SUPPORTED_VERIFIERS, episode_support_refusal, load_turn_renderer,
    )

    episode = {"env": {"package": "reliquary-swe", "version": args.env_commit, "split": "train",
                       "num_images": args.num_images},
               "harness": "bash", "renderer": SUPPORTED_RENDERER, "verifiers": SUPPORTED_VERIFIERS,
               "max_turns": 40, "max_tokens_per_turn": 8192, "max_total_tokens": 60000,
               # Both forgeries are replayed whatever their grade.
               "replay_fraction_failed": 1.0}
    if args.start_minio:
        start_minio(container=MINIO_CONTAINER, credentials_from_env=True)
    state.mkdir(parents=True, exist_ok=True)
    (state / "env.json").write_text(json.dumps(public_r2_settings(os.environ), indent=1))
    asyncio.run(ensure_bucket())
    preflight = asyncio.run(conditional_put_preflight())
    if not all(preflight.values()):
        raise SystemExit(f"the bucket does not honour conditional PUT: {preflight}")

    from huggingface_hub import snapshot_download

    from reliquary.cli.main import prepare_corpus_job
    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.infrastructure.corpus_job_store import write_job
    from reliquary.protocol.agentic_source import AGENTIC_SWE_ENVIRONMENT

    directory = Path(snapshot_download(MODEL))
    architecture = json.loads((directory / "config.json").read_text())["architectures"][0]
    manifest, entry = prepare_corpus_job(
        job_id=JOB_ID, task_id=JOB_ID, model=MODEL, model_revision=directory.name,
        model_architecture=architecture, checkpoint_sha256=checkpoint_fingerprint(directory),
        from_profile=None, prompt_encoding=None, prompt_source=AGENTIC_SWE_ENVIRONMENT,
        prompt_count=args.prompt_count, prompt_start=args.prompt_start,
        renderer_id=SUPPORTED_RENDERER, eos_token_id=load_turn_renderer(str(directory)).terminator_id,
        slots_per_prompt=args.slots, max_new_tokens=8192, cap=args.cap, min_incentive_share=0.0,
        audit_params={"audit_q": 1.0}, prompt_order="free", episode=episode)
    # This box mines and grades too: refuse before the bucket holds a job.
    refusal = episode_support_refusal(parse_job(manifest).episode, need_verifiers=True)
    if refusal:
        raise SystemExit(f"this box cannot run the job: {refusal}")
    asyncio.run(write_job(manifest, None))
    (state / "job.json").write_text(json.dumps(manifest, indent=1))
    (state / "contract.json").write_text(json.dumps(entry.contract, sort_keys=True))
    (state / "entry.json").write_text(json.dumps(asdict(entry), indent=1, sort_keys=True, default=str))
    print(json.dumps({"job": JOB_ID, "checkpoint": str(directory), "bucket": os.environ["R2_BUCKET_ID"],
                      "prompts": [args.prompt_start, args.prompt_start + args.prompt_count]}))


def audit_attention(flash_attn_installed: bool, env: Mapping[str, str]) -> str | None:
    """The GRAIL_ATTN_IMPL the audit must be given, or None to keep the
    environment's. The validator defaults to flash_attention_2 and refuses to
    load the model without the package; the test GPU box's vLLM venv has none
    (found 2026-10-03), and gate M1 measured the 27B's TOPLOC bands with sdpa."""
    if env.get("GRAIL_ATTN_IMPL") or flash_attn_installed:
        return None
    return "sdpa"


def validator(args) -> None:
    state = Path(args.state)
    import importlib.util

    attention = audit_attention(importlib.util.find_spec("flash_attn") is not None, os.environ)
    if attention and not args.intake_only:
        print(f"flash_attn is not installed: the audit runs with GRAIL_ATTN_IMPL={attention}",
              file=sys.stderr, flush=True)
        os.environ["GRAIL_ATTN_IMPL"] = attention
    _load_env(state)
    from scripts.corpus_e2e import load_entry
    from reliquary.validator.corpus_validator import run_corpus_validator

    entry = load_entry(state / "entry.json", task_id=JOB_ID, job_id=JOB_ID)
    # Settlement never runs (`check` reads grades, voids and miner states, not pay).
    asyncio.run(run_corpus_validator(
        entry=entry, cap=float(entry.params["cap"]), wallet=None, netuid=0, signer_client=None,
        http_host="127.0.0.1", http_port=args.port, set_weights=False, settle_every_seconds=1e9,
        registration_gate=False, intake_only=args.intake_only))


def mine(args) -> None:
    state = Path(args.state)
    _load_env(state)
    if (state / "hotkeys.json").exists():
        raise SystemExit(f"{state} was mined already (its hotkeys are gone); name a new --state")
    import bittensor as bt
    import httpx
    from huggingface_hub import snapshot_download
    from reliquary_swe import corpus

    from reliquary.corpus.job import parse_job
    from reliquary.environment.agentic_swe import episode_support_refusal
    from reliquary.miner.agentic_miner import Identity, run_agentic_miner
    from reliquary.miner.corpus_miner import HttpCorpusClient, submits_scoped
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.protocol.signatures import sign_corpus_submission
    from reliquary.shared.modeling import load_tokenizer

    client = HttpCorpusClient(httpx.Client(base_url=f"http://127.0.0.1:{args.validator_port}",
                                           timeout=300.0), job_id=JOB_ID)
    job = parse_job(client.job())
    client.scoped_submit = submits_scoped(job)
    refusal = episode_support_refusal(job.episode, need_verifiers=True)
    if refusal:
        raise SystemExit(f"refusing to mine: {refusal}")
    directory = snapshot_download(job.checkpoint_repo, revision=job.checkpoint_revision)
    # As `corpus mine-agentic` does: a checkpoint other than the job's would
    # only fail TOPLOC, and the run would read as a forgery test gone wrong.
    from reliquary.corpus.encoding import checkpoint_fingerprint

    if checkpoint_fingerprint(directory) != job.checkpoint_sha256:
        raise SystemExit("refusing to mine: the downloaded checkpoint does not match the "
                         "job's fingerprint")
    # Throwaway, unregistered keys held in memory only: the run never needs them again.
    def new_key():
        return bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())

    keys = {"honest": new_key()}
    keys.update(pick_forger_keys(job, keys["honest"].ss58_address,
                                 honest_episodes=args.honest_episodes, make_key=new_key))
    (state / "hotkeys.json").write_text(json.dumps({r: k.ss58_address for r, k in keys.items()}))

    def signer(keypair):
        wallet = SimpleNamespace(hotkey=keypair)
        return lambda body: sign_corpus_submission(wallet, body)

    rows = corpus.load_swesmith_rows(job.episode.env.num_images)

    def forge_diff(built, index):
        return replace(built, final_diff=forged_diff(rows[index].gold_patch))

    identities = [
        Identity(keys["honest"].ss58_address, signer(keys["honest"]), episodes=args.honest_episodes),
        Identity(keys["forge_diff"].ss58_address, signer(keys["forge_diff"]), episodes=1,
                 transform=forge_diff),
        Identity(keys["forge_obs"].ss58_address, signer(keys["forge_obs"]), episodes=1,
                 harness_env=FORGED_BASH_ENV),
    ]
    counts = asyncio.run(run_agentic_miner(
        job=job, checkpoint_dir=directory, proof=toploc_proof(ACTIVE_PROTOCOL_PROFILE),
        tokenizer=load_tokenizer(directory), identities=identities, client=client,
        concurrency=args.concurrency, port=args.port,
        gpu_memory_utilization=args.gpu_memory_utilization, max_num_seqs=args.max_num_seqs))
    by_role = {role: dict(counts[keys[role].ss58_address]) for role in ROLES}
    (state / "mine.json").write_text(json.dumps(by_role, indent=1))
    print(json.dumps(by_role))


def _grading_status(port: int) -> dict | None:
    """The control's grading backlog (held_executors, unindexed,
    dispatcher_waiting...), or None when it does not answer."""
    import urllib.request

    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/corpus/jobs/{JOB_ID}/status",
                                    timeout=10) as response:
            return json.loads(response.read()).get("grading")
    except (OSError, ValueError):
        return None


async def _wait(kind: str, timeout: float, port: int, poll: float) -> None:
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    records = BucketRecordStore()
    lister = records.list_grade_ids if kind == "grades" else records.list_verdict_ids
    deadline = time.time() + timeout
    while time.time() < deadline:
        submissions = set(await records.list_submission_ids(JOB_ID))
        done = set(await lister(JOB_ID))
        print(json.dumps({"at": round(time.time()), "submissions": len(submissions),
                          kind: len(done & submissions),
                          "grading": await asyncio.to_thread(_grading_status, port)}), flush=True)
        if wait_complete(submissions, done):
            return
        await asyncio.sleep(poll)
    raise SystemExit(f"{kind} not complete after {timeout:.0f}s")


def wait_grades(args) -> None:
    _load_env(Path(args.state))
    asyncio.run(_wait("grades", args.timeout, args.validator_port, args.poll))


def wait_verdicts(args) -> None:
    _load_env(Path(args.state))
    asyncio.run(_wait("verdicts", args.timeout, args.validator_port, args.poll))


def _bash_observations(renderer, job, record) -> int | None:
    from reliquary.corpus.trajectory_parse import TrajectoryRefused, parse_trajectory

    try:
        trajectory = record["completions"][0]
        parsed = parse_trajectory(
            renderer, prompt_ids=trajectory["prompt_tokens"], tokens=trajectory["tokens"],
            spans=[(turn["start"], turn["end"]) for turn in trajectory["turns"]],
            stop=trajectory["stop"], max_turns=job.episode.max_turns)
    except (TrajectoryRefused, KeyError, IndexError, TypeError):
        return None
    return forgeable_bash_observations(parsed.actions)


async def _summary(state: Path) -> dict:
    from reliquary.cli.main import _episode_tokenizer_dir, _quarantined_grade_executors
    from reliquary.corpus.delivery import episode_rows
    from reliquary.corpus.job import parse_job
    from reliquary.environment.agentic_swe import load_swe_source, load_turn_renderer
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    records = BucketRecordStore()
    job = parse_job(json.loads((state / "job.json").read_text()))
    hotkeys = json.loads((state / "hotkeys.json").read_text())
    role_of = {hotkey: role for role, hotkey in hotkeys.items()}
    renderer = await asyncio.to_thread(load_turn_renderer,
                                       await asyncio.to_thread(_episode_tokenizer_dir, job))
    voided = set(await records.list_voided_ids(JOB_ID))
    miners, _ = await records.read_miners(JOB_ID)
    roles = {role: {"hotkey": hotkeys[role], "submissions": [],
                    "state": {"confirmed_failures": len((miners.get(hotkeys[role]) or {})
                                                        .get("confirmed_failures") or [])}}
             for role in ROLES}
    for sid in await records.list_submission_ids(JOB_ID):
        record = await records.read_submission(JOB_ID, sid)
        role = role_of.get((record or {}).get("hotkey"))
        if role is None:
            continue
        verdict = await records.read_verdict(JOB_ID, sid) or {}
        grade = await records.read_regrade(JOB_ID, sid) or await records.read_grade(JOB_ID, sid) or {}
        void = await records.read_voided(JOB_ID, sid) if sid in voided else None
        roles[role]["submissions"].append({
            "sid": sid, "prompt_index": record.get("prompt_index"),
            "verdict_passed": bool(verdict.get("passed")), "verdict_reason": verdict.get("reason"),
            "grade_status": grade.get("status"), "graded_success": bool(grade.get("graded_success")),
            "graded_by": grade.get("graded_by"),
            "replay_certified": bool(grade.get("replay_certified")), "voided": sid in voided,
            "void_reason": (void or {}).get("reason"), "replay": grade.get("replay"),
            "bash_observations": await asyncio.to_thread(_bash_observations, renderer, job, record)})
    source = await asyncio.to_thread(load_swe_source, job.episode.env.num_images)
    quarantined = await _quarantined_grade_executors()
    sft = []
    counts: dict = {}
    async for row in episode_rows(job=job, records=records, renderer=renderer, source=source,
                                  counts=counts, sft_only=True, quarantined=quarantined):
        record = await records.read_submission(JOB_ID, row["submission_id"])
        sft.append({**row, "hotkey": record["hotkey"]})
    with open(state / "sft.jsonl", "w", encoding="utf-8") as handle:
        for row in sft:
            handle.write(json.dumps({**row, "messages": json.loads(row["messages"]),
                                     "turns": json.loads(row["turns"])}) + "\n")
    return {"roles": roles, "quarantined_grade_executors": quarantined,
            "export": {"sft_rows": len(sft), "counts": counts,
                       "sft_hotkeys": sorted({r["hotkey"] for r in sft})}}


def check(args) -> None:
    state = Path(args.state)
    _load_env(state)
    summary = asyncio.run(_summary(state))
    failures, inconclusive = evaluate(summary)
    mined = state / "mine.json"
    summary.update(failures=failures, inconclusive=inconclusive,
                   mine=json.loads(mined.read_text()) if mined.exists() else None)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps({"failures": failures, "inconclusive": inconclusive,
                      "sft_rows": summary["export"]["sft_rows"]}, indent=1))
    raise SystemExit(1 if failures else (2 if inconclusive else 0))


COMMANDS = {"prepare": prepare, "validator": validator, "mine": mine, "wait-grades": wait_grades,
            "wait-verdicts": wait_verdicts, "check": check}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)
    for name in COMMANDS:
        command = sub.add_parser(name)
        command.add_argument("--state", required=True)
    prep = sub.choices["prepare"]
    prep.add_argument("--env-commit", required=True,
                      help="reliquary-environments commit; must be the installed reliquary-swe")
    prep.add_argument("--num-images", type=int, default=4)
    prep.add_argument("--prompt-start", type=int, default=0)
    prep.add_argument("--prompt-count", type=int, default=8)
    prep.add_argument("--slots", type=int, default=2)
    prep.add_argument("--cap", type=float, default=0.05)
    prep.add_argument("--start-minio", action="store_true",
                      help="Start MinIO on 127.0.0.1:9000 with the R2_* credentials of the environment")
    sub.choices["validator"].add_argument("--port", type=int, default=8100)
    sub.choices["validator"].add_argument("--intake-only", action="store_true")
    mine_parser = sub.choices["mine"]
    mine_parser.add_argument("--validator-port", type=int, default=8100)
    mine_parser.add_argument("--port", type=int, default=8011,
                             help="Loopback port of the miner's generate endpoint")
    mine_parser.add_argument("--honest-episodes", type=int, default=6)
    mine_parser.add_argument("--concurrency", type=int, default=8)
    mine_parser.add_argument("--max-num-seqs", type=int, default=16)
    mine_parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    for name, timeout in (("wait-grades", 4 * 3600), ("wait-verdicts", 3600)):
        sub.choices[name].add_argument("--timeout", type=float, default=float(timeout))
        sub.choices[name].add_argument("--poll", type=float, default=30.0)
        sub.choices[name].add_argument("--validator-port", type=int, default=8100)
    sub.choices["check"].add_argument("--out", required=True)
    return p


def main() -> None:
    args = build_parser().parse_args()
    COMMANDS[args.command](args)


if __name__ == "__main__":
    main()
