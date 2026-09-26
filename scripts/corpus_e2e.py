"""End-to-end run of a corpus task on one card: declare, mine, audit, settle, export.

One orchestrator drives three separate processes on the same GPU (the validator,
then an honest miner, then a dishonest one), because two vLLM engines or two
capture hooks cannot share one process. Storage is a throwaway MinIO
(``--start-minio``) or whatever test bucket ``R2_*`` names; the production
bucket ``reliquary`` is refused.

    R2_BUCKET_ID=reliquary-corpus-e2e python scripts/corpus_e2e.py --start-minio

With ``--audit-q/--audit-probation/--audit-hold-seconds`` the task samples its
audits, and ``--late-cheater-steps K`` adds a hotkey that mines honestly through
probation, then K steps with the dishonest model: it must turn suspect at its
first failed audit, every record it still has in hold must be audited, and none
of its switched records may be paid.

Prints one JSON summary on stdout and exits non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import secrets
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
# Run from a source tree, not an installed package.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
MINIO_CONTAINER = "corpus-e2e-minio"
MINIO_IMAGE = "quay.io/minio/minio:latest"
PRODUCTION_BUCKET = "reliquary"

logger = logging.getLogger("corpus_e2e")


# --------------------------------------------------------------------------
# Environment shared by every process of the run
# --------------------------------------------------------------------------

def _child_env(state: Path, task_id: str) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{ROOT}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    env["RELIQUARY_TASK_CONTRACT"] = str(state / "contract.json")
    env["RELIQUARY_TASK_ID"] = task_id
    # The capture hook needs the in-process V1 runner; the box has no flash-attn.
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    env.setdefault("VLLM_USE_V2_MODEL_RUNNER", "0")
    env.setdefault("GRAIL_ATTN_IMPL", "sdpa")
    return env


def _refuse_production_bucket() -> None:
    bucket = os.environ.get("R2_BUCKET_ID", "")
    if not bucket or bucket == PRODUCTION_BUCKET:
        raise SystemExit(
            f"refusing to run: R2_BUCKET_ID is {bucket!r}; name a test bucket "
            "(an unset one defaults to the production bucket)"
        )


def _refuse_busy_card(allow: bool) -> None:
    # The card may be shared with another session: three processes of this run
    # plus someone else's job would OOM both, so stop before loading anything.
    used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                          capture_output=True, text=True, check=True).stdout.split()
    if not allow and any(int(mib) > 2048 for mib in used):
        raise SystemExit(f"refusing to run: the card already has {used} MiB in use")


def _snapshot(repo: str, revision: str | None) -> Path:
    from huggingface_hub import snapshot_download

    # Online on purpose: huggingface_hub refuses an offline snapshot missing its
    # README/LICENSE, and the validator resolves the job's checkpoint the same way.
    return Path(snapshot_download(repo, revision=revision))


# --------------------------------------------------------------------------
# Storage: MinIO and the conditional-PUT preflight
# --------------------------------------------------------------------------

def start_minio() -> None:
    user, password = f"e2e{secrets.token_hex(6)}", secrets.token_hex(20)
    subprocess.run(["docker", "rm", "-f", MINIO_CONTAINER], capture_output=True)
    subprocess.run(
        ["docker", "run", "-d", "--name", MINIO_CONTAINER,
         "-e", f"MINIO_ROOT_USER={user}", "-e", f"MINIO_ROOT_PASSWORD={password}",
         "-p", "127.0.0.1:9000:9000", MINIO_IMAGE, "server", "/data"],
        check=True, capture_output=True,
    )
    os.environ.update({
        "R2_ENDPOINT_URL": "http://127.0.0.1:9000",
        "R2_ACCESS_KEY_ID": user,
        "R2_SECRET_ACCESS_KEY": password,
        "R2_REGION": "us-east-1",
    })
    import urllib.request

    deadline = time.time() + 60
    while time.time() < deadline:
        try:
            with urllib.request.urlopen("http://127.0.0.1:9000/minio/health/live", timeout=2):
                return
        except OSError:
            time.sleep(1)
    raise RuntimeError("MinIO did not come up within 60s")


def stop_minio() -> None:
    subprocess.run(["docker", "rm", "-f", MINIO_CONTAINER], capture_output=True)


async def ensure_bucket() -> None:
    from botocore.exceptions import ClientError

    from reliquary.infrastructure.storage import get_s3_client

    async with get_s3_client() as client:
        try:
            await client.head_bucket(Bucket=os.environ["R2_BUCKET_ID"])
        except ClientError:
            await client.create_bucket(Bucket=os.environ["R2_BUCKET_ID"])


async def conditional_put_preflight() -> dict:
    """The job, record and settlement stores are only correct if the bucket
    refuses a create over an existing key and a write against a stale ETag."""
    from reliquary.infrastructure.corpus_job_store import CorpusStoreConflict, _put

    key = f"reliquary/corpus/e2e-preflight/{secrets.token_hex(8)}.json"
    first = await _put(key, b"{}", None)
    result = {"create_over_existing_refused": False, "stale_etag_refused": False}
    try:
        await _put(key, b"{}", None)
    except CorpusStoreConflict:
        result["create_over_existing_refused"] = True
    second = await _put(key, b'{"v":2}', first)
    try:
        await _put(key, b'{"v":3}', first)
    except CorpusStoreConflict:
        result["stale_etag_refused"] = True
    result["fresh_etag_accepted"] = bool(second) and second != first
    return result


# --------------------------------------------------------------------------
# Declaration: the contract and the job
# --------------------------------------------------------------------------

def declare_task(state: Path, args, *, task_id: str, job_id: str, revision: str) -> dict:
    """The registry entry `jobs create` would write, built by the same code, so
    the contract this run serves under is the one production would carry."""
    from dataclasses import asdict

    from reliquary.cli.main import build_corpus_task_entry
    from reliquary.shared.task_registry import validate_entry

    entry = build_corpus_task_entry(
        task_id=task_id, job_id=job_id, from_profile=args.base_profile,
        model_id=args.honest_model, model_revision=revision,
        model_architecture=args.model_architecture, prompt_source=args.prompt_source,
        cap=args.cap, overrides={}, audit_params=audit_params_from_args(args),
    )
    validate_entry(entry)
    (state / "contract.json").write_text(json.dumps(entry.contract, sort_keys=True))
    (state / "entry.json").write_text(json.dumps(asdict(entry), indent=1, sort_keys=True, default=str))
    return dict(entry.contract)


async def declare_job(state: Path, args, *, job_id: str, revision: str, sha256: str,
                      eos: int, renderer_id: str, contract: dict) -> dict:
    from reliquary.cli.main import build_job_manifest
    from reliquary.infrastructure.corpus_job_store import write_job
    from reliquary.protocol.profiles import profile_from_contract

    sampling = contract["sampling"]
    manifest = build_job_manifest(
        job_id=job_id, checkpoint_repo=args.honest_model, checkpoint_revision=revision,
        checkpoint_sha256=sha256, prompt_source=args.prompt_source,
        prompt_count=args.prompt_count, renderer_id=renderer_id, eos_token_id=eos,
        slots_per_prompt=args.slots_per_prompt, temperature=sampling["temperature"],
        top_p=sampling["top_p"], top_k=sampling["top_k"], min_new_tokens=args.min_new_tokens,
        # A rehearsal may pass a short cap to run fast; a real job takes the
        # template's budget for the source, as `jobs create` does by default.
        max_new_tokens=(args.max_new_tokens or
                        contract["environments"][args.prompt_source]["max_new_tokens"]),
        n=args.n,
        grader_id=args.prompt_source, threshold=1.0,
        prompt_order="miner_walk", deadline_round=None,
        from_profile=profile_from_contract(contract),
    )
    await write_job(manifest, None)
    (state / "job.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# --------------------------------------------------------------------------
# Child: the validator
# --------------------------------------------------------------------------

def run_validator(args) -> None:
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION
    from reliquary.validator.corpus_validator import run_corpus_validator

    # The audit parameters live in the entry's params, as the registry would carry them.
    params = json.loads(Path(args.entry).read_text())["params"]
    entry = SimpleNamespace(task_id=args.task_id, job_id=args.job_id,
                            mechanism=MECHANISM_CORPUS_GENERATION, params=params)
    # Settlement is driven by the orchestrator once every verdict is in, so the
    # archive it checks is the only one written.
    asyncio.run(run_corpus_validator(
        entry=entry, wallet=None, netuid=0, signer_client=None,
        http_host="127.0.0.1", http_port=args.port, cap=args.cap,
        set_weights=False, settle_every_seconds=1e9,
    ))


# --------------------------------------------------------------------------
# Child: a miner
# --------------------------------------------------------------------------

class _TimedGenerator:
    def __init__(self, inner, eos: int) -> None:
        self._inner, self._eos = inner, eos
        self.seconds, self.tokens, self.completions, self.eos_terminated = 0.0, 0, 0, 0

    def generate(self, prompt_ids, n):
        started = time.perf_counter()
        generations = self._inner.generate(prompt_ids, n)
        self.seconds += time.perf_counter() - started
        for g in generations:
            self.tokens += len(g.tokens)
            self.completions += 1
            self.eos_terminated += int(bool(g.tokens) and g.tokens[-1] == self._eos)
        return generations


def _http_client(base_url: str):
    import httpx

    from reliquary.miner.corpus_miner import CorpusPermanentFailure, CorpusTransientFailure

    http = httpx.Client(base_url=base_url, timeout=120.0)

    def issue(call):
        try:
            response = call()
        except httpx.TransportError as exc:
            raise CorpusTransientFailure(str(exc)) from exc
        if response.status_code == 503:
            raise CorpusTransientFailure("503")
        if response.status_code >= 400:
            raise CorpusPermanentFailure(str(response.status_code), status=response.status_code,
                                         detail=response.text[:500])
        return response.json()

    class Client:
        def job(self):
            return issue(lambda: http.get("/corpus/job"))

        def cursor(self, hotkey):
            return int(issue(lambda: http.get(f"/corpus/cursor/{hotkey}"))["cursor"])

        def submit(self, body):
            return issue(lambda: http.post("/corpus/submit", json=body))

    return Client()


def run_miner(args) -> None:
    import bittensor as bt

    from reliquary.corpus.encoding import checkpoint_fingerprint
    from reliquary.corpus.job import parse_job
    from reliquary.miner.corpus_miner import VllmGenerator, mine_steps
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE, toploc_proof
    from reliquary.protocol.signatures import sign_corpus_submission
    from reliquary.shared.modeling import load_tokenizer
    from reliquary.validator.corpus_service import prompt_job_for_spec, renderer_for_job

    client = _http_client(args.validator_url)
    job = parse_job(client.job())
    job_directory = _snapshot(job.checkpoint_repo, job.checkpoint_revision)
    if checkpoint_fingerprint(job_directory) != job.checkpoint_sha256:
        raise SystemExit("the job's checkpoint does not match its fingerprint")
    # The dishonest miner generates with another model but everything it
    # submits (tokenizer, prompt, fingerprint) claims the job's checkpoint.
    generate_directory = _snapshot(args.model, args.revision) if args.model else job_directory
    tokenizer = load_tokenizer(str(job_directory))

    def encode(text):
        encoded = tokenizer.encode(text, add_special_tokens=False)
        return list(getattr(encoded, "ids", encoded))

    renderer = renderer_for_job(job, encode, tokenizer=tokenizer)
    prompts = prompt_job_for_spec(job)
    keypair = bt.Keypair.create_from_mnemonic(os.environ["CORPUS_E2E_MNEMONIC"])
    signer = SimpleNamespace(hotkey=keypair)
    proof = toploc_proof(ACTIVE_PROTOCOL_PROFILE)

    load_started = time.perf_counter()
    generator = _TimedGenerator(
        VllmGenerator(str(generate_directory), job.sampling, proof, job.eos_token_id,
                      gpu_memory_utilization=args.gpu_memory_utilization),
        job.eos_token_id,
    )
    load_seconds = time.perf_counter() - load_started
    started = time.perf_counter()
    def mine(steps):
        return mine_steps(
            job=job, hotkey=keypair.ss58_address, client=client, generator=generator,
            tokenizer=tokenizer, render=lambda i: renderer.initial_text(prompts.task_for(i)),
            sign=lambda body: sign_corpus_submission(signer, body), max_steps=steps,
        )

    if args.until_accepted:
        # Probation counts accepted submissions: keep stepping until exactly
        # that many landed, whatever the route refused on the way.
        counts: dict = {}
        for _ in range(3 * args.until_accepted):
            if counts.get("accepted", 0) >= args.until_accepted:
                break
            for reason, count in mine(1).items():
                counts[reason] = counts.get(reason, 0) + count
    else:
        counts = mine(args.steps)
    elapsed = time.perf_counter() - started
    Path(args.out).write_text(json.dumps({
        "hotkey": keypair.ss58_address, "model": str(generate_directory), "counts": counts,
        "load_seconds": round(load_seconds, 1), "loop_seconds": round(elapsed, 1),
        "generation_seconds": round(generator.seconds, 1), "completions": generator.completions,
        "completion_tokens": generator.tokens, "eos_terminated": generator.eos_terminated,
        "generation_tokens_per_second": round(generator.tokens / max(generator.seconds, 1e-9), 1),
    }, indent=1))


# --------------------------------------------------------------------------
# Orchestrator
# --------------------------------------------------------------------------

def _spawn(state: Path, env: dict, name: str, argv: list[str]) -> subprocess.Popen:
    log = open(state / f"{name}.log", "w")
    return subprocess.Popen([sys.executable, str(Path(__file__).resolve()), *argv],
                            env=env, stdout=log, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True)


def _wait_http(url: str, process: subprocess.Popen, timeout: float) -> None:
    import httpx

    deadline = time.time() + timeout
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"the validator exited with {process.returncode} before serving")
        try:
            if httpx.get(url, timeout=5).status_code == 200:
                return
        except httpx.TransportError:
            pass
        time.sleep(3)
    raise RuntimeError(f"{url} did not answer within {timeout:.0f}s")


def _stop(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _run_miner_process(state: Path, env: dict, args, role: str, mnemonic: str, steps: int,
                       model: str | None, revision: str | None, *, until_accepted: bool = False) -> dict:
    out = state / f"miner-{role}.json"
    # The mnemonic goes through the environment, not argv, which any process can list.
    env = {**env, "CORPUS_E2E_MNEMONIC": mnemonic}
    argv = ["miner", "--validator-url", f"http://127.0.0.1:{args.port}",
            "--steps", str(steps), "--out", str(out),
            "--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if model:
        argv += ["--model", model, "--revision", revision]
    if until_accepted:
        argv += ["--until-accepted", str(steps)]
    process = _spawn(state, env, f"miner-{role}", argv)
    code = process.wait(timeout=args.miner_timeout)
    if code != 0 or not out.exists():
        raise RuntimeError(f"the {role} miner exited with {code}; see {state}/miner-{role}.log")
    return json.loads(out.read_text())


async def _wait_verdicts(job_id: str, expected: int, timeout: float, validator) -> tuple[list, list]:
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    records = BucketRecordStore()
    deadline = time.time() + timeout
    while True:
        submitted = await records.list_submission_ids(job_id)
        judged = await records.list_verdict_ids(job_id)
        if len(submitted) >= expected and set(submitted) <= set(judged):
            break
        if validator.poll() is not None:
            raise RuntimeError(f"the validator exited with {validator.returncode} mid-audit")
        if time.time() > deadline:
            raise RuntimeError(f"{len(judged)}/{len(submitted)} verdicts after {timeout:.0f}s")
        await asyncio.sleep(5)
    subs = [await records.read_submission(job_id, sid) for sid in submitted]
    verdicts = [await records.read_verdict(job_id, sid) for sid in submitted]
    return subs, verdicts


def _toploc_summary(verdicts: list[dict]) -> dict:
    if not verdicts:
        return {}
    fields = ("worst_exp", "worst_mant_mean", "worst_mant_median")
    return {
        "max": {f: max(v[f] for v in verdicts) for f in fields},
        "min": {f: min(v[f] for v in verdicts) for f in fields},
        "reasons": sorted({str(v.get("reason")) for v in verdicts}),
    }


# A record audited this soon after its arrival was audited on arrival; any later
# one waited for its beacon round, a rescan or a backward audit.
ON_ARRIVAL_SECONDS = 30.0


def _audit_throughput(subs: list[dict], verdicts: list[dict]) -> dict:
    # The auditor drains a FIFO queue, so each verdict's busy time is from the
    # later of its arrival and the previous verdict to its own verdict. Only
    # records audited on arrival are counted: a drawn one, or one audited
    # later, waited (for a beacon round or a rescan) and that wait is not
    # audit time; an unaudited or voided record cost no GPU. Every verdict
    # still marks when the auditor was last busy.
    pairs = sorted(zip(subs, verdicts), key=lambda p: p[1]["audited_at"])
    busy, previous, tokens, counted, later = 0.0, None, 0, 0, 0
    for sub, verdict in pairs:
        start = sub["received_at"] if previous is None else max(previous, sub["received_at"])
        previous = verdict["audited_at"]
        if not verdict.get("audited", True):
            continue
        # A drawn record first waited for its beacon round and a rescan.
        if verdict.get("draw") or verdict["audited_at"] - sub["received_at"] > ON_ARRIVAL_SECONDS:
            later += 1
            continue
        busy += max(0.0, verdict["audited_at"] - start)
        tokens += int(sub["token_count"])
        counted += 1
    return {"audited_submissions": counted, "audited_later_not_counted": later,
            "completion_tokens": tokens, "busy_seconds": round(busy, 1),
            "completion_tokens_per_second": round(tokens / max(busy, 1e-9), 1)}


AUDIT_FLAGS = (
    ("audit_q", "audit_q"),
    ("audit_probation", "audit_probation_submissions"),
    ("audit_hold_seconds", "audit_hold_seconds"),
    ("audit_suspect_seconds", "audit_suspect_seconds"),
    ("audit_ban_after_failures", "audit_ban_after_failures"),
)


def audit_params_from_args(args) -> dict:
    """The task's `audit_*` params, as `jobs create` would declare them; none
    given means V0 (every submission audited)."""
    return {key: getattr(args, attr) for attr, key in AUDIT_FLAGS
            if getattr(args, attr, None) is not None}


def submission_rows(subs: list[dict], verdicts: list[dict], *, role_of: dict[str, str],
                    switch_at: float | None, settled: set[str]) -> list[dict]:
    """One row per accepted submission, in arrival order. The late cheater's
    rows are `pre_switch` or `switched` by arrival against the moment its
    honest process ended; `paid` is a passing verdict the settlement consumed."""
    rows = []
    for sub, verdict in sorted(zip(subs, verdicts), key=lambda p: p[0]["received_at"]):
        role = role_of.get(sub["hotkey"], "unknown")
        phase = None
        if role == "late":
            phase = "switched" if switch_at is not None and sub["received_at"] >= switch_at else "pre_switch"
        sid = verdict["submission_id"]
        rows.append({
            "submission_id": sid, "role": role, "phase": phase,
            "received_at": sub["received_at"], "audited_at": verdict["audited_at"],
            "token_count": int(sub["token_count"]),
            "audited": bool(verdict.get("audited", True)), "draw": verdict.get("draw"),
            "passed": bool(verdict["passed"]), "reason": verdict.get("reason"),
            "worst_exp": verdict.get("worst_exp"),
            "paid": bool(verdict["passed"]) and sid in settled,
        })
    return rows


def _caught_by(first: dict, rows: list[dict], *, q: float, hold: float,
               probation: int | None = None) -> str:
    """Why the late cheater's first failed record was audited at all."""
    if first["phase"] != "switched":
        return "pre_switch"
    if first["draw"] and first["draw"].get("drawn"):
        return "draw"
    passes = sum(1 for r in rows if r["role"] == "late" and r["audited"] and r["passed"]
                 and r["audited_at"] < first["audited_at"])
    if probation is not None and passes < probation:
        return "probation"
    # Not drawn yet audited while sampled: the hotkey was below 1/q per hold
    # (spec §7.4), or the beacon could not be fetched (audit, fail safe).
    t = first["audited_at"]
    recent = sum(1 for r in rows if r["role"] == "late" and t - hold < r["received_at"] <= t)
    return "slow_hotkey" if recent < 1.0 / q else "no_beacon"


def partial_audit_checks(rows: list[dict], miner_states: dict[str, dict], *,
                         q: float, hold: float, probation: int | None = None) -> tuple[dict, dict]:
    """(report, checks) for a run with an honest miner, a late cheater that is
    honest through probation then switches model, and optionally an immediate
    cheater. `miner_states` maps a role to its miners.json fields plus
    `effective_state` at the end of the run."""
    def of(role, phase=None):
        return [r for r in rows if r["role"] == role and (phase is None or r["phase"] == phase)]

    checks: dict[str, bool] = {}
    honest, pre, post = of("honest"), of("late", "pre_switch"), of("late", "switched")
    checks["honest_all_passed_and_paid"] = bool(honest) and all(r["passed"] and r["paid"] for r in honest)
    checks["sampling_exercised"] = any(r["passed"] and not r["audited"] for r in honest)
    checks["late_pre_switch_all_paid"] = bool(pre) and all(r["passed"] and r["paid"] for r in pre)
    checks["late_post_switch_none_paid"] = bool(post) and not any(r["passed"] or r["paid"] for r in post)

    failed = sorted((r for r in of("late") if r["audited"] and not r["passed"]),
                    key=lambda r: r["audited_at"])
    first = failed[0] if failed else None
    checks["late_caught"] = first is not None
    report: dict = {"detection": None}
    state = miner_states.get("late") or {}
    if first is None:
        checks["late_suspect_at_first_failure"] = False
        checks["late_held_records_all_audited"] = False
        checks["late_no_pass_after_detection"] = False
    else:
        t = first["audited_at"]
        failures = state.get("confirmed_failures") or []
        # The state is written before the verdict, so its first confirmed
        # failure is no later than the first failed verdict.
        checks["late_suspect_at_first_failure"] = (
            state.get("effective_state") in ("suspect", "banned") and bool(failures)
            and min(failures) <= t)
        held = [r for r in of("late") if r["received_at"] <= t < r["audited_at"]]
        checks["late_held_records_all_audited"] = all(r["audited"] for r in held)
        checks["late_no_pass_after_detection"] = not any(
            r["passed"] for r in of("late") if r["audited_at"] >= t)
        report["detection"] = {
            "submission_id": first["submission_id"], "audited_at": t, "draw": first["draw"],
            "caught_by": _caught_by(first, rows, q=q, hold=hold, probation=probation),
            "post_switch_before_detection": sum(1 for r in post if r["received_at"] < first["received_at"]),
            "held_at_detection": len(held),
            "held_at_detection_audited": sum(r["audited"] for r in held),
        }
    immediate = of("dishonest")
    if immediate:
        checks["immediate_cheater_none_paid"] = not any(r["passed"] or r["paid"] for r in immediate)
    report["by_role"] = {
        f"{role}{'/' + phase if phase else ''}": {
            "submissions": len(rs), "audited": sum(r["audited"] for r in rs),
            "drawn": sum(bool(r["draw"] and r["draw"].get("drawn")) for r in rs),
            "passed_unaudited": sum(r["passed"] and not r["audited"] for r in rs),
            "passed": sum(r["passed"] for r in rs), "paid": sum(r["paid"] for r in rs),
            "voided_banned": sum(r["reason"] == "banned" for r in rs),
        }
        for role, phase, rs in (("honest", None, honest), ("late", "pre_switch", pre),
                                ("late", "switched", post), ("dishonest", None, immediate))
        if rs
    }
    return report, checks


async def _read_miner_states(job_id: str, params: dict, role_of: dict[str, str]) -> dict:
    from reliquary.corpus.audit_policy import AuditParams, MinerState, effective_state
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore

    document, _ = await BucketRecordStore().read_miners(job_id)
    audit = AuditParams.from_params(params)
    now = time.time()
    states = {}
    for hotkey, fields in (document or {}).items():
        state = MinerState.from_dict(fields)
        states[role_of.get(hotkey, hotkey)] = {
            "hotkey": hotkey, "effective_state": effective_state(state, now, audit),
            **{k: v for k, v in state.to_dict().items() if k != "mant_mean_history"},
            "mant_mean_history_len": len(state.mant_mean_history),
        }
    return states


def _route_ok(counts: dict, steps: int, *, may_be_banned: bool) -> bool:
    # A cheater may be banned mid-run; the route then refuses it, nothing else.
    allowed = {"accepted", "miner_banned"} if may_be_banned else {"accepted"}
    return set(counts) <= allowed and sum(counts.values()) == steps


def orchestrate(args) -> int:
    _refuse_production_bucket()
    _refuse_busy_card(args.allow_busy_gpu)
    audit_params = audit_params_from_args(args)
    if args.late_cheater_steps and not args.audit_probation:
        raise SystemExit("--late-cheater-steps needs --audit-probation: it mines honestly that many times")
    summary: dict = {"ok": False, "checks": {}, "audit_params": audit_params}
    checks = summary["checks"]
    stamp = time.strftime("%Y%m%d%H%M%S", time.gmtime())
    job_id, task_id = f"e2e-{stamp}", f"corpus-e2e-{stamp}"
    state = Path(args.state_dir) / stamp
    state.mkdir(parents=True, exist_ok=True)
    os.environ["RELIQUARY_TASK_ID"] = task_id
    validator = None
    if args.start_minio:
        start_minio()
    try:
        asyncio.run(ensure_bucket())
        preflight = asyncio.run(conditional_put_preflight())
        summary["conditional_put"] = preflight
        if not all(preflight.values()):
            raise RuntimeError(f"the bucket does not honour conditional PUT: {preflight}")

        from reliquary.corpus.encoding import checkpoint_fingerprint
        from reliquary.shared.modeling import load_tokenizer

        honest_dir = _snapshot(args.honest_model, args.honest_revision)
        honest_revision = honest_dir.name
        sha256 = checkpoint_fingerprint(honest_dir)
        eos = load_tokenizer(str(honest_dir)).eos_token_id
        contract = declare_task(state, args, task_id=task_id, job_id=job_id, revision=honest_revision)
        params = json.loads((state / "entry.json").read_text())["params"]
        renderer_id = args.renderer or contract["environments"][args.prompt_source]["prompt_template"]["id"]
        manifest = asyncio.run(declare_job(
            state, args, job_id=job_id, revision=honest_revision, sha256=sha256, eos=eos,
            renderer_id=renderer_id, contract=contract,
        ))
        dishonest_revision = _snapshot(args.dishonest_model, args.dishonest_revision).name
        summary["contract_proofs"] = contract["proofs"]
        summary.update({"job_id": job_id, "task_id": task_id, "state_dir": str(state),
                        "checkpoint": f"{args.honest_model}@{honest_revision}",
                        "checkpoint_sha256": sha256, "eos_token_id": eos,
                        "dishonest_model": f"{args.dishonest_model}@{dishonest_revision}",
                        "prompt_source": args.prompt_source, "sampling": manifest["sampling"]})

        env = _child_env(state, task_id)
        validator = _spawn(state, env, "validator", [
            "validator", "--task-id", task_id, "--job-id", job_id,
            "--port", str(args.port), "--cap", str(args.cap), "--entry", str(state / "entry.json")])
        started = time.time()
        _wait_http(f"http://127.0.0.1:{args.port}/corpus/job", validator, args.validator_timeout)
        summary["validator_start_seconds"] = round(time.time() - started, 1)

        import bittensor as bt

        mnemonics = {role: bt.Keypair.generate_mnemonic() for role in ("honest", "late", "dishonest")}
        miners, route = {}, {}
        miners["honest"] = _run_miner_process(state, env, args, "honest", mnemonics["honest"],
                                              args.honest_steps, None, None)
        route["honest"] = _route_ok(miners["honest"]["counts"], args.honest_steps, may_be_banned=False)
        switch_at = None
        if args.late_cheater_steps:
            # One hotkey, two processes: honest through probation, then the
            # other model under the same mnemonic, from where its cursor stands.
            miners["late_pre_switch"] = _run_miner_process(
                state, env, args, "late-pre-switch", mnemonics["late"], args.audit_probation, None, None,
                until_accepted=True)
            switch_at = time.time()
            miners["late_switched"] = _run_miner_process(
                state, env, args, "late-switched", mnemonics["late"], args.late_cheater_steps,
                args.dishonest_model, dishonest_revision)
            # Exactly `probation` accepted, so the switch lands right as it ends.
            route["late_pre_switch"] = (
                miners["late_pre_switch"]["counts"].get("accepted") == args.audit_probation)
            route["late_switched"] = _route_ok(miners["late_switched"]["counts"],
                                               args.late_cheater_steps, may_be_banned=True)
        if args.dishonest_steps:
            miners["dishonest"] = _run_miner_process(
                state, env, args, "dishonest", mnemonics["dishonest"], args.dishonest_steps,
                args.dishonest_model, dishonest_revision)
            route["dishonest"] = _route_ok(miners["dishonest"]["counts"], args.dishonest_steps,
                                           may_be_banned=True)
        summary["miners"] = miners
        summary["switch_at"] = switch_at
        for name, ok in route.items():
            checks[f"route_{name}"] = ok
        accepted = sum(m["counts"].get("accepted", 0) for m in miners.values())
        role_of = {m["hotkey"]: name.split("_")[0] for name, m in miners.items()}

        subs, verdicts = asyncio.run(_wait_verdicts(job_id, accepted, args.audit_timeout, validator))
        summary["audit_throughput"] = _audit_throughput(subs, verdicts)
        summary["verdicts"] = {}
        for role in sorted(set(role_of.values())):
            vs = [v for v in verdicts if role_of.get(v["hotkey"]) == role]
            audited = [v for v in vs if v.get("audited", True)]
            summary["verdicts"][role] = {
                "passed": sum(bool(v["passed"]) for v in vs),
                "failed": sum(not v["passed"] for v in vs), "toploc": _toploc_summary(audited)}

        summary["archive"] = asyncio.run(_settle_and_read(task_id, job_id, args.cap))
        settled = set(summary["archive"].pop("settled_ids", []))
        rows = submission_rows(subs, verdicts, role_of=role_of, switch_at=switch_at, settled=settled)
        summary["submissions"] = rows
        summary["miner_states"] = asyncio.run(_read_miner_states(job_id, params, role_of))
        report, audit_checks = partial_audit_checks(
            rows, summary["miner_states"], q=float(params.get("audit_q", 1.0)),
            hold=float(params.get("audit_hold_seconds", 4320.0)),
            probation=int(params.get("audit_probation_submissions", 100)))
        summary["partial_audit"] = report
        if not args.late_cheater_steps:
            # No late cheater to judge: keep the checks that still apply.
            audit_checks = {k: v for k, v in audit_checks.items()
                            if k in ("honest_all_passed_and_paid", "immediate_cheater_none_paid")
                            or (k == "sampling_exercised" and params.get("audit_q", 1.0) < 1.0)}
        checks.update(audit_checks)

        rewards = summary["archive"].get("rewards_by_hotkey") or {}
        paid_tokens: dict[str, int] = {}
        for r in rows:
            if r["paid"]:
                hotkey = next(h for h, role in role_of.items() if role == r["role"])
                paid_tokens[hotkey] = paid_tokens.get(hotkey, 0) + r["token_count"]
        total = sum(paid_tokens.values())
        checks["archive_pays_paid_records_by_tokens"] = set(rewards) == set(paid_tokens) and all(
            abs(rewards[h] - args.cap * t / total) < 1e-9 for h, t in paid_tokens.items())
        checks["archive_sums_to_cap"] = abs(sum(rewards.values()) - args.cap) < 1e-9
        checks["second_settlement_is_a_noop"] = summary["archive"]["second_settle"] is None

        summary["export"] = _export(state, env, job_id)
        checks["export_rows"] = summary["export"]["rows"] == sum(r["paid"] for r in rows) * args.n
        checks["export_only_paid_hotkeys"] = summary["export"]["hotkeys"] == sorted(paid_tokens)
        summary["ok"] = all(checks.values())
    except Exception as exc:
        logger.exception("the end-to-end run failed")
        summary["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        _stop(validator)
        if args.start_minio and not args.keep_minio:
            stop_minio()
        (state / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True))
        print(json.dumps(summary, indent=1, sort_keys=True), flush=True)
    return 0 if summary["ok"] else 1


async def _settle_and_read(task_id: str, job_id: str, cap: float) -> dict:
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.infrastructure.storage import dataset_object_key, download_json
    from reliquary.validator.corpus_settlement import CorpusSettler, R2Archives

    records = BucketRecordStore()
    settler = CorpusSettler(task_id=task_id, job_id=job_id, cap=cap,
                            records=records, archives=R2Archives())
    window = await settler.settle_once()
    if window is None:
        return {"window": None, "second_settle": None, "settled_ids": []}
    key = dataset_object_key(window, task_id)
    archive = await download_json(key, strict=True)
    return {"window": window, "key": key, "rewards_by_hotkey": archive["rewards_by_hotkey"],
            "sum": sum(archive["rewards_by_hotkey"].values()),
            "window_status": archive.get("window_status"),
            "second_settle": await settler.settle_once(),
            "settled_ids": (await records.read_settlement(job_id))[0].get("settled") or []}


def _export(state: Path, env: dict, job_id: str) -> dict:
    counts = {}
    for name, flags in (("all", ["--apply-filter"]), ("accepted", ["--apply-filter", "--only-accepted"])):
        out = state / f"export-{name}.jsonl"
        subprocess.run([sys.executable, "-m", "reliquary.cli.main", "jobs", "export", job_id,
                        "--out", str(out), *flags], env=env, check=True, capture_output=True,
                       stdin=subprocess.DEVNULL, timeout=600)
        counts[name] = [json.loads(line) for line in out.read_text().splitlines()]
    rows = counts["all"]
    return {"rows": len(rows), "accepted_by_filter": len(counts["accepted"]),
            "hotkeys": sorted({r["hotkey"] for r in rows}),
            "prompts": len({r["prompt_index"] for r in rows})}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="role")

    run = sub.add_parser("run")
    for p in (parser, run):
        p.add_argument("--start-minio", action="store_true")
        p.add_argument("--keep-minio", action="store_true")
        p.add_argument("--allow-busy-gpu", action="store_true")
        p.add_argument("--state-dir", default=str(ROOT / "e2e-state"))
        p.add_argument("--honest-model", default="Qwen/Qwen3-4B-Base")
        p.add_argument("--honest-revision", default=None)
        p.add_argument("--dishonest-model", default="Qwen/Qwen3-4B")
        p.add_argument("--dishonest-revision", default=None)
        p.add_argument("--base-profile", default="qwen3-4b-reliquary-logic-v8-dev1")
        p.add_argument("--model-architecture", default="Qwen3ForCausalLM")
        p.add_argument("--prompt-source", default="reliquarylogic_v1")
        p.add_argument("--renderer", default=None,
                       help="e.g. chat-template-thinking-v1; omit for the contract's own template")
        p.add_argument("--prompt-count", type=int, default=200)
        p.add_argument("--slots-per-prompt", type=int, default=4)
        p.add_argument("--n", type=int, default=4)
        p.add_argument("--min-new-tokens", type=int, default=16)
        p.add_argument("--max-new-tokens", type=int, default=None,
                       help="omit for the template's budget; a short value only speeds a rehearsal")
        p.add_argument("--honest-steps", type=int, default=20)
        p.add_argument("--dishonest-steps", type=int, default=5)
        p.add_argument("--cap", type=float, default=0.1)
        p.add_argument("--port", type=int, default=18080)
        p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
        p.add_argument("--validator-timeout", type=float, default=900)
        p.add_argument("--miner-timeout", type=float, default=3600)
        p.add_argument("--audit-timeout", type=float, default=1800)
        # Partial audit (spec 2026-09-25); none given keeps V0's full audit.
        p.add_argument("--audit-q", type=float, default=None)
        p.add_argument("--audit-probation", type=int, default=None)
        p.add_argument("--audit-hold-seconds", type=float, default=None)
        p.add_argument("--audit-suspect-seconds", type=float, default=None)
        p.add_argument("--audit-ban-after-failures", type=int, default=None)
        p.add_argument("--late-cheater-steps", type=int, default=0,
                       help="a hotkey honest for --audit-probation steps, then this many with the dishonest model")

    v = sub.add_parser("validator")
    v.add_argument("--task-id", required=True)
    v.add_argument("--job-id", required=True)
    v.add_argument("--port", type=int, required=True)
    v.add_argument("--cap", type=float, required=True)
    v.add_argument("--entry", required=True)

    m = sub.add_parser("miner")
    m.add_argument("--validator-url", required=True)
    m.add_argument("--steps", type=int, required=True)
    m.add_argument("--out", required=True)
    m.add_argument("--model", default=None)
    m.add_argument("--revision", default=None)
    m.add_argument("--gpu-memory-utilization", type=float, default=None)
    m.add_argument("--until-accepted", type=int, default=None)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.role == "validator":
        run_validator(args)
        return 0
    if args.role == "miner":
        run_miner(args)
        return 0
    return orchestrate(args)


if __name__ == "__main__":
    sys.exit(main())
