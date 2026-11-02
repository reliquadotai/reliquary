"""Eval jobs on a bridged env other than reliquary-swe (TMax's `tmax_sft` here): the job
names the env with `episode.bridge` (the gateway's options and image lock), and every
side reads the env through `BridgedEpisodeSource`, as an RL run does. Signed only, eval
only; nothing changes for reliquary-swe jobs.

The bridged env is faked: row ``i`` has prompt ``"tmax prompt {i}"`` and id ``"tmax-{i}"``."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.corpus.job import JobError, parse_job
from reliquary.environment import agentic_swe
from reliquary.eval import prompt_source as ps
from reliquary.eval import rows as eval_rows
from reliquary.eval.prompt_source import eval_source_for
from reliquary.sandbox.tasks import ResolvedTask
from tests.unit.test_corpus_job_episode import COMMIT, _manifest
from tests.unit.test_corpus_job_signed_sandbox import sandbox_spec, signed_episode

PACKAGE = "reliquary-terminal"
SPLIT = "tmax_sft"
ENV_PACKAGE = "reliquary-terminal==0.3.1+gcd4df2dcc1aaf175.vfb2"
OPTIONS = {"tools": ["bash"], "splits": {SPLIT: {"split": SPLIT}}}
LOCK = "1" * 64
BRIDGE = {"options": OPTIONS, "image_lock_sha256": LOCK}
INDICES = (17, 3, 9)
IMAGE = "tmax-base@sha256:" + "a" * 64
SET_ID = "reliquary-terminal-tmax-sft-k3"


class FakeBridged:
    """What the signed path reads of a `BridgedEpisodeSource`."""

    identity, options_sha256, lock_sha256 = ENV_PACKAGE, "o" * 64, LOCK

    def __init__(self, rows=50):
        self.rows, self.resolved = rows, []

    def __len__(self):
        return self.rows

    def prompt(self, index):
        return f"tmax prompt {index}"

    def instance_id(self, index):
        return f"tmax-{index}"

    async def resolve(self, index):
        self.resolved.append(index)
        return ResolvedTask(IMAGE, {}, self.options_sha256, None, 4000, self.lock_sha256)


def terminal_episode(**overrides):
    episode = signed_episode(sandbox=sandbox_spec(env=PACKAGE, env_package=ENV_PACKAGE, tools=["bash"]),
                             bridge=BRIDGE)
    episode["env"] = {"package": PACKAGE, "version": COMMIT, "split": SPLIT, "num_images": 1}
    episode.update(overrides)
    return episode


@pytest.fixture
def bridged(monkeypatch):
    fake = FakeBridged()
    opened = []

    def open_bridged(package, version, split, options_json, lock):
        opened.append((package, split, json.loads(options_json), lock))
        return fake

    monkeypatch.setattr(agentic_swe, "_bridged_sources", open_bridged)
    monkeypatch.setattr(agentic_swe.importlib.metadata, "version", lambda name: "0.3.1")
    monkeypatch.setattr(ps, "_loaded", {})
    eval_rows._row_map.cache_clear()
    yield SimpleNamespace(source=fake, opened=opened)
    eval_rows._row_map.cache_clear()


def _set_rows(ids=None):
    return [{"problem_id": f"p{i}", "set_id": SET_ID, "env": PACKAGE, "interaction": "signed_bridged",
             "split": SPLIT, "source_index": index, "instance_id": (ids or {}).get(index, f"tmax-{index}")}
            for i, index in enumerate(INDICES)]


def _eval_job(monkeypatch, ids=None, **episode):
    body = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in _set_rows(ids))
    monkeypatch.setattr(ps, "FETCHERS", [lambda source: body])
    source = eval_source_for(SET_ID, body, len(INDICES)).name
    return parse_job(_manifest(prompt_source=source, prompt_count=len(INDICES),
                               episode=terminal_episode(**episode)))


EVAL_SOURCE = "eval-set:tmax:3:" + "e" * 64


# -- the manifest ----------------------------------------------------------------------

def test_a_terminal_eval_job_parses_and_round_trips():
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=terminal_episode()))
    assert job.episode.is_bridged and job.episode.bridge == BRIDGE
    assert job.episode.sandbox.tools == ("bash",)
    assert job.to_contract()["episode"]["bridge"] == BRIDGE
    assert parse_job(job.to_contract()) == job


def test_a_swe_job_carries_no_bridge():
    job = parse_job(_manifest(episode=signed_episode()))
    assert not job.episode.is_bridged and "bridge" not in job.to_contract()["episode"]
    with pytest.raises(JobError, match="episode.bridge is only for"):
        parse_job(_manifest(episode=signed_episode(bridge=BRIDGE)))


@pytest.mark.parametrize("change, why", [
    (dict(prompt_source="reliquary_agentic_swe_v1"), "split must be 'train'"),
    (dict(episode=terminal_episode(execution=None)), "only in an eval job on the signed sandbox"),
    (dict(episode={k: v for k, v in terminal_episode().items() if k != "bridge"}), "needs episode.bridge"),
    (dict(episode=terminal_episode(bridge={"options": OPTIONS, "image_lock_sha256": "x"})), "64-hex"),
    (dict(episode=terminal_episode(bridge={"options": {}, "image_lock_sha256": LOCK})), "options"),
    (dict(episode=terminal_episode(bridge={**BRIDGE, "extra": 1})), "unknown fields"),
])
def test_a_terminal_job_is_signed_eval_and_bridged_only(change, why):
    raw = _manifest(prompt_source=EVAL_SOURCE, episode=terminal_episode())
    raw.update(change)
    if raw["episode"].get("execution") is None:
        raw["episode"] = {k: v for k, v in raw["episode"].items() if k not in ("execution", "sandbox")}
    with pytest.raises(JobError, match=why):
        parse_job(raw)


def test_the_bash_harness_runs_terminal_but_suite_does_not():
    raw = _manifest(prompt_source=EVAL_SOURCE, episode=terminal_episode(harness="suite"))
    with pytest.raises(JobError):
        parse_job(raw)


# -- the source ------------------------------------------------------------------------

def test_the_signed_source_reads_the_bridged_env_through_the_sets_rows(bridged, monkeypatch):
    source = agentic_swe.job_signed_source(_eval_job(monkeypatch))
    assert source.split == SPLIT
    assert source.prompt(0) == "tmax prompt 17" and source.instance_id(2) == "tmax-9"
    assert source.task_ref(1).index == 3
    assert bridged.opened == [(PACKAGE, SPLIT, OPTIONS, LOCK)]     # built with the gateway's options


def test_a_stale_set_is_refused(bridged, monkeypatch):
    with pytest.raises(ValueError, match="gone"):
        agentic_swe.job_signed_source(_eval_job(monkeypatch, ids={3: "gone"}))


def test_support_refuses_a_bridge_whose_tools_are_not_the_sessions(bridged, monkeypatch):
    from reliquary import sandbox as sandbox_support

    job = _eval_job(monkeypatch, bridge={"options": {**OPTIONS, "tools": ["bash", "edit"]},
                                         "image_lock_sha256": LOCK})
    monkeypatch.setattr(sandbox_support, "require_sandbox", lambda: None)
    monkeypatch.setattr(sandbox_support, "sandbox_commit_refusal", lambda commit: None)
    monkeypatch.setattr(agentic_swe.importlib, "import_module", lambda name: None)
    monkeypatch.setattr(agentic_swe, "installed_env_package", lambda package: ENV_PACKAGE)
    assert "tools" in agentic_swe.sandbox_support_refusal(job.episode)
    good = _eval_job(monkeypatch)
    assert agentic_swe.sandbox_support_refusal(good.episode) is None


def test_support_names_the_package_it_checks(monkeypatch):
    job = parse_job(_manifest(prompt_source=EVAL_SOURCE, episode=terminal_episode()))
    monkeypatch.setattr(agentic_swe, "harness_pin_refusal", lambda *a, **k: None)
    monkeypatch.setattr(agentic_swe, "_renderers_version", lambda: agentic_swe.RENDERERS_VERSION)
    asked = []
    monkeypatch.setattr(agentic_swe, "_dist_commit", lambda name: asked.append(name) or "f" * 40)
    refusal = agentic_swe.episode_support_refusal(job.episode, need_verifiers=False)
    assert asked == [PACKAGE] and refusal.startswith(f"{PACKAGE} is installed at")


# -- placement, intake and miner -------------------------------------------------------

def test_a_session_places_the_bridged_task(bridged, monkeypatch, tmp_path):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.validator.sandbox_wiring import wire_signed_job
    from tests.unit.test_signed_grading_and_wiring import _services

    job = _eval_job(monkeypatch)
    _, services = _services(tmp_path)

    class Router:
        async def slots_remaining(self, index):
            return 2

    w = SimpleNamespace(job=job, is_banned=None)
    wire_signed_job(w, services=services, routes=lambda: SimpleNamespace(routers={job.job_id: Router()},
                                                                         retired=set()),
                    checkpoint_dir="/ck", tokenizer=None, vocab_size=None, chunk_tokens=32,
                    intake_factory=lambda job, **kw: SimpleNamespace(source=None))
    engagement = {"kind": "corpus", "job_id": job.job_id, "prompt_index": 0}
    terms = asyncio.run(services.issuer._engagements["corpus"].terms("5Hot", engagement))
    assert (terms.env, terms.split, terms.index, terms.image) == (PACKAGE, SPLIT, 17, IMAGE)
    assert bridged.source.resolved == [17]


def _intake(job, monkeypatch, *, final_diff, state_sha256):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.validator import signed_intake

    opened = SimpleNamespace(tools_version=next(iter(signed_intake.TOOLS_VERSIONS)), tools=("bash",),
                             env_package=ENV_PACKAGE)
    final = SimpleNamespace(state_sha256=state_sha256, reward=1.0)
    claims = SimpleNamespace(expires_at=10**12, session_id="s", machine_id="m", hotkey="5Hot")
    monkeypatch.setattr(signed_intake, "verify_transcript", lambda *a, **k: SimpleNamespace(
        ok=True, reasons=[], claims=claims, open=opened, final=final, calls=[]))
    monkeypatch.setattr(signed_intake, "check_turn_spans", lambda *a: SimpleNamespace(ok=True))
    monkeypatch.setattr(signed_intake, "parse_signed_trajectory", lambda *a, **k: None)
    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda episode, **kw: None)
    monkeypatch.setattr(agentic_swe, "sandbox_support_refusal", lambda episode, **kw: None)
    monkeypatch.setattr(agentic_swe, "job_turn_renderer", lambda *a, **kw: None)
    intake = signed_intake.build_signed_episode_intake(
        job, checkpoint_dir="/ck", tokenizer=None, vocab_size=None, chunk_tokens=32,
        directory=lambda now: object(), token_verifier=None, sessions=None)
    monkeypatch.setattr(intake, "_prompt_refusal", lambda request: ([1, 2], None))
    monkeypatch.setattr(intake, "_shape_refusal", lambda *a: "past the state check")
    request = SimpleNamespace(trajectory=SimpleNamespace(transcript={}, tokens=[1, 2, 3], turns=[],
                                                         stop="agent_completed", final_diff=final_diff),
                              completions=[], prompt_index=1, miner_hotkey="5Hot")
    return intake.check(request, received=0.0)


def test_the_intake_takes_no_state_from_a_bridged_env(bridged, monkeypatch):
    job = _eval_job(monkeypatch)
    # The gateway hashed its archive; the miner submits nothing, and that is not a mismatch.
    assert _intake(job, monkeypatch, final_diff="", state_sha256="d" * 64) == "past the state check"
    refused = _intake(job, monkeypatch, final_diff="a diff", state_sha256="d" * 64)
    assert refused.reason == "sandbox_state_mismatch"


def test_the_miner_submits_no_state_for_a_bridged_env(bridged, monkeypatch):
    pytest.importorskip("reliquary_sandbox.attest")
    from reliquary.miner import signed_episode
    from tests.unit.test_signed_miner import HOT, FakeSessions

    job = _eval_job(monkeypatch)
    miner = signed_episode.SignedSweEpisodeRunner(
        job=job, hotkey=HOT, sign_binding=lambda binding: "sig", sessions=FakeSessions(),
        model_name="m", renderer_model_dir="/ck", generate_url="http://127.0.0.1:1",
        sampling=job.sampling)
    assert miner.requires_text_state is False and miner._source.prompt(2) == "tmax prompt 9"
    assert signed_episode.SignedSweEpisodeRunner.requires_text_state is True   # SWE jobs unchanged


# -- the set and the eval job ----------------------------------------------------------

def test_a_set_freezes_references_to_the_bridged_env(monkeypatch, tmp_path):
    from reliquary.eval import sets

    fake = FakeBridged()
    card = sets.build_source_set(PACKAGE, out=tmp_path / "set", split=SPLIT, start=0, count=50,
                                 sample=3, seed=7, bridge=BRIDGE,
                                 open_bridged=lambda *a: fake)
    rows = [json.loads(line) for line in (tmp_path / "set" / "prompts.jsonl").read_text().splitlines()]
    assert card["interaction"] == "signed_bridged" and card["split"] == SPLIT
    assert card["bridge"] == {**BRIDGE, "options_sha256": fake.options_sha256, "env_package": ENV_PACKAGE}
    assert [row["instance_id"] for row in rows] == [f"tmax-{row['source_index']}" for row in rows]
    assert ps.set_interaction(rows) == "signed_bridged"
    with pytest.raises(ValueError, match="bridge"):
        sets.build_source_set(PACKAGE, out=tmp_path / "other", split=SPLIT, count=3)


def test_the_grade_scheme_is_the_signed_sandboxs():
    from reliquary.eval.grading import EPISODE_SCHEMES

    assert EPISODE_SCHEMES["signed_bridged"] == EPISODE_SCHEMES["swe"] == "signed-sandbox"


def _card(**over):
    card = {"interaction": "signed_bridged", "split": SPLIT, "source": PACKAGE,
            "bridge": {**BRIDGE, "options_sha256": "o" * 64, "env_package": ENV_PACKAGE}}
    card.update(over)
    return card


@pytest.mark.parametrize("episode, why", [
    (terminal_episode(), None),
    (terminal_episode(execution="replay"), "signed sandbox"),
    (terminal_episode(bridge={**BRIDGE, "image_lock_sha256": "2" * 64}), "bridge"),
    (terminal_episode(sandbox=sandbox_spec(env=PACKAGE, env_package=ENV_PACKAGE.replace("cd4", "000"),
                                           tools=["bash"])), "frozen from"),
])
def test_jobs_create_holds_the_episode_to_the_set(episode, why):
    from reliquary.cli.main import _refuse_episode_disagreeing_with_set

    if why is None:
        _refuse_episode_disagreeing_with_set(_card(), SET_ID, episode, episode["renderer"])
        return
    with pytest.raises(ValueError, match=why):
        _refuse_episode_disagreeing_with_set(_card(), SET_ID, episode, episode["renderer"])


def test_jobs_create_refuses_another_package():
    from reliquary.cli.main import _refuse_episode_disagreeing_with_set

    episode = terminal_episode()
    with pytest.raises(ValueError, match="environment"):
        _refuse_episode_disagreeing_with_set(_card(source="reliquary-other"), SET_ID, episode,
                                             episode["renderer"])


def test_the_contract_statement_is_an_agentic_body():
    from reliquary.protocol.agentic_source import (
        AGENTIC_BODIES,
        AGENTIC_SIGNED_ENVIRONMENT,
        AGENTIC_SWE_BODY,
        AGENTIC_SWE_ENVIRONMENT,
    )

    assert AGENTIC_BODIES[AGENTIC_SWE_ENVIRONMENT] is AGENTIC_SWE_BODY      # unchanged
    assert AGENTIC_BODIES[AGENTIC_SIGNED_ENVIRONMENT].environment_contract_id == "reliquary/agentic-signed/v1"
