"""Signed episodes are inert for legacy RL tasks, single-turn v2 orders and corpus jobs.

What this file proves, behaviourally:

* a cold import of every legacy, single-turn v2 and corpus entry point loads no signed-episode module;
* every signed-episode entry point is a tripwire (``tests/unit/rl_tripwires.py``: ``EPISODE_ARMED`` raise and record,
  ``EPISODE_CONDITIONAL`` record when a shared function takes its episode branch). The legacy and corpus suites
  run under ALL tripwires (``test_next_rl_run_inertness``); here the single-turn v2 suites, the suites of the
  shared modules signed episodes edited and the corpus miner / sandbox suites rerun in a child pytest under the signed-episode
  ones (``-p tests.unit.episode_tripwires``: they legitimately run the phase 1 stack);
* what a legacy task, a single-turn v2 order and a corpus job put on the wire and on disk is byte for byte what the
  tree before signed episodes (1c6930f4) produced (``tests/unit/legacy_wire_probe.py``, golden
  ``tests/unit/data/legacy_wire_1c6930f4.json``), with ONE known exception pinned exactly: a remote proof worker's
  result now carries three ``episode_stop_*`` keys (null for every legacy proof);
* the corpus validator keeps refusing RL engagements, its routes, its session keys and store prefix.

The tests choose their tripwires explicitly (``legacy_wires`` / ``episode_wires``) instead of the brief's module-wide
autouse import: the single-turn v2 probe runs the phase 1 runtime, which the legacy set refuses by design.
"""
from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.unit import rl_tripwires
from tests.unit.test_next_rl_run_inertness import LEGACY_AND_CORPUS_SUITES, _child_env, _limit_child_memory

REPO = Path(__file__).parents[2]
GOLDEN = Path(__file__).parent / "data" / "legacy_wire_1c6930f4.json"
DEFAULT_PROFILE = os.environ.get("RELIQUARY_PROTOCOL_PROFILE", "default") in ("", "default")

# Every module signed-episode added (the precommit wire, the RL session engagements and routes, the admission and intake
# of a group, the validator wiring, the environment, the miner). ``reliquary.protocol.episode_retry`` is not one of
# them on purpose: the legacy server imports it (a frozenset of stage names, it imports nothing).
EPISODE_MODULES = (
    "reliquary.protocol.service_episode", "reliquary.sandbox.rl_engagements", "reliquary.sandbox.rl_routes",
    "reliquary.validator.episode_admission", "reliquary.validator.episode_intake",
    "reliquary.validator.rl_sandbox_wiring", "reliquary.environment.signed_episode", "reliquary.miner.forced_draw",
    "reliquary.miner.episode_group_miner", "reliquary.miner.episode_commit", "reliquary.miner.rl_episode_client",
    "reliquary.miner.episode_mining",
)


@pytest.fixture
def legacy_wires(monkeypatch):
    """Every tripwire, phase 1 and signed-episode: a legacy RL task or a corpus job."""
    calls: list[str] = []
    rl_tripwires.arm_legacy(monkeypatch, calls)
    yield calls
    assert not calls, f"RL service entry points were reached: {sorted(set(calls))}"


@pytest.fixture
def episode_wires(monkeypatch):
    """The signed-episode tripwires only: a single-turn v2 order (which runs the phase 1 stack)."""
    calls: list[str] = []
    rl_tripwires.arm_episode(monkeypatch, calls)
    yield calls
    assert not calls, f"signed-episode entry points were reached: {sorted(set(calls))}"


def _golden() -> dict:
    return json.loads(GOLDEN.read_text())


# ------------------------------------------------------------------ nothing loads an episode module

def test_every_plan_2c_module_exists_and_is_listed():
    for name in EPISODE_MODULES:
        assert (REPO / (name.replace(".", "/") + ".py")).exists(), name


def test_a_cold_import_of_the_legacy_single_turn_and_corpus_entry_points_loads_no_episode_module():
    code = (
        "import glob, importlib, os, sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "names = ['reliquary.validator.service', 'reliquary.validator.batcher', 'reliquary.validator.admission',\n"
        "         'reliquary.validator.server', 'reliquary.validator.weight_only', 'reliquary.miner.engine',\n"
        "         'reliquary.validator.corpus_validator', 'reliquary.shared.training_payload',\n"
        "         'reliquary.validator.verifier', 'reliquary.validator.toploc_check', 'reliquary.sandbox.sessions',\n"
        "         'reliquary.sandbox.routes', 'reliquary.infrastructure.sandbox_store',\n"
        "         'reliquary.validator.sandbox_wiring', 'reliquary.validator.remote_proof',\n"
        "         'reliquary.validator.remote_proof_protocol', 'reliquary.validator.remote_proof_server',\n"
        "         'reliquary.protocol.submission', 'reliquary.protocol.signatures', 'reliquary.protocol.seed_pool',\n"
        "         'reliquary.protocol.service_contract', 'reliquary.protocol.service_submission',\n"
        "         'reliquary.protocol.toploc_proof', 'reliquary.environment.registry',\n"
        "         'reliquary.services.runtime', 'reliquary.services.admission_policy',\n"
        "         'reliquary.miner.corpus_generate_server', 'reliquary.miner.signed_episode',\n"
        "         'reliquary.miner.agentic_episode', 'reliquary.trainer.publisher', 'reliquary.cli.main']\n"
        "names += ['reliquary.validator.' + os.path.basename(p)[:-3] for p in glob.glob('reliquary/validator/corpus_*.py')]\n"
        "names += ['reliquary.corpus.' + os.path.basename(p)[:-3] for p in glob.glob('reliquary/corpus/*.py')\n"
        "          if not p.endswith('__init__.py')]\n"
        "for name in names:\n"
        "    importlib.import_module(name)\n"
        f"print(len(names), sorted(m for m in sys.modules if m in {EPISODE_MODULES!r}))\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    count, loaded = done.stdout.strip().splitlines()[-1].split(" ", 1)
    assert int(count) > 60 and loaded == "[]", done.stdout


# ------------------------------------------------------------------ the admission's interaction binding

def test_an_episode_v1_environment_still_takes_episode_v1_and_only_it(legacy_wires):
    from reliquary.environment.registry import ENVIRONMENT_SPECS
    from reliquary.validator.admission import submission_interaction_matches
    from tests.unit.test_episode_protocol import _episode_commit

    names = [name for name, spec in ENVIRONMENT_SPECS.items() if spec.interaction_mode == "episode"]
    assert names, "no Episode v1 environment is registered: the test would prove nothing"
    commit = _episode_commit()
    for name in names:
        assert submission_interaction_matches(SimpleNamespace(rollouts=[SimpleNamespace(commit=commit)]), name)
        signed = copy.deepcopy(commit)
        signed["rollout"]["episode"] = {"schema_version": "reliquary/signed-episode/v1"}
        assert not submission_interaction_matches(SimpleNamespace(rollouts=[SimpleNamespace(commit=signed)]), name)
        plain = {"rollout": {"prompt_length": 1}}
        assert not submission_interaction_matches(SimpleNamespace(rollouts=[SimpleNamespace(commit=plain)]), name)


def test_a_single_turn_rollout_still_takes_no_episode(legacy_wires):
    from reliquary.environment.registry import ENVIRONMENT_SPECS
    from reliquary.validator.admission import submission_interaction_matches
    from tests.unit.test_episode_protocol import _episode_commit

    single = [name for name, spec in ENVIRONMENT_SPECS.items() if spec.interaction_mode == "single_turn"]
    assert "reliquary_dapo_math_v1" in single
    plain = {"rollout": {"prompt_length": 1}}
    for episode in ({"schema_version": "reliquary/signed-episode/v1"}, _episode_commit()["rollout"]["episode"], {}):
        with_episode = {"rollout": {"prompt_length": 1, "episode": episode}}
        for name in single:
            assert submission_interaction_matches(SimpleNamespace(rollouts=[SimpleNamespace(commit=plain)]), name)
            assert not submission_interaction_matches(
                SimpleNamespace(rollouts=[SimpleNamespace(commit=with_episode)]), name)


def test_no_registered_environment_is_a_signed_episode_one():
    """Signed episodes add the mode, not an env: every registered env keeps its mode and manifest (golden below)."""
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    assert {spec.interaction_mode for spec in ENVIRONMENT_SPECS.values()} <= {"single_turn", "episode"}


def test_a_legacy_admission_context_carries_no_episode_field(legacy_wires):
    import dataclasses

    from reliquary.validator.admission import AdmissionContext, PreparedSubmission
    from reliquary.validator.server import _episode_admission_fields

    defaults = {f.name: f.default for f in dataclasses.fields(AdmissionContext)}
    assert (defaults["signed_episode"], defaults["episode_max_tokens"]) == (False, None)
    assert {f.name: f.default for f in dataclasses.fields(PreparedSubmission)}["episode_pending"] is False
    for policy in (None, "not-a-dict"):
        assert _episode_admission_fields(SimpleNamespace(service_policy=policy, env=SimpleNamespace(name="x"))) == {}


def test_a_single_turn_v2_admission_context_carries_no_episode_field(episode_wires):
    """The server's per-batcher admission context of a single-turn v2 env (no suite builds it through the server):
    the contract is read, the env has no episode block, so the context keeps its phase 1 fields."""
    from reliquary.validator.server import _episode_admission_fields
    from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2_dict

    policy = {"contract": contract_v2_dict()}
    for env in (MATH, CODE, "not_in_the_order"):
        assert _episode_admission_fields(SimpleNamespace(service_policy=policy, env=SimpleNamespace(name=env))) == {}


# ------------------------------------------------------------------ the corpus validator

def test_the_corpus_validator_still_refuses_rl_engagements_and_keeps_its_session_keys(legacy_wires):
    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox.sessions import EngagementTerms, RlPrecommitEngagements

    refusal = asyncio.run(RlPrecommitEngagements().terms("5Hot", {"kind": "rl_precommit"}))
    assert refusal.reason == "engagement_kind_unsupported"
    assert sandbox_store.session_key("s-1", 1_800_000_000).startswith("reliquary/sandbox/sessions/")
    assert sandbox_store.SESSION_PREFIX == "reliquary/sandbox/sessions/"
    assert sandbox_store.R2SessionStore()._prefix == sandbox_store.SESSION_PREFIX
    terms = EngagementTerms(engagement="corpus:job:1", env="e", split="s", index=1, checkpoint="c", image="i",
                            env_package="p", budgets={})
    assert terms.exclusive is False and terms.still_valid is None


def test_the_corpus_sandbox_wiring_is_the_corpus_one(legacy_wires, tmp_path):
    """The corpus validator's sandbox services, built by its own wiring: corpus and the refusing RL stub as the
    engagement kinds, only the corpus routes, the corpus store prefix, and no RL session index ever filled."""
    from reliquary.sandbox.sessions import CorpusEngagements, RlPrecommitEngagements
    from tests.unit.test_signed_grading_and_wiring import _services

    _, services = _services(tmp_path)
    engagements = services.issuer._engagements
    assert set(engagements) == {"corpus", "rl_precommit"}
    assert isinstance(engagements["corpus"], CorpusEngagements)
    assert isinstance(engagements["rl_precommit"], RlPrecommitEngagements)
    assert {route.path for route in services.router.routes} == {
        "/corpus/sandbox/sessions", "/corpus/sandbox/sessions/{session_id}/close"}
    asyncio.run(services.start())
    assert services.book._by_precommit == {}


def test_a_corpus_session_never_enters_the_rl_precommit_index(legacy_wires):
    """``SessionBook.add`` parses every record's engagement as an RL one; a corpus engagement is never
    one, so a corpus book's index stays empty and its bookkeeping is the corpus one."""
    from reliquary.corpus.signed_reasons import corpus_engagement
    from reliquary.sandbox.sessions import SandboxPolicy, SessionBook, SessionRecord

    book = SessionBook(SandboxPolicy())
    for job, index in (("job-7", 3), ("rl", 1), ("rl:5:" + "a" * 64, 2)):
        book.add(SessionRecord(session_id=f"s-{index}", hotkey="5Hot", request_id=f"r-{index}",
                               engagement_sha256="a" * 64, kind="corpus", engagement=corpus_engagement(job, index),
                               env="swe", split="train", index=index, checkpoint="c" * 40, job_id=job,
                               prompt_index=index, machine_id="m-1", issued_at=1_800_000_000,
                               expires_at=1_800_003_600, token_sha256="b" * 64))
    assert book._by_precommit == {}
    assert len(book.records()) == 3


# ------------------------------------------------------------------ bytes against the tree before signed episodes

def test_a_payload_without_signed_episodes_keeps_its_header(legacy_wires):
    from tests.unit.test_training_payload_codec import _payload_bytes

    with np.load(io.BytesIO(_payload_bytes()), allow_pickle=False) as npz:
        header = json.loads(bytes(npz["header"]))
    assert "rollout_checkpoints" not in header


needs_default_profile = pytest.mark.skipif(
    not DEFAULT_PROFILE, reason="the golden was captured under the default protocol profile")


@needs_default_profile
@pytest.mark.parametrize("section", ["payload", "submission", "signature_messages", "environments",
                                     "corpus_sessions"])
def test_legacy_and_corpus_bytes_are_the_ones_before_plan_2c(section, legacy_wires):
    """Training payload (header, every array, the decoded legacy and historical Episode v1 payloads), the
    submission wire (and its length cap), the message each legacy commit's signature is checked against, the
    registered environments' manifests, the corpus session documents and keys."""
    from tests.unit import legacy_wire_probe

    assert getattr(legacy_wire_probe, section)() == _golden()[section]


@needs_default_profile
def test_a_single_turn_v2_order_keeps_its_bytes(episode_wires):
    """Its contract digest, the announcement (no episode capability), its observations, events and envelope."""
    from tests.unit import legacy_wire_probe

    now = legacy_wire_probe.single_turn_v2()
    assert now == _golden()["single_turn_v2"]
    assert "signed-sandbox-episode/v1" not in now["supported_capabilities"]


@needs_default_profile
def test_a_legacy_proof_request_is_unchanged(legacy_wires):
    from tests.unit import legacy_wire_probe

    assert legacy_wire_probe.proof_wire()["proof_input"] == _golden()["proof_wire"]["proof_input"]


EPISODE_PROOF_KEYS = ("episode_stop_cdf_miss", "episode_stop_first_bad_turn", "episode_stop_picks_ok")


@needs_default_profile
@pytest.mark.parametrize("which", ["proof_values", "proof_values_toploc"])
def test_a_legacy_proof_result_crosses_the_remote_wire(which, legacy_wires):
    """Today's validator reads what a worker before signed episodes wrote, into the same verdict, and a worker of today writes
    those very bytes back: the three ``episode_stop_*`` keys are omitted while None, so a controller before signed episodes
    (``extra="forbid"``) still accepts a legacy proof result."""
    from reliquary.validator.remote_proof_protocol import ProofValues, canonical_bytes
    from tests.unit import legacy_wire_probe

    old = _golden()["proof_wire"][which]
    values = ProofValues.read(old.encode())
    kernel = values.to_kernel()
    assert (kernel.episode_stop_picks_ok, kernel.episode_stop_first_bad_turn, kernel.episode_stop_cdf_miss) == (
        None, None, None)
    assert canonical_bytes(values.model_dump()).decode() == old
    assert legacy_wire_probe.proof_wire()[which] == old


# ------------------------------------------------------------------ the suites, rerun under the signed-episode tripwires

# Suites that never declare a signed-episode env: the single-turn v2 order, the shared modules signed episodes edited
# (payload, submission wire and signatures, remote proof, TOPLOC, verifier, Episode v1, profiles) and the corpus
# miner / sandbox. The legacy and corpus suites of ``test_next_rl_run_inertness`` run under ALL tripwires there.
EPISODE_ARMED_SUITES = (
    # single-turn v2 orders
    "test_service_runtime_v2.py", "test_service_contract_v2.py", "test_service_admission_policy.py",
    "test_service_policy_admission.py", "test_service_submission.py", "test_seed_pool.py",
    "test_service_server_state.py", "test_service_weight_replay.py", "test_service_window_build.py",
    "test_service_exploration_lane.py", "test_service_settlement_v2.py", "test_service_final_verdicts.py",
    "test_service_async_verify.py",
    # shared modules
    "test_training_payload_codec.py", "test_batch_submission_schema.py", "test_signatures.py",
    "test_envelope_signature.py", "test_remote_proof_controller.py", "test_remote_proof_toploc.py",
    "test_toploc_check.py", "test_toploc_verdict.py", "test_forced_seed_verifier.py", "test_episode_protocol.py",
    "test_profile_golden_digests.py",
    # corpus miner and sandbox
    "test_corpus_generate_server.py", "test_sandbox_sessions.py", "test_sandbox_routes.py",
    "test_signed_grading_and_wiring.py", "test_signed_intake.py", "test_signed_miner.py",
    "test_corpus_signature.py",
)


def test_the_armed_suite_lists_do_not_overlap():
    assert not set(EPISODE_ARMED_SUITES) & set(LEGACY_AND_CORPUS_SUITES)
    assert len(set(EPISODE_ARMED_SUITES)) == len(EPISODE_ARMED_SUITES)


@pytest.mark.skipif(not DEFAULT_PROFILE,
                    reason="the armed suites run once, under the default protocol profile only")
@pytest.mark.parametrize("suite", EPISODE_ARMED_SUITES)
def test_the_suites_pass_with_every_episode_entry_point_armed(suite):
    assert (Path(__file__).parent / suite).exists()
    done = subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/unit/{suite}", "-q", "-p", "no:cacheprovider",
         "-p", "tests.unit.episode_tripwires"],
        cwd=REPO, capture_output=True, text=True, timeout=600, env=_child_env(), preexec_fn=_limit_child_memory)
    tail = (done.stdout + done.stderr)[-3000:]
    assert done.returncode == 0, tail
    assert " passed" in done.stdout, tail


def test_the_episode_tripwires_really_fire_and_record(episode_wires):
    """Guard of the guard: every signed-episode entry point is refused AND recorded; a conditional one runs for real when
    its episode branch is not taken (nothing recorded) and is recorded when it is."""
    from reliquary.protocol.service_contract import supported_v2_capabilities
    from reliquary.shared import training_payload
    from reliquary.validator import verifier
    from tests.unit.service_v2_fixtures import contract_v2

    for module_name, dotted in rl_tripwires.EPISODE_ARMED:
        owner, attr = rl_tripwires._resolve(module_name, dotted)
        armed = owner.__dict__[attr]
        function = armed.__func__ if isinstance(armed, (staticmethod, classmethod)) else armed
        before = len(episode_wires)
        with pytest.raises(AssertionError, match="ran for a legacy"):
            function()
        assert episode_wires[before:] == [f"{module_name}.{dotted}"]
    episode_wires.clear()

    # Not taken: the real answer, nothing recorded.
    assert verifier.signed_episode_spans({"prompt_length": 1}, 4) is None
    assert training_payload._is_signed_episode({"episode": {"schema_version": "reliquary/episode/v1"}}) is False
    assert "signed-sandbox-episode/v1" not in supported_v2_capabilities(contract_v2())
    assert episode_wires == []
    # Taken: recorded and refused.
    signed = {"prompt_length": 1, "episode": {"schema_version": "reliquary/signed-episode/v1",
                                              "assistant_spans": [[1, 3]]}}
    with pytest.raises(AssertionError, match="took its episode branch"):
        verifier.signed_episode_spans(signed, 4)
    with pytest.raises(AssertionError, match="took its episode branch"):
        training_payload._is_signed_episode(signed)
    assert episode_wires == ["reliquary.validator.verifier.signed_episode_spans",
                             "reliquary.shared.training_payload._is_signed_episode"]
    episode_wires.clear()
    assert len(rl_tripwires.EPISODE_ARMED) >= 36 and len(rl_tripwires.EPISODE_CONDITIONAL) >= 16
