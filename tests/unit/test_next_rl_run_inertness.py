"""Phase 1 of the next RL run is inert for legacy RL tasks and for corpus jobs.

The global constraint of the branch: the whole ``service-contract/v2`` mechanism is OFF for a task that is
not one, so a legacy task keeps its archive bytes, its ``/state`` bytes and its weight replay, and a corpus
job never touches any of it. Golden files already pin the bytes (``test_legacy_archive_golden``, the
``/state`` golden of ``test_service_server_state``, ``test_service_weight_replay``); this file proves the
other half, that the new code does not RUN:

* every entry point of the service stack is armed with a tripwire (``tests/unit/rl_tripwires.py``): it raises
  AND records the call, and the test fails afterwards if one was recorded, even when a caller swallowed it;
* the legacy and corpus suites are rerun, in a child pytest, with those tripwires armed;
* a cold import of every legacy entry point loads no ``reliquary.services`` module at all;
* the legacy boot, the legacy window loop, the legacy weight replay (against numbers the tree BEFORE this
  work produced) and the legacy miner engine are driven with the tripwires armed.
"""
from __future__ import annotations

import ast
import asyncio
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import reliquary.validator.service as service_module
from tests.unit.rl_tripwires import ARMED, RUNTIME_ONLY, real_entry, rl_service_tripwires  # noqa: F401  (autouse in this module)

REPO = Path(__file__).parents[2]
ROOT = REPO / "reliquary"
GOLDEN_REPLAY = Path(__file__).parent / "data" / "legacy_weight_replay_adce8966.json"
# The service stack: nothing in a corpus job or a legacy task may import it at load time.
SERVICE_STACK_PREFIXES = ("reliquary.services", "reliquary.protocol.service_")


# ------------------------------------------------------------------ nothing imports the stack

def _imported_modules(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def test_corpus_modules_never_import_the_rl_service_stack():
    paths = sorted((ROOT / "validator").glob("corpus_*.py")) + sorted((ROOT / "corpus").rglob("*.py"))
    assert len(paths) > 20, "the corpus module list is empty: the test would prove nothing"
    for path in paths:
        leaked = sorted(m for m in _imported_modules(path) if m.startswith(SERVICE_STACK_PREFIXES))
        assert not leaked, f"{path.relative_to(REPO)} imports the RL service stack: {leaked}"


def test_a_cold_import_of_every_legacy_entry_point_loads_no_service_module():
    """Fresh interpreter: the validator, batcher, admission, server, weight-only validator, the miner engine and
    every corpus module are importable without loading one ``reliquary.services`` module."""
    code = (
        "import glob, importlib, os, sys\n"
        "sys.path.insert(0, os.getcwd())\n"
        "names = ['reliquary.validator.service', 'reliquary.validator.batcher', 'reliquary.validator.admission',\n"
        "         'reliquary.validator.server', 'reliquary.validator.weight_only', 'reliquary.miner.engine']\n"
        "names += ['reliquary.validator.' + os.path.basename(p)[:-3] for p in glob.glob('reliquary/validator/corpus_*.py')]\n"
        "names += ['reliquary.corpus.' + os.path.basename(p)[:-3] for p in glob.glob('reliquary/corpus/*.py')\n"
        "          if not p.endswith('__init__.py')]\n"
        "for name in names:\n"
        "    importlib.import_module(name)\n"
        "print(len(names), sorted(m for m in sys.modules if m.startswith('reliquary.services')))\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", code], cwd=REPO, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr[-2000:]
    count, loaded = done.stdout.strip().split(" ", 1)
    assert int(count) > 30 and loaded == "[]", done.stdout


# ------------------------------------------------------------------ the legacy and corpus suites, armed

LEGACY_AND_CORPUS_SUITES = (
    # legacy RL: archive bytes, batcher, admission, money, weight replay, miner
    "test_legacy_archive_golden.py", "test_archive_window_content.py", "test_grpo_window_batcher.py",
    "test_deferred_proof.py", "test_service_v2.py", "test_bounded_fill_service.py", "test_v6_emission.py",
    "test_admission_budget_refund.py", "test_admission_lane_isolation.py", "test_episode_admission_replay.py",
    "test_weight_only_validator.py",
    "test_miner_engine_v2.py",
    # corpus jobs: the validator entry, the service, the registry, the end-to-end seam
    "test_corpus_validator.py", "test_corpus_service.py", "test_corpus_multi_job_service.py",
    "test_corpus_end_to_end.py", "test_corpus_task_registry.py",
)


def _child_env() -> dict:
    """Only what the suites need: no RELIQUARY_* switch (a profile or an operator's setting must not leak in)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def _limit_child_memory() -> None:
    """The same address-space cap as the parent run (common rules: ulimit -v 16000000 KiB)."""
    import resource
    limit = 16_000_000 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft == resource.RLIM_INFINITY or soft > limit:
        resource.setrlimit(resource.RLIMIT_AS, (limit, hard))


@pytest.mark.skipif(os.environ.get("RELIQUARY_PROTOCOL_PROFILE", "default") not in ("", "default"),
                    reason="the armed legacy/corpus suites run once, under the default protocol profile only")
@pytest.mark.parametrize("suite", LEGACY_AND_CORPUS_SUITES)
def test_the_legacy_and_corpus_suites_pass_with_every_service_entry_point_armed(suite):
    """The suite runs in a child pytest with ``-p tests.unit.rl_tripwires``: an autouse fixture arms every entry
    point of the service stack for each of its tests. A test that reaches one fails there."""
    assert (Path(__file__).parent / suite).exists()
    done = subprocess.run(
        [sys.executable, "-m", "pytest", f"tests/unit/{suite}", "-q", "-p", "no:cacheprovider",
         "-p", "tests.unit.rl_tripwires"],
        cwd=REPO, capture_output=True, text=True, timeout=600, env=_child_env(), preexec_fn=_limit_child_memory)
    tail = (done.stdout + done.stderr)[-3000:]
    assert done.returncode == 0, tail
    assert " passed" in done.stdout and "skipped" not in done.stdout.splitlines()[-1], tail


def test_the_tripwires_really_fire_and_record(rl_service_tripwires):
    """Guard of the guard: EVERY armed entry point is refused AND recorded (so a caller that swallows the error
    is still caught), and the runtime-only ones are recorded only on an instance that has a runtime."""
    from tests.unit.rl_tripwires import _resolve

    for module_name, dotted in ARMED:
        owner, attr = _resolve(module_name, dotted)
        armed = owner.__dict__[attr]
        function = armed.__func__ if isinstance(armed, (staticmethod, classmethod)) else armed
        before = len(rl_service_tripwires)
        with pytest.raises(AssertionError, match="ran for a legacy"):
            function()
        assert len(rl_service_tripwires) == before + 1, f"{module_name}.{dotted} did not record"
        assert rl_service_tripwires[-1] == f"{module_name}.{dotted}"
    rl_service_tripwires.clear()

    for module_name, dotted in RUNTIME_ONLY:
        owner, attr = _resolve(module_name, dotted)
        with pytest.raises(AssertionError, match="with a service runtime"):
            owner.__dict__[attr](SimpleNamespace(_service_runtime=object()))
        assert rl_service_tripwires == [f"{module_name}.{dotted}"]
        rl_service_tripwires.clear()
        result = owner.__dict__[attr](SimpleNamespace(_service_runtime=None, _signer_client=None))   # legacy: the real early return
        if inspect.iscoroutine(result):
            assert asyncio.run(result) is None
        assert rl_service_tripwires == []
    assert len(ARMED) >= 27 + 36 and len(RUNTIME_ONLY) == 3


# ------------------------------------------------------------------ the legacy boot

def _legacy_service(**kwargs):
    from reliquary.validator.service import ValidationService
    from tests.unit.test_service_v2 import _LateDropFakeEnv, _LateDropFakeWallet

    tokenizer = MagicMock()
    tokenizer.eos_token_id = 99
    return ValidationService(wallet=_LateDropFakeWallet(), model=MagicMock(), tokenizer=tokenizer,
                             env=_LateDropFakeEnv(), netuid=99, **kwargs)


def test_booting_a_legacy_task_builds_no_runtime_publisher_schedule_store_or_advice(monkeypatch, tmp_path):
    # Every service switch an operator could have left in the environment is ON; a legacy task ignores them.
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_BUCKET", "fake-public-bucket")
    monkeypatch.setenv("RELIQUARY_SERVICE_QUALIFICATION", str(tmp_path / "qualification.json"))
    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path / "state"))
    for signer in (None, object()):        # a remote signer + a bucket is a refusal for a SERVICE run only
        svc = _legacy_service(use_drand=False, signer_client=signer)
        assert svc._service_runtime is None and svc._service_schedule_store is None
        assert svc._candidate_service_window is None and svc._candidate_service_pools is None
        assert svc._service_sealed_windows == set() and svc._service_recovery_attempts == {}
        assert svc._service_recovery_context == {}
        assert svc.server.service_health_redaction is False
        svc._require_publication_signer()
        svc._start_observation_publication()            # needs a loop to schedule a task: it must return first
        assert getattr(svc, "_observation_publisher", None) is None and getattr(svc, "_observation_task", None) is None
        svc._refresh_service_active()                    # a legacy service has no runtime to ask
        assert not (tmp_path / "state").exists(), "a legacy boot created a service state folder"


@pytest.mark.asyncio
async def test_a_legacy_task_runs_real_windows_with_every_service_switch_on(monkeypatch, tmp_path):
    from tests.unit.test_legacy_archive_golden import ROOT as CHECKPOINT
    from tests.unit.test_service_v2 import _build_late_drop_service
    from tests.unit.test_service_window_build import _journal, _run

    monkeypatch.setattr(service_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr("reliquary.constants.EMISSION_PRICE_ARMED", False)
    monkeypatch.setenv("RELIQUARY_OBSERVATIONS_BUCKET", "fake-public-bucket")
    svc = _build_late_drop_service()
    svc._derive_randomness = AsyncMock(return_value=("drand-material", None))
    svc._checkpoint_store = SimpleNamespace(current_manifest=lambda: SimpleNamespace(
        repo_id="models/test", revision=CHECKPOINT, checkpoint_n=0))
    archives = _journal(svc, monkeypatch, tmp_path)
    assert await _run(svc, monkeypatch, windows=2) == [0, 1, 2]
    pending = archives.pending_archives(start_window=1, end_window=2)
    assert sorted(pending) == [1, 2]
    assert not any(key.startswith("service_") for archive in pending.values() for key in archive)
    assert getattr(svc, "_observation_publisher", None) is None and getattr(svc, "_observation_task", None) is None
    assert svc._service_runtime is None and svc._candidate_service_window is None


# ------------------------------------------------------------------ the wire of a legacy task

def test_a_legacy_submission_carries_no_service_field_and_has_no_policy():
    from reliquary.services.admission_policy import validate_submission_policy
    from tests.unit.test_batch_submission_schema import _valid_rollouts
    from reliquary.protocol.submission import BatchSubmissionRequest, GrpoBatchState

    request = BatchSubmissionRequest(miner_hotkey="hk" * 24, prompt_idx=42, window_start=1000,
                                     merkle_root="00" * 32, rollouts=_valid_rollouts(k=4),
                                     checkpoint_hash="sha256:test")
    assert request.pool_selection is None and request.service_binding is None
    assert validate_submission_policy(request, None) is None
    assert not {"pool_selection", "service_binding"} & set(request.model_dump())
    assert not {"pool_selection", "service_binding"} & set(json.loads(request.model_dump_json()))
    assert GrpoBatchState.model_fields["service_policy"].default is None


def test_the_legacy_cooldown_horizon_is_unchanged():
    from reliquary.validator.cooldown import CooldownMap

    cooldown = CooldownMap(cooldown_windows=1_000_000)
    cooldown.record_batched(1, 5)
    assert cooldown.is_in_cooldown(1, 6) and cooldown.is_in_cooldown(1, 1_000_004)
    assert not cooldown.is_in_cooldown(1, 1_000_005) and cooldown.export_state() == {1: 5}


# ------------------------------------------------------------------ the legacy weight replay

def test_a_legacy_only_stream_replays_to_the_numbers_the_tree_before_this_work_produced():
    """The golden file holds the stream AND the weights ``WeightOnlyValidator._replay_ema`` returned at
    adce8966 (the merge base, no service code) for it: per-task caps engaged, an aborted window, pruned
    dust and a status-less old archive. The same stream goes through today's chain unchanged.

    How the golden was generated (to regenerate or audit it): ``git archive adce8966 reliquary | tar -x -C <empty dir>``,
    then with that tree first on ``sys.path`` call ``WeightOnlyValidator._replay_ema(archives, caps=caps)`` (and once
    with no ``caps`` for ``weights_uncapped``) on the ``archives`` / ``caps`` stored in this very file's JSON; the
    two weight maps are what that old tree returned. Never regenerate it from today's tree."""
    from reliquary.validator.weight_only import WeightOnlyValidator

    golden = json.loads(GOLDEN_REPLAY.read_text())
    archives, caps = golden["archives"], golden["caps"]
    declared = {task: SimpleNamespace(mechanism="auction", service_contract=None, params={"cap": cap})
                for task, cap in caps.items()}
    validated = WeightOnlyValidator._validated_service_archives(archives, declared)
    assert sorted(map(id, validated)) == sorted(map(id, archives))     # the very same objects, untouched (only ordered)
    for source in (archives, validated):
        capped = WeightOnlyValidator._replay_ema(source, caps=caps)
        assert capped == golden["weights"] and json.dumps(capped, sort_keys=True) == json.dumps(golden["weights"], sort_keys=True)
        assert WeightOnlyValidator._replay_ema(source) == golden["weights_uncapped"]
        explicit = WeightOnlyValidator._replay_ema(source, caps=caps, service_tasks=frozenset())
        assert explicit == golden["weights"]
    assert sum(golden["weights_uncapped"].values()) > sum(golden["weights"].values())  # the caps were engaged


def test_a_service_task_declared_but_silent_leaves_a_legacy_stream_byte_identical():
    """A registry that DECLARES a v2 task next to the legacy ones changes nothing for a stream without its windows."""
    from reliquary.validator.weight_only import WeightOnlyValidator
    from tests.unit.test_service_weight_replay import TASK, declared
    # Declaring a v2 task parses its contract (that is what a declaration is); nothing else may run.
    with real_entry("reliquary.protocol.service_contract", "ServiceContract.from_dict"):
        golden = json.loads(GOLDEN_REPLAY.read_text())
        registry = declared(extra={task: SimpleNamespace(mechanism="auction", service_contract=None, params={"cap": cap})
                             for task, cap in golden["caps"].items()})
        validated = WeightOnlyValidator._validated_service_archives(golden["archives"], registry)
        assert sorted(map(id, validated)) == sorted(map(id, golden["archives"]))
        assert WeightOnlyValidator._replay_ema(validated, caps=golden["caps"],
                                               service_tasks=frozenset({TASK})) == golden["weights"]


def test_the_weight_only_validator_reads_a_legacy_task_with_the_legacy_projection(monkeypatch):
    """Item 13's bounds are fetch options of a SERVICE task only: a legacy task is asked for what it always was."""
    from reliquary.validator import weight_only
    from tests.unit.test_service_weight_replay import TASK, declared, wire

    seen = {}

    async def recent(current_window, n, *, task_id=None, fields=None, **kw):
        seen[task_id] = (fields, kw)
        return []

    with real_entry("reliquary.protocol.service_contract", "ServiceContract.from_dict"):
        wov, _ = wire(monkeypatch, [], [], declared())
    monkeypatch.setattr(weight_only.storage, "list_recent_datasets", recent)
    import asyncio
    with real_entry("reliquary.protocol.service_contract", "ServiceContract.from_dict"):
        asyncio.run(wov.submit_once())            # parses the declared v2 contract, nothing else of the stack
    assert seen["default"] == (("window_start", "window_status", "rewards_by_hotkey"), {})
    assert seen[TASK][1] and "number_map_fields" in seen[TASK][1]


# ------------------------------------------------------------------ the legacy miner

def test_a_legacy_miner_with_no_observation_source_mines_as_before(monkeypatch):
    import asyncio

    import reliquary.miner.engine as engine_module
    from reliquary.constants import M_ROLLOUTS
    from tests.unit.test_observation_client import _Rng, bare_engine

    for name in ("URL", "RUN_ID", "VALIDATOR_HOTKEY", "DIR"):
        monkeypatch.delenv(f"RELIQUARY_OBSERVATIONS_{name}", raising=False)
    engine = bare_engine()
    engine._observations = engine._prompt_policy = None
    offloaded = []
    monkeypatch.setattr(engine_module.asyncio, "to_thread", lambda *a, **k: offloaded.append(a))
    engine._configure_observations()                     # ObservationClient.__init__ is a tripwire: never built
    assert engine._observations is None and engine._prompt_policy is None
    for state in (SimpleNamespace(service_policy=None, checkpoint_n=3), SimpleNamespace(service_policy=object(), checkpoint_n=3)):
        options = asyncio.run(engine._cooldown_options(state, _Rng()))
        assert len(options) == 1 and options[0] is engine._cooldown_per_env     # the engine's own cooldown set, nothing merged
    assert offloaded == []                                # no sync of any observation table
    asked = []
    group = engine.choose_public_seed_group(SimpleNamespace(group_size=M_ROLLOUTS),
                                            lambda seeds: asked.append(tuple(seeds)) or "g", problem=None, env=None)
    assert group == "g" and asked == [tuple(range(M_ROLLOUTS))]                # seeds 0..M-1, as ever
