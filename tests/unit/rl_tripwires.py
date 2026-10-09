"""Tripwires on every entry point of the next-RL-run service stack (phase 1) and of signed episodes.

A legacy RL task and a corpus job must never reach them; a single-turn v2 order never reaches the signed-episode ones. ``rl_service_tripwires`` is autouse where it is
imported: it replaces each entry point by a wrapper that RAISES and RECORDS the call, and fails the test
afterwards if anything was recorded (a caller that swallows the exception does not hide it).

Used two ways: ``from tests.unit.rl_tripwires import rl_service_tripwires`` in a test module (armed for that
module only), and ``pytest -p tests.unit.rl_tripwires <legacy test file>`` (``test_next_rl_run_inertness``
reruns the legacy and corpus suites that way). ``tests.unit.episode_tripwires`` arms the signed-episode entry points
only (``EPISODE_ARMED`` and ``EPISODE_CONDITIONAL``), for the single-turn v2 suites that legitimately run the phase 1
stack (``test_episode_inertness``).
"""
from __future__ import annotations

import contextlib
import functools
import importlib
import inspect

import pytest

# label -> (owner, attr, original attribute) of the armed entry points, filled by the fixture.
_ORIGINALS: dict[str, tuple] = {}


@contextlib.contextmanager
def real_entry(module_name: str, dotted: str):
    """Run a block with ONE armed entry point restored (a test that legitimately declares a service task needs
    its contract parsed). Everything else stays armed."""
    owner, attr, original = _ORIGINALS[f"{module_name}.{dotted}"]
    armed = owner.__dict__[attr]
    setattr(owner, attr, original)
    try:
        yield
    finally:
        setattr(owner, attr, armed)

# (module, qualified attribute). ``validate_submission_policy`` is deliberately NOT here: the legacy admission
# path calls it with no policy and it must return None (checked by ``_policy_must_stay_none`` below).
ARMED = (
    ("reliquary.services.runtime", "ServiceRuntime.__init__"),
    ("reliquary.services.publication", "ObservationPublisher.__init__"),
    ("reliquary.services.schedule", "ScheduleRequestStore.__init__"),
    ("reliquary.services.cooldown_advisor", "recommend_cooldown"),
    ("reliquary.services.exploration", "ExplorationLedger.__init__"),
    ("reliquary.services.run_log", "RunObservationLog.__init__"),
    ("reliquary.services.settlement", "service_archive_rewards"),
    ("reliquary.services.settlement", "settle_window"),
    ("reliquary.services.scoring", "classify_signal"),
    ("reliquary.services.admission_policy", "service_lane"),
    ("reliquary.services.admission_policy", "parse_service_announcement"),
    ("reliquary.services.admission_policy", "missing_box_is_uncertain"),
    ("reliquary.services.admission_policy", "missing_box_problems"),
    ("reliquary.services.admission_policy", "exploration_pay_entitlement"),
    ("reliquary.miner.observation_client", "ObservationClient.__init__"),
    ("reliquary.protocol.service_submission", "ServiceBinding.from_dict"),
    ("reliquary.protocol.seed_pool", "PoolSelection.from_dict"),
    ("reliquary.protocol.seed_pool", "pool_from_service_policy"),
    # Archive validation, contract / schedule parsing, admission length rule, boundary work, lineage, guard.
    # Both the defining module and the name a caller imported into its own namespace are armed.
    ("reliquary.services.settlement", "validate_service_archive_v2"),
    ("reliquary.services.runtime", "validate_service_archive_v2"),
    ("reliquary.protocol.service_contract", "ServiceContract.from_dict"),
    ("reliquary.protocol.service_schedule", "ServiceSchedule.from_dict"),
    ("reliquary.validator.admission", "service_length_valid"),
    ("reliquary.validator.service", "ValidationService._service_window_plan"),
    ("reliquary.validator.service", "ValidationService._enqueue_aborted_service_window"),
    ("reliquary.services.runtime", "ServiceRuntime.ensure_checkpoint"),
    ("reliquary.services.runtime", "ServiceRuntime.record_pending_install"),
    ("reliquary.services.exploration", "audit_failure_class"),
    ("reliquary.services.runtime", "ServiceRuntime.apply_pending_schedule_request"),
    ("reliquary.validator.batcher", "GrpoWindowBatcher._service_pre_forward_guard"),
    ("reliquary.validator.batcher", "GrpoWindowBatcher._classify_audit_failure"),
)

# Signed episodes in the v2 path: never reached by a legacy task, a single-turn order or a corpus job.
EPISODE_ARMED = (
    ("reliquary.services.runtime", "ServiceRuntime.record_episode_precommit"),
    ("reliquary.services.runtime", "ServiceRuntime.episode_precommit"),
    ("reliquary.services.runtime", "ServiceRuntime.episode_precommit_sha"),
    ("reliquary.services.runtime", "ServiceRuntime.episode_precommit_recorded_at"),
    ("reliquary.services.runtime", "ServiceRuntime.prune_episode_precommits"),
    ("reliquary.protocol.service_episode", "EpisodePrecommit.from_dict"),
    ("reliquary.protocol.service_episode", "episode_commit_material"),
    ("reliquary.protocol.signatures", "build_service_episode_commit_binding"),
    ("reliquary.protocol.signatures", "build_episode_precommit_binding"),
    ("reliquary.protocol.toploc_proof", "span_proofs_b64"),
    ("reliquary.sandbox.rl_engagements", "RlEpisodeEngagements.terms"),
    ("reliquary.sandbox.sessions", "SessionIssuer.claim_all"),
    ("reliquary.sandbox.sessions", "SessionIssuer.persist_submitted"),
    ("reliquary.sandbox.sessions", "SessionIssuer.submitted_all"),
    ("reliquary.sandbox.sessions", "SessionIssuer.hand_back"),
    ("reliquary.sandbox.sessions", "SessionBook.engagement_held"),
    ("reliquary.sandbox.sessions", "SessionBook.of_precommit"),
    ("reliquary.sandbox.sessions", "SessionBook.precommits_held"),
    ("reliquary.validator.episode_admission", "EpisodeGroupChecker.check"),
    ("reliquary.validator.episode_admission", "finish_prepared"),
    ("reliquary.validator.episode_intake", "EpisodeGroupIntake.admit"),
    ("reliquary.validator.rl_sandbox_wiring", "build_rl_episode_services"),
    ("reliquary.validator.toploc_check", "toploc_span_verdict"),
    ("reliquary.validator.verifier", "_episode_stop_picks"),
    ("reliquary.validator.batcher", "signed_episode_proof_refusal"),
    ("reliquary.validator.batcher", "signed_episode_unchecked_turn"),
    ("reliquary.validator.remote_proof", "_config_namespace"),
    ("reliquary.validator.server", "ValidatorServer._admit_episode_group"),
    ("reliquary.validator.server", "ValidatorServer._persist_episode_claim"),
    ("reliquary.validator.server", "ValidatorServer._start_episode_settlement"),
    ("reliquary.validator.service", "ValidationService._episode_task_in_cooldown"),
    ("reliquary.validator.service", "ValidationService._note_episode_window"),
    ("reliquary.validator.service", "ValidationService._current_service_window"),
    ("reliquary.validator.service", "ValidationService._episode_stop_sets"),
    ("reliquary.validator.service", "ValidationService._episode_precommit_retention_s"),
    ("reliquary.validator.service", "ValidationService._default_episode_renderer"),
)
ARMED = ARMED + EPISODE_ARMED


def _kw(args, kwargs, name, position):
    return kwargs[name] if name in kwargs else (args[position] if len(args) > position else None)


def _batcher_hands_back(args, kwargs, result):
    self, pending = args[0], _kw(args, kwargs, "pending", 1)
    return self.episode_proof_inconclusive is not None or (
        pending is not None and self._signed_episode_pending(pending))


# Signed-episode entry points a legacy task, a single-turn order or a corpus job DOES call, where the new code returns
# at once: wrapped, run for real, and recorded only when ``predicate(args, kwargs, result)`` says the episode
# branch was taken. Both the defining module and the name a caller imported are wrapped.
EPISODE_CONDITIONAL = (
    ("reliquary.validator.service", "ValidationService._start_episode_services",
     lambda a, k, r: getattr(getattr(getattr(a[0], "_service_runtime", None), "contract", None),
                             "episode_environments", ())),
    ("reliquary.validator.service", "ValidationService._stop_episode_services",
     lambda a, k, r: getattr(a[0], "_episode_services", None) is not None),
    ("reliquary.validator.batcher", "GrpoWindowBatcher._episode_proof_inconclusive", _batcher_hands_back),
    ("reliquary.validator.verifier", "signed_episode_spans", lambda a, k, r: r is not None),
    ("reliquary.validator.server", "_episode_admission_fields", lambda a, k, r: bool(r)),
    ("reliquary.validator.admission", "service_length_valid",
     lambda a, k, r: k.get("episode_max_tokens") is not None),
    ("reliquary.protocol.service_contract", "ServiceContract.episode_policy", lambda a, k, r: r is not None),
    ("reliquary.protocol.service_contract", "supported_v2_capabilities",
     lambda a, k, r: "signed-sandbox-episode/v1" in r),
    ("reliquary.services.admission_policy", "supported_v2_capabilities",
     lambda a, k, r: "signed-sandbox-episode/v1" in r),
    ("reliquary.services.runtime", "supported_v2_capabilities",
     lambda a, k, r: "signed-sandbox-episode/v1" in r),
    ("reliquary.protocol.seed_pool", "validate_rollout_selection", lambda a, k, r: bool(k.get("signed_episodes"))),
    ("reliquary.protocol.service_submission", "validate_service_rollout_bindings",
     lambda a, k, r: bool(k.get("signed_episodes"))),
    ("reliquary.shared.training_payload", "_is_signed_episode", lambda a, k, r: bool(r)),
    ("reliquary.miner.corpus_generate_server", "GenerateEngine.__init__", lambda a, k, r: k.get("draws") is not None),
    ("reliquary.miner.engine", "_single_turn_envs", lambda a, k, r: r[0] is not a[0]),
    ("reliquary.infrastructure.sandbox_store", "R2SessionStore.__init__",
     lambda a, k, r: k.get("prefix", "reliquary/sandbox/sessions/") != "reliquary/sandbox/sessions/"),
)

# Entry points a legacy boot / window DOES call, returning at once when there is no runtime: they are wrapped,
# not blocked, and recorded only when called on an instance that has a service runtime.
RUNTIME_ONLY = (
    ("reliquary.validator.service", "ValidationService._start_observation_publication"),
    ("reliquary.validator.service", "ValidationService._start_episode_services"),
    ("reliquary.validator.service", "ValidationService._refresh_service_active"),
)


def _resolve(module_name: str, dotted: str):
    owner = importlib.import_module(module_name)
    *path, attr = dotted.split(".")
    for name in path:
        owner = getattr(owner, name)
    return owner, attr


def arm_conditional(monkeypatch, calls: list[str], conditional) -> None:
    """Wrap each ``(module, dotted, predicate)``: the real code runs, and the call is recorded (and refused)
    when the predicate says it took the episode branch. Coroutine functions stay coroutine functions."""
    for module_name, dotted, predicate in conditional:
        owner, attr = _resolve(module_name, dotted)
        real = owner.__dict__.get(attr, getattr(owner, attr))
        is_static = isinstance(real, staticmethod)
        function = real.__func__ if is_static else real
        label = f"{module_name}.{dotted}"

        def make(label, function, predicate):
            def check(args, kwargs, result):
                if predicate(args, kwargs, result):
                    calls.append(label)
                    raise AssertionError(f"{label} took its episode branch for a legacy RL task / single-turn "
                                         f"order / corpus job")
                return result

            if inspect.iscoroutinefunction(function):
                async def conditional_async(*args, **kwargs):
                    return check(args, kwargs, await function(*args, **kwargs))
                return functools.wraps(function)(conditional_async)

            def conditional(*args, **kwargs):
                return check(args, kwargs, function(*args, **kwargs))
            return functools.wraps(function)(conditional)
        wrapped = make(label, function, predicate)
        monkeypatch.setattr(owner, attr, staticmethod(wrapped) if is_static else wrapped)


def arm_entry_points(monkeypatch, calls: list[str], armed) -> None:
    """Replace each ``(module, dotted)`` by a wrapper that RAISES and RECORDS the call."""
    for module_name, dotted in armed:
        owner, attr = _resolve(module_name, dotted)
        label = f"{module_name}.{dotted}"
        original = owner.__dict__.get(attr, getattr(owner, attr))
        is_static = isinstance(original, (staticmethod, classmethod))
        _ORIGINALS[label] = (owner, attr, original)

        def make(label):
            def tripwire(*args, **kwargs):
                calls.append(label)
                raise AssertionError(f"{label} ran for a legacy RL task / corpus job")
            return tripwire
        monkeypatch.setattr(owner, attr, staticmethod(make(label)) if is_static else make(label))


def arm_episode(monkeypatch, calls: list[str]) -> None:
    """The signed-episode entry points only (a single-turn v2 order legitimately runs the phase 1 stack)."""
    arm_entry_points(monkeypatch, calls, EPISODE_ARMED)
    arm_conditional(monkeypatch, calls, EPISODE_CONDITIONAL)


def arm_legacy(monkeypatch, calls: list[str]) -> None:
    """Every entry point, phase 1 and signed-episode: what a legacy RL task or a corpus job runs under."""
    arm_entry_points(monkeypatch, calls, ARMED)

    for module_name, dotted in RUNTIME_ONLY:
        owner, attr = _resolve(module_name, dotted)
        real_method = owner.__dict__[attr]

        def make(label, real_method):
            def runtime_only(self, *args, **kwargs):
                if getattr(self, "_service_runtime", None) is not None:
                    calls.append(label)
                    raise AssertionError(f"{label} ran with a service runtime for a legacy RL task / corpus job")
                return real_method(self, *args, **kwargs)
            return runtime_only
        monkeypatch.setattr(owner, attr, make(f"{module_name}.{dotted}", real_method))

    # Wrapped once: an entry point already armed or runtime-only keeps that (stricter) wrapper here.
    stricter = set(ARMED) | set(RUNTIME_ONLY)
    arm_conditional(monkeypatch, calls, [c for c in EPISODE_CONDITIONAL if (c[0], c[1]) not in stricter])

    # The legacy admission path does call this one, with no policy: it must answer "no policy".
    policy = importlib.import_module("reliquary.services.admission_policy")
    real_policy = policy.validate_submission_policy

    def policy_must_stay_none(request, announcement):
        result = real_policy(request, announcement)
        if announcement is not None or result is not None:
            calls.append("reliquary.services.admission_policy.validate_submission_policy (a policy was in force)")
        return result
    monkeypatch.setattr(policy, "validate_submission_policy", policy_must_stay_none)


@pytest.fixture(autouse=True)
def rl_service_tripwires(monkeypatch):
    calls: list[str] = []
    arm_legacy(monkeypatch, calls)
    yield calls
    assert not calls, f"RL service entry points were reached: {sorted(set(calls))}"
