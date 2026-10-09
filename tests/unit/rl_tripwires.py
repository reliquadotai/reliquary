"""Tripwires on every entry point of the next-RL-run service stack (phase 1).

A legacy RL task and a corpus job must never reach them. ``rl_service_tripwires`` is autouse where it is
imported: it replaces each entry point by a wrapper that RAISES and RECORDS the call, and fails the test
afterwards if anything was recorded (a caller that swallows the exception does not hide it).

Used two ways: ``from tests.unit.rl_tripwires import rl_service_tripwires`` in a test module (armed for that
module only), and ``pytest -p tests.unit.rl_tripwires <legacy test file>`` (``test_next_rl_run_inertness``
reruns the legacy and corpus suites that way).
"""
from __future__ import annotations

import contextlib
import importlib

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

# Entry points a legacy boot / window DOES call, returning at once when there is no runtime: they are wrapped,
# not blocked, and recorded only when called on an instance that has a service runtime.
RUNTIME_ONLY = (
    ("reliquary.validator.service", "ValidationService._start_observation_publication"),
    ("reliquary.validator.service", "ValidationService._refresh_service_active"),
)


def _resolve(module_name: str, dotted: str):
    owner = importlib.import_module(module_name)
    *path, attr = dotted.split(".")
    for name in path:
        owner = getattr(owner, name)
    return owner, attr


@pytest.fixture(autouse=True)
def rl_service_tripwires(monkeypatch):
    calls: list[str] = []

    def arm(label: str, owner, attr: str) -> None:
        original = owner.__dict__.get(attr, getattr(owner, attr))
        is_static = isinstance(original, (staticmethod, classmethod))
        _ORIGINALS[label] = (owner, attr, original)

        def tripwire(*args, **kwargs):
            calls.append(label)
            raise AssertionError(f"{label} ran for a legacy RL task / corpus job")
        monkeypatch.setattr(owner, attr, staticmethod(tripwire) if is_static else tripwire)

    for module_name, dotted in ARMED:
        owner, attr = _resolve(module_name, dotted)
        arm(f"{module_name}.{dotted}", owner, attr)

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

    # The legacy admission path does call this one, with no policy: it must answer "no policy".
    policy = importlib.import_module("reliquary.services.admission_policy")
    real_policy = policy.validate_submission_policy

    def policy_must_stay_none(request, announcement):
        result = real_policy(request, announcement)
        if announcement is not None or result is not None:
            calls.append("reliquary.services.admission_policy.validate_submission_policy (a policy was in force)")
        return result
    monkeypatch.setattr(policy, "validate_submission_policy", policy_must_stay_none)

    yield calls
    assert not calls, f"RL service entry points were reached: {sorted(set(calls))}"
