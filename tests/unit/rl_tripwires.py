"""Tripwires on every entry point of the next-RL-run service stack (phase 1).

A legacy RL task and a corpus job must never reach them. ``rl_service_tripwires`` is autouse where it is
imported: it replaces each entry point by a wrapper that RAISES and RECORDS the call, and fails the test
afterwards if anything was recorded (a caller that swallows the exception does not hide it).

Used two ways: ``from tests.unit.rl_tripwires import rl_service_tripwires`` in a test module (armed for that
module only), and ``pytest -p tests.unit.rl_tripwires <legacy test file>`` (``test_next_rl_run_inertness``
reruns the legacy and corpus suites that way).
"""
from __future__ import annotations

import importlib

import pytest

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
    ("reliquary.services.admission_policy", "service_signal_admits"),
    ("reliquary.services.admission_policy", "missing_box_is_uncertain"),
    ("reliquary.services.admission_policy", "missing_box_problems"),
    ("reliquary.services.admission_policy", "exploration_pay_entitlement"),
    ("reliquary.miner.observation_client", "ObservationClient.__init__"),
    ("reliquary.protocol.service_submission", "ServiceBinding.from_dict"),
    ("reliquary.protocol.seed_pool", "PoolSelection.from_dict"),
    ("reliquary.protocol.seed_pool", "pool_from_service_policy"),
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

        def tripwire(*args, **kwargs):
            calls.append(label)
            raise AssertionError(f"{label} ran for a legacy RL task / corpus job")
        monkeypatch.setattr(owner, attr, staticmethod(tripwire) if is_static else tripwire)

    for module_name, dotted in ARMED:
        owner, attr = _resolve(module_name, dotted)
        arm(f"{module_name}.{dotted}", owner, attr)

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
