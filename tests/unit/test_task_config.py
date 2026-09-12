"""A validator that cannot find itself in the registry does not run."""

from __future__ import annotations

import pytest

from reliquary.environment.abi import canonical_sha256
from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry
from reliquary.validator.task_config import TaskConfigError, resolve_task_config

CONTRACT = {"model_id": "demo", "environments": {}}
DIGEST = canonical_sha256(CONTRACT)
PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.6,
    "median_rounds": 4800,
}


def _entry(**overrides) -> TaskEntry:
    base = dict(
        task_id="default",
        profile_id="demo-profile",
        profile_sha256=DIGEST,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=dict(PARAMS),
        status="active",
        retired_at=None,
    )
    return TaskEntry(**{**base, **overrides})


def _resolve(entries, task_id="default"):
    return resolve_task_config(
        entries, task_id, profile_id="demo-profile", generation_contract=CONTRACT
    )


def test_a_declared_task_yields_its_price_parameters():
    config = _resolve({"default": _entry()})

    assert config.emission_cap == 0.6
    assert config.price_params.cap == 0.6
    assert config.price_params.median_rounds == 4800


def test_an_undeclared_task_refuses():
    with pytest.raises(TaskConfigError, match="not declared"):
        _resolve({"other": _entry(task_id="other")}, task_id="default")


def test_an_oversubscribed_registry_refuses():
    entries = {
        "default": _entry(params={**PARAMS, "cap": 0.8}),
        "other": _entry(task_id="other", params={**PARAMS, "cap": 0.5}),
    }

    with pytest.raises(TaskConfigError, match="1.3"):
        _resolve(entries)


def test_a_profile_the_binary_does_not_match_refuses():
    with pytest.raises(TaskConfigError, match="profile"):
        _resolve({"default": _entry(profile_sha256="b" * 64)})


def test_a_profile_id_mismatch_refuses():
    with pytest.raises(TaskConfigError, match="demo-profile"):
        resolve_task_config(
            {"default": _entry()},
            "default",
            profile_id="another-profile",
            generation_contract=CONTRACT,
        )


def test_an_unknown_mechanism_refuses():
    # Refused by delegation: validate_registry rejects the entry before we
    # ever look it up, and the wrapper names the mechanism.
    with pytest.raises(TaskConfigError, match="vibes"):
        _resolve({"default": _entry(mechanism="vibes")})


def test_a_retired_task_refuses_to_start():
    with pytest.raises(TaskConfigError, match="retired"):
        _resolve({"default": _entry(status="retired")})


# --- One legacy-fallback predicate, called by both the startup path and the
# weight submitter. Written twice they drifted; the drift is how a retired
# task kept being paid. ---

def test_the_legacy_fallback_is_armed_only_by_a_wholly_absent_registry():
    from reliquary.validator.task_config import legacy_registry_fallback

    assert legacy_registry_fallback({}, ["default"]) is True


def test_a_registry_that_exists_never_arms_the_legacy_fallback():
    from reliquary.validator.task_config import legacy_registry_fallback

    assert legacy_registry_fallback({"other": _entry(task_id="other")}, ["default"]) is False


def test_anything_but_the_legacy_task_alone_does_not_arm_the_fallback():
    from reliquary.validator.task_config import legacy_registry_fallback

    assert legacy_registry_fallback({}, ["logic-probe"]) is False
    assert legacy_registry_fallback({}, ["default", "logic-probe"]) is False
    assert legacy_registry_fallback({}, []) is False


def test_both_call_sites_use_the_shared_predicate():
    """The two fallbacks were written independently, already differed in
    subject, and nothing linked them."""
    import pathlib
    import subprocess

    root = pathlib.Path(__file__).resolve().parents[2] / "reliquary"
    for module in ("cli/main.py", "validator/weight_only.py"):
        text = (root / module).read_text()
        assert "legacy_registry_fallback" in text, module

    # And neither re-implements it: no call site still spells the condition
    # out against DEFAULT_TASK_ID by hand.
    hits = subprocess.run(
        ["grep", "-n", "DEFAULT_TASK_ID",
         str(root / "cli" / "main.py"), str(root / "validator" / "weight_only.py")],
        capture_output=True, text=True,
    ).stdout.strip()
    assert hits == "", f"a fallback predicate is still written by hand:\n{hits}"
