"""v6 coexists: it is selectable, and it leaves v4/v5 untouched."""
import os
import subprocess
import sys

from reliquary.protocol.profiles import resolve_protocol_profile


def test_v6_is_selectable_and_carries_the_v5_generation_contract():
    v5 = resolve_protocol_profile("qwen3-4b-base-dapo-reasoning-v5")
    v6 = resolve_protocol_profile("qwen3-4b-base-dapo-fill-closed-v6")

    assert v6.protocol_version == 6
    # v6 changes the WINDOW, not what a miner generates. Everything a miner
    # samples from must be byte-identical or the change is not what it claims.
    assert v6.sampling == v5.sampling
    assert v6.model_id == v5.model_id
    assert v6.model_revision == v5.model_revision
    assert {name: env.max_new_tokens for name, env in v6.environments.items()} == \
           {name: env.max_new_tokens for name, env in v5.environments.items()}


def test_v6_has_no_throughput_tiebreak():
    """There is no ranking in v6, so there is nothing to break ties in."""
    v6 = resolve_protocol_profile("qwen3-4b-base-dapo-fill-closed-v6")
    assert v6.throughput_tiebreak is None


def test_the_capability_cannot_be_armed_under_a_non_v6_profile():
    """The target derives from B_BATCH and the publish interval, both of which
    are profile-dependent. Arming this under v4 would size a v6 window from
    another protocol's batch shape."""
    # A fresh interpreter per case, not importlib.reload: ACTIVE_PROTOCOL_PROFILE
    # is resolved once at import of reliquary.protocol.profiles, and
    # reliquary.constants binds PROTOCOL_VERSION from it at its own import —
    # both need to observe the env var from process start, which is exactly
    # what test_profile_atomically_drives_runtime_constants (above, in
    # test_protocol_profiles.py) already does for this reason.
    script = (
        "from reliquary import constants as c; "
        "print(c.FILL_CLOSED_ENABLED)"
    )
    base_env = {
        k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")
    }
    base_env["RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED"] = "1"

    # Armed under v4: must fail closed despite the env var asking for it.
    v4_env = dict(base_env, RELIQUARY_PROTOCOL_PROFILE="qwen3-4b-base-dapo-v4")
    v4_result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=v4_env,
    )
    assert v4_result.stdout.strip() == "False"

    # Armed under v6: the same env var now takes effect.
    v6_env = dict(
        base_env,
        RELIQUARY_PROTOCOL_PROFILE="qwen3-4b-base-dapo-fill-closed-v6",
    )
    v6_result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
        env=v6_env,
    )
    assert v6_result.stdout.strip() == "True"


def _constants_under(**env_overrides):
    """Import ``reliquary.constants`` in a fresh interpreter under a v6
    profile plus ``env_overrides``. Returns the CompletedProcess."""
    base_env = {
        k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")
    }
    base_env["RELIQUARY_PROTOCOL_PROFILE"] = "qwen3-4b-base-dapo-fill-closed-v6"
    base_env["RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED"] = "1"
    base_env.update(env_overrides)
    return subprocess.run(
        [
            sys.executable, "-c",
            "from reliquary import constants as c; "
            "print(c.FILL_CLOSED_TARGET_GROUPS_PER_ENV, "
            "c.FILL_CLOSED_EMISSIONS_PER_WINDOW, c.B_BATCH)",
        ],
        capture_output=True, text=True, env=base_env,
    )


def _v6_script(source: str, **env_overrides):
    """Run a service-level assertion with the real v6 constants loaded."""
    base_env = {
        k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")
    }
    base_env.update({
        "RELIQUARY_PROTOCOL_PROFILE": "qwen3-4b-base-dapo-fill-closed-v6",
        "RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED": "1",
    })
    base_env.update(env_overrides)
    return subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True,
        text=True,
        env=base_env,
    )


def test_the_default_target_sits_exactly_on_the_emission_ceiling():
    """The target is what one window's key range can actually hold: every
    emission is one B_BATCH-per-environment batch, and a window owns
    FILL_CLOSED_EMISSIONS_PER_WINDOW keys."""
    result = _constants_under()

    assert result.returncode == 0, result.stderr
    target, emissions, b_batch = (int(x) for x in result.stdout.split())
    assert target == emissions * b_batch


def test_a_target_past_the_emission_ceiling_refuses_to_import():
    """I2: the target is env-overridable, the emission count is a literal.
    Raising the target alone lets a window fill past its own key range --
    ``encoded_window_journal_key`` then raises ValueError from
    ``FillClosedBatchAssembler`` on a PROOF thread, which faults the whole
    proof plane. Fail at import, where an operator can read the message."""
    result = _constants_under(
        RELIQUARY_FILL_CLOSED_TARGET_GROUPS_PER_ENV="100000",
    )

    assert result.returncode != 0
    assert "FILL_CLOSED_EMISSIONS_PER_WINDOW" in result.stderr
    assert "journal" in result.stderr


def test_fill_closed_runtime_bounds_are_coherent_and_finite():
    result = _v6_script(
        "from reliquary import constants as c; "
        "assert c.FILL_CLOSED_EMISSIONS_PER_WINDOW == 16; "
        "assert c.FILL_CLOSED_TARGET_GROUPS_PER_ENV == 16 * c.B_BATCH; "
        "assert c.FILL_CLOSED_ADMISSION_BUDGET_PER_ENV >= "
        "c.FILL_CLOSED_TARGET_GROUPS_PER_ENV; "
        "assert c.FILL_CLOSED_GRADING_START_BUDGET_PER_ENV >= "
        "c.FILL_CLOSED_ADMISSION_BUDGET_PER_ENV; "
        "assert c.FILL_CLOSED_PRECOMMIT_SECONDS > 100; "
        "assert c.FILL_CLOSED_PRECOMMIT_SECONDS + "
        "c.SUBMISSION_UPLOAD_GRACE_SECONDS <= c.FILL_CLOSED_MAX_SECONDS"
    )

    assert result.returncode == 0, result.stderr


def test_fill_closed_refuses_incoherent_operator_bounds():
    too_small = _v6_script(
        "from reliquary import constants",
        RELIQUARY_FILL_CLOSED_ADMISSION_BUDGET_PER_ENV="255",
    )
    too_late = _v6_script(
        "from reliquary import constants",
        RELIQUARY_FILL_CLOSED_PRECOMMIT_SECONDS="1800",
    )

    assert too_small.returncode != 0
    assert "ADMISSION_BUDGET_PER_ENV" in too_small.stderr
    assert too_late.returncode != 0
    assert "leave SUBMISSION_UPLOAD_GRACE_SECONDS" in too_late.stderr


def test_real_v6_service_accepts_after_100_seconds_and_can_reach_16_picks():
    """Exercise the configured v6 profile through service construction.

    This catches both historical seams: the constructed batcher inherited the
    profile's 100-second precommit cutoff and the legacy 96-candidate ceiling,
    even though its shared fill state asked for sixteen B_BATCH picks.
    """
    result = _v6_script(
        r'''
from reliquary import constants as c
from reliquary.validator import batcher as batcher_module
from reliquary.validator.cooldown import ContentCooldownMap, CooldownMap
from tests.unit.test_grpo_window_batcher import FakeEnv, PrivateRewardFakeEnv
from tests.unit.test_service_v2 import _build_late_drop_service

svc = _build_late_drop_service()
svc.server.set_registered_hotkeys(
    {"miner"}, operator_by_hotkey={"miner": "operator"}
)
svc.envs = {
    "openmathinstruct": FakeEnv(),
    "opencodeinstruct": PrivateRewardFakeEnv(),
}
svc.env_mix = [(name, c.B_BATCH) for name in svc.envs]
svc.env = svc.envs["openmathinstruct"]
svc._cooldown_per_env = {
    name: CooldownMap(cooldown_windows=1_000_000) for name in svc.envs
}
svc._content_cooldown_per_env = {
    name: ContentCooldownMap(cooldown_windows=1_000_000)
    for name in svc.envs
}
batchers = svc._build_window_batchers(999)

for batcher in batchers.values():
    assert batcher.collection_seconds == c.FILL_CLOSED_PRECOMMIT_SECONDS
    assert batcher.max_productive_candidates == (
        c.FILL_CLOSED_ADMISSION_BUDGET_PER_ENV
    )
    assert batcher.max_grading_starts == (
        c.FILL_CLOSED_GRADING_START_BUDGET_PER_ENV
    )
    batcher._emit_training_batch_fn = lambda *_args: None

math = batchers["openmathinstruct"]
math.mark_window_opened(monotonic_time=1_000.0, wall_time=10_000.0)
math._time_fn = lambda: 1_101.0
accepted, reason, _ = math.try_register_upload_precommit(
    "after-100-seconds",
    "miner",
    t_arrival_wall=10_101.0,
    payload_bytes=1,
    payload_sha256="ab" * 32,
)
assert accepted is True, reason

accepted, reason, _ = math.try_register_upload_precommit(
    "past-cutoff",
    "miner",
    t_arrival_wall=(
        math.window_opened_wall_ts + math.collection_seconds + 0.1
    ),
    payload_bytes=1,
    payload_sha256="cd" * 32,
)
assert accepted is False
assert reason == "collection_closed"

shared = math.fill_state
assert shared is not None
target = c.FILL_CLOSED_EMISSIONS_PER_WINDOW * c.B_BATCH
assert target == 256
assert c.FILL_CLOSED_ADMISSION_BUDGET_PER_ENV >= target
for environment, batcher in batchers.items():
    with shared.lock:
        for index in range(target):
            assert shared.may_admit(environment)
            shared.reserve(environment)
            shared.record_proven(environment)
            batcher._proven_groups.setdefault(environment, []).append(
                batcher_module._ProvenGroup(
                    value=f"{environment}-{index}",
                    rate=1.0,
                    payload_bytes=1,
                    receipt_id=f"{environment}-{index}",
                )
            )

svc._fill_closed_pick_gate_open = lambda *_args, **_kwargs: True
events = 0
while svc._drive_fill_closed_picks(list(batchers.values())):
    events += 1
assert events == 16
assert shared.snapshot()["picks_emitted"] == 16
assert shared.is_closed() is True
'''
    )

    assert result.returncode == 0, result.stderr


def test_the_two_experimental_capabilities_refuse_to_run_together():
    """R30: fill-closed and checkpoint-epoch are mutually exclusive.

    With both armed, C2's tombstone gate (``and checkpoint_epoch is None``)
    falls back to the RAW journal key, colliding with v6's encoded key
    space -- the original bug -- and ``_open_checkpoint_epoch`` would
    ``close()`` a lane's assembler before its window is done. Import is the
    only place an operator reads the refusal.
    """
    result = _constants_under(
        RELIQUARY_EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED="1",
    )

    assert result.returncode != 0
    assert "RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED" in result.stderr
    assert "RELIQUARY_EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED" in result.stderr


def test_checkpoint_epoch_alone_still_imports():
    """The refusal is about the PAIR: the epoch capability on its own (v6
    profile, fill-closed disarmed) is untouched by this branch."""
    result = _constants_under(
        RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED="0",
        RELIQUARY_EXPERIMENTAL_CHECKPOINT_EPOCH_ENABLED="1",
    )

    assert result.returncode == 0, result.stderr
