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
