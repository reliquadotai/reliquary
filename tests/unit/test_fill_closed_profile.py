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
