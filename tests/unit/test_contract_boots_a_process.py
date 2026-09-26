"""A process booting under a contract file must derive exactly the constants it
would have derived from the equivalent compiled profile. Nothing short of a
real import in a real subprocess proves it: `constants` derives 21 values at
import time, and 46 modules import it."""

import json
import subprocess
import sys

from reliquary.protocol.profiles import (
    DEFAULT_PROFILE_ID,
    PROFILES,
    TASK_CONTRACT_ENV_VAR,
)

PROBE = """
import json, sys
from reliquary import constants
from reliquary.protocol.release_contract import canonical_sha256
print(json.dumps({
    "contract_sha256": canonical_sha256(constants.PROTOCOL_GENERATION_CONTRACT),
    "profile_id": constants.PROTOCOL_PROFILE_ID,
    "model_id": constants.PROTOCOL_MODEL_ID,
    "model_revision": constants.PROTOCOL_MODEL_REVISION,
    "protocol_version": constants.PROTOCOL_VERSION,
    "rollouts": constants.M_ROLLOUTS,
    "temperature": constants.T_PROTO,
    "top_p": constants.TOP_P_PROTO,
    "top_k": constants.TOP_K_PROTO,
}))
"""


def _probe(env):
    """Run the probe with a clean slate for the contract variable.

    It is REMOVED rather than set to "", because an empty value is fatal by
    design — a variable emptied by broken templating must not silently fall
    back to the compiled catalogue.
    """
    import os

    merged = {k: v for k, v in os.environ.items() if k != TASK_CONTRACT_ENV_VAR}
    merged.update(env)
    result = subprocess.run(
        [sys.executable, "-c", PROBE],
        capture_output=True, text=True, env=merged, timeout=300,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def test_a_contract_file_derives_the_same_constants(tmp_path):
    # The DEFAULT profile, not sorted(PROFILES)[0]: some profiles are gated
    # behind an experimental capability flag and cannot import `constants` at
    # all without it. The default is by definition the one every process boots
    # under today, so it is both bootable and the right thing to compare.
    profile_id = DEFAULT_PROFILE_ID
    source = PROFILES[profile_id]

    baseline = _probe({"RELIQUARY_PROTOCOL_PROFILE": profile_id})

    path = tmp_path / "contract.json"
    path.write_text(json.dumps(source.to_generation_contract()))
    under_contract = _probe({TASK_CONTRACT_ENV_VAR: str(path)})

    assert under_contract == baseline


def test_an_unusable_contract_stops_the_process(tmp_path):
    import os

    path = tmp_path / "contract.json"
    path.write_text("{not json")
    result = subprocess.run(
        [sys.executable, "-c", "from reliquary import constants"],
        capture_output=True, text=True, timeout=300,
        env={**os.environ, TASK_CONTRACT_ENV_VAR: str(path)},
    )
    assert result.returncode != 0
    assert "contract" in result.stderr.lower()


def test_a_task_the_cli_created_boots_and_the_validator_accepts_it(tmp_path):
    """The whole point of the feature, end to end, across the one boundary no
    other test crosses: the CLI seals an entry, a deployment mounts its
    contract, the process boots under it, and the validator accepts the pair.

    Every test on either side supplies its own fixture, so a field the CLI
    writes but the round trip drops passes both sides and fails only here.
    """
    from reliquary.cli.main import build_contract_task_entry
    from reliquary.protocol.profiles import profile_from_contract
    from reliquary.validator.task_config import resolve_task_config

    entry = build_contract_task_entry(
        task_id="glm-run",
        from_profile=DEFAULT_PROFILE_ID,
        model_id="org/GLM",
        model_revision="abc123",
        model_architecture="Qwen3ForCausalLM",
        environments=None,
        cap=0.30,
        overrides={},
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(entry.contract))

    booted = _probe({TASK_CONTRACT_ENV_VAR: str(path)})
    assert booted["profile_id"] == "glm-run"
    assert booted["model_id"] == "org/GLM"
    # The digest the booted process computes is the one the registry attests.
    assert booted["contract_sha256"] == entry.profile_sha256

    # And the startup refusals pass on exactly that round trip.
    config = resolve_task_config(
        {"glm-run": entry},
        "glm-run",
        profile_id=booted["profile_id"],
        generation_contract=profile_from_contract(
            entry.contract
        ).to_generation_contract(),
    )
    assert config.task_id == "glm-run"


def _template_carrying_the_newest_environment_fields():
    """The profile whose contract exercises the fields added most recently.

    The default profile predates them, so the test above cannot catch a
    reader that drops one: it is the newest fields that are unread, every
    time.
    """
    for profile_id in sorted(PROFILES):
        contract = PROFILES[profile_id].to_generation_contract()
        for body in contract["environments"].values():
            if body.get("prompt_cooldown_windows") is not None:
                return profile_id
    import pytest

    pytest.fail(
        "no compiled profile exercises 'prompt_cooldown_windows'; "
        "this test cannot run"
    )


def test_a_task_seeded_from_the_newest_profile_boots(tmp_path):
    """A generation task on a policy with per-environment cooldown and
    reasoning settings, created the way an operator creates one.

    This is the case the merge broke: the contract carried two fields the
    reader did not honour, so the entry was attested and could not start.
    """
    from reliquary.cli.main import build_contract_task_entry

    template = _template_carrying_the_newest_environment_fields()
    entry = build_contract_task_entry(
        task_id="newest-run",
        from_profile=template,
        model_id="org/Policy",
        model_revision="abc123",
        model_architecture="Qwen3_5ForConditionalGeneration",
        environments=None,
        cap=0.30,
        overrides={},
        verification="streamed",
    )
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(entry.contract))

    booted = _probe({TASK_CONTRACT_ENV_VAR: str(path)})
    assert booted["profile_id"] == "newest-run"
    assert booted["contract_sha256"] == entry.profile_sha256
    # Pinning the replica must not have moved the digest.
    assert entry.verification == "streamed"
    assert "verification" not in entry.contract
