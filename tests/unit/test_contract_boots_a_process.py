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
print(json.dumps({
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
