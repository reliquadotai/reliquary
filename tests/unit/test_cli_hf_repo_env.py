"""``reliquary validate --hf-repo-id`` defaults to RELIQUARY_HF_REPO_ID, so an ``rl supervise`` controller (which
inherits the supervisor's environment and cannot be given options) names the repo the remote signer expects."""
import os
import subprocess
import sys

PROBE = (
    "import inspect; from reliquary.cli import main; "
    "print(inspect.signature(main.validate).parameters['hf_repo_id'].default.default)"
)


def _default(env_value):
    env = {k: v for k, v in os.environ.items() if k != "RELIQUARY_HF_REPO_ID"}
    if env_value is not None:
        env["RELIQUARY_HF_REPO_ID"] = env_value
    out = subprocess.run([sys.executable, "-c", PROBE], env=env, capture_output=True, text=True, check=True)
    return out.stdout.strip().splitlines()[-1]


def test_hf_repo_id_defaults_to_the_environment():
    assert _default("ReliquaryForge/example-repo") == "ReliquaryForge/example-repo"


def test_hf_repo_id_keeps_the_constant_without_the_environment():
    from reliquary.constants import DEFAULT_HF_REPO_ID

    assert _default(None) == DEFAULT_HF_REPO_ID
