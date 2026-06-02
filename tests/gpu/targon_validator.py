"""Deploy a Reliquary validator on a Targon H100.

Uses pytorch:devel (torch preinstalled, working apt/pip) and avoids any version
pin in pip_install (SDK doesn't shell-escape and '>=' becomes a redirect).

Usage:
    targon deploy tests/gpu/targon_validator.py --name reliquary-validator
"""

import os
import targon


CHECKPOINT = os.environ.get("VALIDATOR_CHECKPOINT", "gpt2")
NETWORK = os.environ.get("BT_NETWORK", "finney")
NETUID = int(os.environ.get("NETUID", "81"))
WALLET_NAME = os.environ.get("BT_WALLET_NAME", "subnet")
HOTKEY = os.environ.get("BT_HOTKEY", "hotkey1")
WALLET_PATH = os.environ.get("BT_WALLET_PATH", "/opt/reliquary/wallets")
HTTP_PORT = 8888

# Secrets are read from the shell environment at deploy-build time and
# baked into the image. Never hardcode them here — GitHub secret scanning
# will (correctly) block the push and they end up in git history forever.
def _require(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise SystemExit(
            f"{name} must be set in the shell environment before running "
            f"`targon deploy tests/gpu/targon_validator.py`."
        )
    return val


ENV_VARS = {
    "PYTHONUNBUFFERED": "1",
    "HF_HOME": "/tmp/hf",
    "BT_NETWORK": NETWORK,
    "NETUID": str(NETUID),
    "BT_WALLET_NAME": WALLET_NAME,
    "BT_HOTKEY": HOTKEY,
    "BT_WALLET_PATH": WALLET_PATH,
    "R2_ENDPOINT_URL": os.environ.get(
        "R2_ENDPOINT_URL", "https://s3.us-east-1.amazonaws.com"
    ),
    "R2_ACCOUNT_ID": os.environ.get("R2_ACCOUNT_ID", "dummy"),
    "R2_ACCESS_KEY_ID": _require("R2_ACCESS_KEY_ID"),
    "R2_SECRET_ACCESS_KEY": _require("R2_SECRET_ACCESS_KEY"),
    "R2_REGION": os.environ.get("R2_REGION", "us-east-1"),
    "R2_BUCKET_ID": os.environ.get("R2_BUCKET_ID", "reliquary-catalyst-test"),
    "RELIQUARY_STATE_HMAC_KEY": os.environ.get(
        "RELIQUARY_STATE_HMAC_KEY", "catalyst-local-test-hmac"
    ),
}

IGNORE = [
    "__pycache__", ".venv", ".git", ".pytest_cache", ".mypy_cache",
    "*.pyc", "docs/superpowers", "tests/gpu/__pycache__",
]


image = (
    targon.Image.from_registry(
        "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel",
        add_python="3.11",
    )
    # NO version pins in pip_install args — SDK doesn't escape '>=', it becomes
    # a shell redirect inside the RUN line and breaks the build.
    .pip_install("bittensor", "transformers", "safetensors", "fastapi", "uvicorn",
                 "aiobotocore", "typer", "rich", "tenacity", "pyarrow", "httpx",
                 "huggingface-hub", "datasets", "pydantic")
    .add_local_dir("/home/ubuntu/Catalyst", "/app", ignore=IGNORE)
    .run_commands("cd /app && pip install -e . --no-deps")
    .add_local_dir(
        "/home/ubuntu/Catalyst/wallet_deploy",
        WALLET_PATH,
    )
    .env(ENV_VARS)
)


app = targon.App("reliquary-validator", image=image)


@app.function(
    resource=targon.Compute.H100_SMALL,
    min_replicas=1,
    max_replicas=1,
    timeout=0,
    startup_timeout=1500,
)
@targon.web_server(port=HTTP_PORT, startup_timeout=1500)
def serve():
    import subprocess
    cmd = [
        "python", "-m", "reliquary.cli.main", "validate",
        "--network", NETWORK,
        "--netuid", str(NETUID),
        "--wallet-name", WALLET_NAME,
        "--hotkey", HOTKEY,
        "--checkpoint", CHECKPOINT,
        "--http-host", "0.0.0.0",
        "--http-port", str(HTTP_PORT),
        "--log-level", "INFO",
    ]
    subprocess.Popen(cmd, cwd="/app")
