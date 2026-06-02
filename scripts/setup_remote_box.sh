#!/bin/bash
# Bootstrap a fresh Targon SSH container to run a Reliquary validator or miner.
#
# Designed for Ubuntu 24.04 containers with no Python tooling preinstalled and
# a CUDA 12.8 driver (H100/H200 on Targon at time of writing).
#
# Usage (from the Targon box):
#     curl -fsSL https://raw.githubusercontent.com/romain13190/Catalyst/main/scripts/setup_remote_box.sh | bash
# Or after cloning:
#     bash scripts/setup_remote_box.sh

set -e

REPO_URL="${RELIQUARY_REPO_URL:-https://github.com/romain13190/reliquary.git}"
BRANCH="${RELIQUARY_BRANCH:-main}"
INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-/root/reliquary}"
BT_WALLET_PATH="${BT_WALLET_PATH:-/opt/reliquary/wallets}"

echo "[setup] apt install python3.12-venv + git"
apt update -qq
apt install -y -qq python3.12-venv git

if [ ! -d "$INSTALL_DIR" ]; then
  echo "[setup] git clone $REPO_URL ($BRANCH) -> $INSTALL_DIR"
  git clone --branch "$BRANCH" "$REPO_URL" "$INSTALL_DIR"
else
  echo "[setup] $INSTALL_DIR exists — running git pull"
  (cd "$INSTALL_DIR" && git fetch origin && git checkout "$BRANCH" && git pull)
fi

cd "$INSTALL_DIR"

if [ ! -d .venv ]; then
  echo "[setup] creating venv"
  python3 -m venv .venv
fi

echo "[setup] pip install reliquary"
.venv/bin/pip install --upgrade pip setuptools wheel
.venv/bin/pip install -e .

# The default torch pulled by bittensor is built for CUDA 13, which doesn't
# run against the CUDA 12.8 driver that Targon boxes ship with. Pin to the
# cu128 wheel instead (use --no-deps so bittensor's other deps are untouched).
echo "[setup] pin torch to cu128 wheel for H100/H200 driver"
.venv/bin/pip install --force-reinstall --no-deps 'torch==2.8.0' \
    --index-url https://download.pytorch.org/whl/cu128

echo "[setup] verify torch sees CUDA"
.venv/bin/python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

mkdir -p "$BT_WALLET_PATH"

echo "[setup] done. Next: copy your validator signing credentials to BT_WALLET_PATH, then source scripts/.env and run scripts/launch_validator.sh (or launch_miner.sh)."
