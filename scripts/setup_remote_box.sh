#!/bin/bash
# Bootstrap a fresh Targon SSH container to run a Reliquary validator or miner.
#
# Designed for Ubuntu 24.04 containers with no Python tooling preinstalled and
# an NVIDIA driver version 580 or newer for the CUDA 13 runtime.
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

DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1 | cut -d. -f1)"
if [[ ! "$DRIVER_MAJOR" =~ ^[0-9]+$ ]] || [ "$DRIVER_MAJOR" -lt 580 ]; then
  echo "[setup] CUDA 13 requires NVIDIA driver 580 or newer" >&2
  exit 1
fi

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
.venv/bin/pip install 'torch==2.13.0+cu130' \
    --index-url https://download.pytorch.org/whl/cu130
.venv/bin/pip install -e .

echo "[setup] verify torch sees CUDA"
.venv/bin/python -c "import torch; assert torch.__version__.split('+')[0] == '2.13.0'; assert torch.version.cuda == '13.0'; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))"

mkdir -p "$BT_WALLET_PATH"

# Proof slots need an MPS server to overlap on one card; without it they
# time-slice and buy almost nothing. Read the slot count from docker/.env when
# it exists — on a first run this script clones the repo and that file does not
# exist yet, so the shell environment is the only source. A failure here is
# fatal: N slots without MPS waste 10.2 GB of VRAM each for a few percent.
SLOTS="${RELIQUARY_PROOF_SLOTS_PER_DEVICE:-}"
if [ -z "$SLOTS" ] && [ -f "$INSTALL_DIR/docker/.env" ]; then
  SLOTS=$(grep -E '^[[:space:]]*RELIQUARY_PROOF_SLOTS_PER_DEVICE=' \
    "$INSTALL_DIR/docker/.env" | tail -1 | sed 's/.*=//' | tr -cd '0-9')
fi
if [ "${SLOTS:-1}" -gt 1 ] 2>/dev/null; then
  echo "[setup] proof slots = $SLOTS -> starting CUDA MPS"
  bash "$INSTALL_DIR/scripts/setup_cuda_mps.sh"
else
  echo "[setup] one proof slot (or unset) -> skipping CUDA MPS"
fi

echo "[setup] done. Next: copy your validator signing credentials to BT_WALLET_PATH, then source scripts/.env and run scripts/launch_validator.sh (or launch_miner.sh)."
echo "[setup] Running more than one proof process per GPU? See docs/validating.md \"Proof slots\" and run scripts/setup_cuda_mps.sh on this host."
