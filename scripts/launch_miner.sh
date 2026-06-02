#!/bin/bash
# Launch a Reliquary miner in the background. Source scripts/.env first.
#
# Usage:
#     source scripts/.env
#     bash scripts/launch_miner.sh

set -e

INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-/root/Catalyst}"
LOG_FILE="${RELIQUARY_MINER_LOG:-/root/miner.log}"
PID_FILE="${RELIQUARY_MINER_PID:-/root/miner.pid}"

: "${BT_WALLET_NAME?BT_WALLET_NAME not set; source scripts/.env}"
: "${BT_HOTKEY?BT_HOTKEY not set; source scripts/.env}"
ENVIRONMENTS="${RELIQUARY_ENVIRONMENTS:-openmathinstruct}"
WALLET_PATH_ARG=""
if [ -n "${BT_WALLET_PATH:-}" ]; then
  WALLET_PATH_ARG="--wallet-path ${BT_WALLET_PATH}"
fi

cd "$INSTALL_DIR"

find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

pkill -9 -f "reliquary.cli.main mine" 2>/dev/null || true
sleep 1
rm -f "$LOG_FILE" "$PID_FILE"

drand_flag="--use-drand"
if [ "${RELIQUARY_USE_DRAND:-1}" != "1" ]; then
  drand_flag="--no-use-drand"
fi

validator_url_arg=""
if [ -n "${RELIQUARY_VALIDATOR_URL:-}" ]; then
  validator_url_arg="--validator-url ${RELIQUARY_VALIDATOR_URL}"
fi

nohup .venv/bin/python -m reliquary.cli.main mine \
    --network "$BT_NETWORK" \
    --netuid "$NETUID" \
    --wallet-name "$BT_WALLET_NAME" \
    --hotkey "$BT_HOTKEY" \
    --checkpoint "${RELIQUARY_CHECKPOINT:-gpt2}" \
    --environments "$ENVIRONMENTS" \
    --log-level INFO \
    $drand_flag $validator_url_arg $WALLET_PATH_ARG \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Launched miner PID=$(cat "$PID_FILE"), log: $LOG_FILE"
