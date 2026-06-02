#!/bin/bash
# Launch a Reliquary validator in the background. Source scripts/.env first.
#
# Usage:
#     source scripts/.env
#     bash scripts/launch_validator.sh

set -e

INSTALL_DIR="${RELIQUARY_INSTALL_DIR:-/root/Catalyst}"
LOG_FILE="${RELIQUARY_LOG:-/root/validator.log}"
PID_FILE="${RELIQUARY_PID:-/root/validator.pid}"
HTTP_HOST="${RELIQUARY_HTTP_HOST:-0.0.0.0}"
HTTP_PORT="${RELIQUARY_HTTP_PORT:-8888}"
ENVIRONMENTS="${RELIQUARY_ENVIRONMENTS:-openmathinstruct}"
WALLET_PATH_ARG=""
if [ -n "${BT_WALLET_PATH:-}" ]; then
  WALLET_PATH_ARG="--wallet-path ${BT_WALLET_PATH}"
fi

: "${BT_WALLET_NAME?BT_WALLET_NAME not set; source scripts/.env}"
: "${BT_HOTKEY?BT_HOTKEY not set; source scripts/.env}"

cd "$INSTALL_DIR"

# Editable installs pick up source changes but stale __pycache__ files can
# mask them — clear on every launch.
find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

pkill -9 -f "reliquary.cli.main validate" 2>/dev/null || true
sleep 1
rm -f "$LOG_FILE" "$PID_FILE"

drand_flag="--use-drand"
if [ "${RELIQUARY_USE_DRAND:-1}" != "1" ]; then
  drand_flag="--no-use-drand"
fi

axon_args=""
if [ -n "${RELIQUARY_EXTERNAL_IP:-}" ]; then
  axon_args="--external-ip ${RELIQUARY_EXTERNAL_IP} --external-port ${RELIQUARY_EXTERNAL_PORT:-$HTTP_PORT}"
fi

nohup .venv/bin/python -m reliquary.cli.main validate \
    --network "$BT_NETWORK" \
    --netuid "$NETUID" \
    --wallet-name "$BT_WALLET_NAME" \
    --hotkey "$BT_HOTKEY" \
    --checkpoint "${RELIQUARY_CHECKPOINT:-gpt2}" \
    --environments "$ENVIRONMENTS" \
    --http-host "$HTTP_HOST" \
    --http-port "$HTTP_PORT" \
    --log-level INFO \
    $drand_flag $axon_args $WALLET_PATH_ARG \
    > "$LOG_FILE" 2>&1 &

echo $! > "$PID_FILE"
echo "Launched validator PID=$(cat "$PID_FILE"), log: $LOG_FILE"
