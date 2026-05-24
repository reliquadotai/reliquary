#!/bin/bash
# Entrypoint for the Reliquary validator image.
#
# Launches:
#   1. Grader server as UID 1001 (no wallet access, no secret env vars)
#   2. Validator main as UID 1000 (owns the hotkey)
set -euo pipefail

: "${BT_WALLET_NAME:?BT_WALLET_NAME is required}"
: "${BT_HOTKEY:?BT_HOTKEY is required}"

# ── Wallet permissions ────────────────────────────────────────────────────
# The host mounts the wallet dir to /home/reliquary/.bittensor. Set
# ownership + mode so UID 1001 cannot read it.
WALLET_DIR="/home/reliquary/.bittensor"
if [[ -d "${WALLET_DIR}" ]]; then
  chown -R 1000:1000 "${WALLET_DIR}"
  chmod -R go-rwx "${WALLET_DIR}"
fi

# ── If grader bundle wasn't built at image-build time, build it now ──────
BUNDLE_ROOTFS="/opt/reliquary/reliquary/environment/grader/bundle/rootfs"
if [[ ! -x "${BUNDLE_ROOTFS}/usr/local/bin/python3" ]]; then
  echo "[entrypoint] Building grader bundle (deferred from image build)..."
  bash /opt/build_grader_bundle.sh
fi

# ── Launch grader server as UID 1001 ─────────────────────────────────────
# Strip secrets from its env so a sandbox escape gains nothing.
echo "[entrypoint] Starting grader server (UID 1001)..."
env -i \
    PATH="/opt/reliquary-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
    HOME="/home/reliquary-grader" \
    GRADER_SOCKET_PATH="/tmp/reliquary-grader.sock" \
    GRADER_BUNDLE_PATH="/opt/reliquary/reliquary/environment/grader/bundle" \
  setpriv --reuid=1001 --regid=1001 --clear-groups --inh-caps=-all \
    python -m reliquary.environment.grader.server --use-runsc &
GRADER_PID=$!
trap 'kill ${GRADER_PID} 2>/dev/null || true' EXIT

# Wait briefly for the grader socket to appear. Hard-fail if it doesn't —
# the validator can't function without the grader, and a silent timeout
# would produce a confusing connection error downstream.
for _ in $(seq 1 30); do
  [[ -S "/tmp/reliquary-grader.sock" ]] && break
  sleep 0.5
done
if [[ ! -S "/tmp/reliquary-grader.sock" ]]; then
  echo "[entrypoint] FATAL: grader socket /tmp/reliquary-grader.sock never appeared within 15s" >&2
  exit 1
fi
chmod 666 /tmp/reliquary-grader.sock

# ── Build the validator argv ─────────────────────────────────────────────
args=(
  --network      "${BT_NETWORK:-finney}"
  --netuid       "${BT_NETUID:-81}"
  --wallet-name  "${BT_WALLET_NAME}"
  --hotkey       "${BT_HOTKEY}"
)

if [[ "${RELIQUARY_TRAIN:-0}" == "1" ]]; then
  : "${RELIQUARY_HF_REPO_ID:?RELIQUARY_HF_REPO_ID required in trainer mode}"
  args+=(
    --train
    --checkpoint   "${RELIQUARY_CHECKPOINT:-Qwen/Qwen3-4B-Instruct-2507}"
    --hf-repo-id   "${RELIQUARY_HF_REPO_ID}"
    --http-host    "${RELIQUARY_HTTP_HOST:-0.0.0.0}"
    --http-port    "${RELIQUARY_HTTP_PORT:-8080}"
  )
  [[ -n "${RELIQUARY_EXTERNAL_IP:-}" ]]   && args+=(--external-ip   "${RELIQUARY_EXTERNAL_IP}")
  [[ -n "${RELIQUARY_EXTERNAL_PORT:-}" ]] && args+=(--external-port "${RELIQUARY_EXTERNAL_PORT}")
  [[ -n "${RELIQUARY_RESUME_FROM:-}" ]]   && args+=(--resume-from   "${RELIQUARY_RESUME_FROM}")
else
  args+=(--no-train)
fi

# ── Launch validator as UID 1000 ─────────────────────────────────────────
echo "[entrypoint] Launching: reliquary validate ${args[*]} (UID 1000)"
exec setpriv --reuid=1000 --regid=1000 --clear-groups --inh-caps=-all \
  reliquary validate "${args[@]}"
