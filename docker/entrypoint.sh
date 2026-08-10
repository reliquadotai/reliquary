#!/bin/bash
# Entrypoint for the Reliquary validator image.
#
# Launches:
#   1. Trusted grader server
#   2. Validator main process
set -euo pipefail

: "${BT_WALLET_NAME:?BT_WALLET_NAME is required}"
: "${BT_HOTKEY:?BT_HOTKEY is required}"

GRADER_PID=""
VALIDATOR_PID=""
CHILD_TERM_GRACE_SECONDS="${RELIQUARY_CHILD_TERM_GRACE_SECONDS:-15}"
if [[ ! "${CHILD_TERM_GRACE_SECONDS}" =~ ^[0-9]+$ ]]; then
  echo "[entrypoint] FATAL: RELIQUARY_CHILD_TERM_GRACE_SECONDS must be a non-negative integer" >&2
  exit 1
fi

signal_children() {
  local signal_name=$1
  local pid
  for pid in "${VALIDATOR_PID}" "${GRADER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "-${signal_name}" "${pid}" 2>/dev/null || true
    fi
  done
}

terminate_children() {
  signal_children TERM
}

children_alive() {
  local pid
  for pid in "${VALIDATOR_PID}" "${GRADER_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      return 0
    fi
  done
  return 1
}

wait_for_children() {
  local deadline=$((SECONDS + CHILD_TERM_GRACE_SECONDS))
  local pid
  while children_alive && (( SECONDS < deadline )); do
    sleep 0.1
  done
  if children_alive; then
    echo "[entrypoint] Child shutdown exceeded ${CHILD_TERM_GRACE_SECONDS}s; sending SIGKILL" >&2
    signal_children KILL
  fi
  for pid in "${VALIDATOR_PID}" "${GRADER_PID}"; do
    if [[ -n "${pid}" ]]; then
      wait "${pid}" 2>/dev/null || true
    fi
  done
}

cleanup() {
  local status=$?
  trap - EXIT TERM INT
  terminate_children
  wait_for_children
  exit "${status}"
}

trap terminate_children TERM INT
trap cleanup EXIT

# ── Credential permissions ────────────────────────────────────────────────
DEFAULT_WALLET_PATH="/opt/reliquary/wallets"
LEGACY_WALLET_PATH="/root/.bittensor/wallets"
if [[ -n "${BT_WALLET_PATH:-}" ]]; then
  WALLET_DIR="${BT_WALLET_PATH}"
elif [[ -d "${DEFAULT_WALLET_PATH}" ]]; then
  WALLET_DIR="${DEFAULT_WALLET_PATH}"
elif [[ -d "${LEGACY_WALLET_PATH}" ]]; then
  WALLET_DIR="${LEGACY_WALLET_PATH}"
else
  WALLET_DIR="${DEFAULT_WALLET_PATH}"
fi
export BT_WALLET_PATH="${WALLET_DIR}"
if [[ ! -d "${WALLET_DIR}" ]]; then
  echo "[entrypoint] FATAL: credential directory is not mounted" >&2
  exit 1
fi
if [[ "${RELIQUARY_DROP_VALIDATOR_PRIVILEGES:-0}" == "1" ]]; then
  read_check=(setpriv --reuid=1000 --regid=1000 --clear-groups test -r "${WALLET_DIR}")
else
  read_check=(test -r "${WALLET_DIR}")
fi
if ! "${read_check[@]}"; then
  echo "[entrypoint] FATAL: credential directory is not readable" >&2
  exit 1
fi

ENVIRONMENTS="${RELIQUARY_ENVIRONMENTS:-openmathinstruct}"

if [[ "${RELIQUARY_TRAIN:-0}" == "1" && ",${ENVIRONMENTS}," == *",opencodeinstruct,"* ]]; then
  # The published image must contain the runsc bundle. Building it at
  # container start would require a Docker socket in the validator container.
  BUNDLE_ROOTFS="/opt/reliquary/reliquary/environment/grader/bundle/rootfs"
  if [[ ! -x "${BUNDLE_ROOTFS}/usr/local/bin/python3" ]]; then
    echo "[entrypoint] FATAL: grader bundle rootfs is missing from the image" >&2
    exit 1
  fi

  # ── Launch grader server ───────────────────────────────────────────────
  echo "[entrypoint] Starting grader server..."
  # runsc needs root inside this privileged container to create its state and
  # cgroups. The trusted supervisor gets a scrubbed env; untrusted miner code
  # still runs inside the runsc worker as the UID/GID from config.json.
  env -i \
      PATH="/opt/reliquary-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin" \
      HOME="/tmp" \
      GRADER_SOCKET_PATH="/tmp/reliquary-grader.sock" \
      GRADER_BUNDLE_PATH="/opt/reliquary/reliquary/environment/grader/bundle" \
    python -m reliquary.environment.grader.server --use-runsc &
  GRADER_PID=$!

  # Wait briefly for the grader socket to appear. Hard-fail if it doesn't.
  for _ in $(seq 1 30); do
    [[ -S "/tmp/reliquary-grader.sock" ]] && break
    sleep 0.5
  done
  if [[ ! -S "/tmp/reliquary-grader.sock" ]]; then
    echo "[entrypoint] FATAL: grader socket /tmp/reliquary-grader.sock never appeared within 15s" >&2
    exit 1
  fi
  chown 0:1000 /tmp/reliquary-grader.sock
  chmod 660 /tmp/reliquary-grader.sock
fi

# ── Build the validator argv ─────────────────────────────────────────────
args=(
  --network      "${BT_NETWORK:-finney}"
  --netuid       "${BT_NETUID:-81}"
  --wallet-name  "${BT_WALLET_NAME}"
  --hotkey       "${BT_HOTKEY}"
  --wallet-path  "${BT_WALLET_PATH}"
  --environments "${ENVIRONMENTS}"
)

if [[ "${RELIQUARY_TRAIN:-0}" == "1" ]]; then
  : "${RELIQUARY_HF_REPO_ID:?RELIQUARY_HF_REPO_ID required in trainer mode}"
  args+=(
    --train
    --checkpoint   "${RELIQUARY_CHECKPOINT:-Qwen/Qwen3.5-2B}"
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

# ── Launch validator ─────────────────────────────────────────────────────
echo "[entrypoint] Launching: reliquary validate ${args[*]}"
if [[ "${RELIQUARY_DROP_VALIDATOR_PRIVILEGES:-0}" == "1" ]]; then
  validator_command=(
    setpriv --reuid=1000 --regid=1000 --clear-groups --inh-caps=-all
    reliquary validate "${args[@]}"
  )
else
  validator_command=(reliquary validate "${args[@]}")
fi

"${validator_command[@]}" &
VALIDATOR_PID=$!

set +e
if [[ -n "${GRADER_PID}" ]]; then
  wait -n "${VALIDATOR_PID}" "${GRADER_PID}"
  status=$?
  if kill -0 "${VALIDATOR_PID}" 2>/dev/null; then
    echo "[entrypoint] FATAL: grader server exited while validator was running" >&2
    [[ "${status}" -ne 0 ]] || status=1
  fi
else
  wait "${VALIDATOR_PID}"
  status=$?
fi
set -e
exit "${status}"
