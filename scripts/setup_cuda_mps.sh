#!/bin/bash
# Start the CUDA MPS control daemon on a validator host.
#
# Proof slots (RELIQUARY_PROOF_SLOTS_PER_DEVICE > 1) only overlap on one card
# through an MPS server. Without it their CUDA contexts time-slice: four slots
# measured 8.3 s over 192 archived rollouts against 5.7 s with MPS. Nothing in
# the CUDA API reports the difference, so this is easy to forget and expensive
# to forget.
#
# Idempotent — safe to re-run, and safe to run on a box with one slot.
#
# Also installs a systemd unit so the daemon comes back after a host
# reboot. Without it the container restarts on its own (restart:
# unless-stopped) while the daemon does not, and the validator quietly
# resumes at ~1.45x instead of ~2x with one WARNING as the only signal.
# Pass --no-service to skip that.
#
# Usage (on the HOST, as root, before `docker compose up -d`):
#     bash scripts/setup_cuda_mps.sh
#
# Then set the SAME path in docker/.env:
#     CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps
# docker-compose.trainer.yml binds that directory into the container and shares
# the host IPC namespace, which MPS also needs.

set -e

INSTALL_SERVICE=1
[ "${1:-}" = "--no-service" ] && INSTALL_SERVICE=0

PIPE_DIR="${CUDA_MPS_PIPE_DIRECTORY:-/tmp/nvidia-mps}"
LOG_DIR="${CUDA_MPS_LOG_DIRECTORY:-/tmp/nvidia-mps-log}"

if ! command -v nvidia-cuda-mps-control >/dev/null 2>&1; then
  echo "[mps] nvidia-cuda-mps-control not found." >&2
  echo "[mps] It ships with the NVIDIA driver. On a host where you cannot" >&2
  echo "[mps] install it (an unprivileged SSH container, for instance), run" >&2
  echo "[mps] RELIQUARY_PROOF_SLOTS_PER_DEVICE=1: extra slots without MPS" >&2
  echo "[mps] buy almost nothing and still cost 10.2 GB of VRAM each." >&2
  exit 1
fi

export CUDA_MPS_PIPE_DIRECTORY="$PIPE_DIR"
export CUDA_MPS_LOG_DIRECTORY="$LOG_DIR"
mkdir -p "$PIPE_DIR" "$LOG_DIR"

if [ -e "$PIPE_DIR/control" ]; then
  echo "[mps] already running — control pipe at $PIPE_DIR/control"
else
  echo "[mps] starting daemon (pipe=$PIPE_DIR log=$LOG_DIR)"
  nvidia-cuda-mps-control -d
  # The daemon forks; the pipe appears a moment later.
  for _ in $(seq 20); do
    [ -e "$PIPE_DIR/control" ] && break
    sleep 0.2
  done
fi

if [ ! -e "$PIPE_DIR/control" ]; then
  echo "[mps] FAILED: no control pipe at $PIPE_DIR/control" >&2
  echo "[mps] check $LOG_DIR/control.log" >&2
  exit 1
fi

# A host reboot clears the pipe but not the bind-mounted directory, and the
# container restarts on its own — so without this the box comes back slower
# with no error anywhere. Best effort: a missing systemd is not a failure.
# Overridable so the install path can be exercised off a real systemd host.
UNIT="${MPS_SYSTEMD_UNIT:-/etc/systemd/system/nvidia-cuda-mps.service}"
if [ "$INSTALL_SERVICE" = "1" ] && command -v systemctl >/dev/null 2>&1; then
  MPS_BIN="$(command -v nvidia-cuda-mps-control)"
  cat > "$UNIT" <<UNITEOF
[Unit]
Description=CUDA MPS control daemon (Reliquary proof slots)
After=nvidia-persistenced.service
Before=docker.service

[Service]
Type=forking
Environment=CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR
Environment=CUDA_MPS_LOG_DIRECTORY=$LOG_DIR
ExecStartPre=/bin/mkdir -p $PIPE_DIR $LOG_DIR
ExecStart=$MPS_BIN -d
ExecStop=/bin/sh -c 'echo quit | $MPS_BIN'
Restart=on-failure

[Install]
WantedBy=multi-user.target
UNITEOF
  systemctl daemon-reload
  systemctl enable nvidia-cuda-mps.service >/dev/null 2>&1 || true
  echo "[mps] systemd unit installed and enabled: $UNIT"
else
  echo "[mps] NO systemd unit installed. The daemon will NOT survive a host"
  echo "[mps] reboot, and the container will restart without it — slower, with"
  echo "[mps] only a boot WARNING to say so. Re-run this script after a reboot."
fi

echo "[mps] ready. Put this in docker/.env:"
echo "[mps]     CUDA_MPS_PIPE_DIRECTORY=$PIPE_DIR"
echo "[mps]     RELIQUARY_IPC_MODE=host"
echo "[mps] The control pipe proves the daemon is up, NOT that the container"
echo "[mps] can reach it. Confirm the real gain by timing a window at 1 slot"
echo "[mps] against N slots — a broken IPC namespace fails silently."
