#!/usr/bin/env bash
set -euo pipefail

EXPECTED_IMAGE_ID="$(cat /etc/reliquary/cpu-executor-image-id)"
EXPECTED_BIND_IP="$(sed -n 's/^RELIQUARY_CPU_EXECUTOR_HOST=//p' /etc/reliquary/cpu-executor.env)"
EXPECTED_PORT="$(sed -n 's/^RELIQUARY_CPU_EXECUTOR_PORT=//p' /etc/reliquary/cpu-executor.env)"

test "$(hostname)" = cpu-exec-01
test -c /dev/kvm
grep -Eq '\b(vmx|svm)\b' /proc/cpuinfo
! swapon --show --noheadings | grep -q .
test "$(systemctl is-system-running)" != failed
systemctl is-active --quiet \
  auditd chrony docker fail2ban prometheus-node-exporter \
  reliquary-executor-egress reliquary-cpu-executor
ufw status | grep -q 'Status: active'
nft list table inet reliquary_executor | grep -q 'ct state established,related accept'

! find /root /home -maxdepth 4 -type d -name wallets -path '*/.bittensor/*' -print -quit | grep -q .
! find /etc/reliquary -type f \( \
  -iname '*wallet*' -o -iname '*hotkey*' -o -iname '*coldkey*' -o \
  -iname '*provider*key*' -o -iname '*r2*key*' -o -iname '*huggingface*token*' \
\) -print -quit | grep -q .
test ! -e /etc/reliquary/cpu-executor-pki/ca.key

CONTAINER_ID="$(docker compose \
  --env-file /etc/reliquary/cpu-executor.env \
  --file /opt/reliquary-cpu-executor/compose.yml ps --quiet reliquary-cpu-executor)"
test -n "${CONTAINER_ID}"
test "$(docker inspect --format '{{.State.Status}}' "${CONTAINER_ID}")" = running
test "$(docker inspect --format '{{.State.Health.Status}}' "${CONTAINER_ID}")" = healthy
test "$(docker inspect --format '{{.Image}}' "${CONTAINER_ID}")" = "${EXPECTED_IMAGE_ID}"
test "$(docker inspect --format '{{.HostConfig.NetworkMode}}' "${CONTAINER_ID}")" = host
test "$(docker inspect --format '{{.HostConfig.Privileged}}' "${CONTAINER_ID}")" = true
test "$(docker inspect "${CONTAINER_ID}" | jq '[
  .[0].Mounts[]
  | select(
      .Type != "bind"
      or .Source != "/etc/reliquary/cpu-executor-pki"
      or .Destination != "/etc/reliquary/pki"
      or .RW != false
    )
] | length')" = 0
test "$(docker inspect "${CONTAINER_ID}" | jq '[
  .[0].Mounts[]
  | select(
      .Type == "bind"
      and .Source == "/etc/reliquary/cpu-executor-pki"
      and .Destination == "/etc/reliquary/pki"
      and .RW == false
    )
] | length')" = 1

ss -H -lnt | awk '{print $4}' | grep -Fxq "${EXPECTED_BIND_IP}:${EXPECTED_PORT}"
! ss -H -lnt | awk '{print $4}' | grep -Eq "^(0\.0\.0\.0|\[::\]):${EXPECTED_PORT}$"
docker exec "${CONTAINER_ID}" runsc --version | grep -q 'runsc version'
docker exec -i "${CONTAINER_ID}" python - <<'PY'
import json
from pathlib import Path

health = json.loads(Path('/tmp/reliquary-cpu-executor-health.json').read_text())
assert health['workers_alive'] == health['pool_size']
assert health['retire_worker_after_batch'] is True
PY

echo "cpu-exec-01 validation passed"
