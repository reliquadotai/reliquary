#!/usr/bin/env bash
set -euo pipefail

EXPECTED_IMAGE_ID="$(cat /etc/reliquary/signer-image-id)"
EXPECTED_BIND_IP="$(sed -n 's/^RELIQUARY_SIGNER_HOST=//p' /etc/reliquary/signer.env)"
EXPECTED_PORT="$(sed -n 's/^RELIQUARY_SIGNER_PORT=//p' /etc/reliquary/signer.env)"
EXPECTED_HOTKEY="$(sed -n 's/^RELIQUARY_SIGNER_EXPECTED_HOTKEY=//p' /etc/reliquary/signer.env)"
EXPECTED_NETUID="$(sed -n 's/^BT_NETUID=//p' /etc/reliquary/signer.env)"
WALLET_NAME="$(sed -n 's/^BT_WALLET_NAME=//p' /etc/reliquary/signer.env)"
HOTKEY_NAME="$(sed -n 's/^BT_HOTKEY=//p' /etc/reliquary/signer.env)"

test "$(hostname)" = signer-01
! swapon --show --noheadings | grep -q .
test "$(systemctl is-system-running)" != failed
systemctl is-active --quiet \
  auditd chrony docker fail2ban prometheus-node-exporter \
  reliquary-signer-egress reliquary-signer
ufw status | grep -q 'Status: active'
nft list table inet reliquary_signer | grep -q 'ct state established,related accept'

test ! -e /etc/reliquary/signer-pki/ca.key
test -f "/var/lib/reliquary-signer/wallets/${WALLET_NAME}/hotkeys/${HOTKEY_NAME}"
test ! -e "/var/lib/reliquary-signer/wallets/${WALLET_NAME}/coldkey"
test ! -e "/var/lib/reliquary-signer/wallets/${WALLET_NAME}/coldkeypub.txt"
test "$(stat -c '%u:%g' "/var/lib/reliquary-signer/wallets/${WALLET_NAME}/hotkeys/${HOTKEY_NAME}")" = "10001:10001"
test "$(stat -c '%a' "/var/lib/reliquary-signer/wallets/${WALLET_NAME}/hotkeys/${HOTKEY_NAME}")" = 600

CONTAINER_ID="$(docker compose \
  --env-file /etc/reliquary/signer.env \
  --file /opt/reliquary-signer/compose.yml ps --quiet reliquary-signer)"
test -n "${CONTAINER_ID}"
test "$(docker inspect --format '{{.State.Status}}' "${CONTAINER_ID}")" = running
test "$(docker inspect --format '{{.State.Health.Status}}' "${CONTAINER_ID}")" = healthy
test "$(docker inspect --format '{{.Image}}' "${CONTAINER_ID}")" = "${EXPECTED_IMAGE_ID}"
test "$(docker inspect --format '{{.Config.User}}' "${CONTAINER_ID}")" = "10001:10001"
test "$(docker inspect --format '{{.HostConfig.Privileged}}' "${CONTAINER_ID}")" = false
test "$(docker inspect --format '{{.HostConfig.ReadonlyRootfs}}' "${CONTAINER_ID}")" = true
test "$(docker inspect --format '{{.HostConfig.NetworkMode}}' "${CONTAINER_ID}")" = host
test "$(docker inspect "${CONTAINER_ID}" | jq '.[0].HostConfig.CapDrop == ["ALL"]')" = true
test "$(docker inspect "${CONTAINER_ID}" | jq '[.[0].Mounts[] | select(.RW == true and .Destination != "/var/lib/reliquary-signer/state")] | length')" = 0

ss -H -lnt | awk '{print $4}' | grep -Fxq "${EXPECTED_BIND_IP}:${EXPECTED_PORT}"
! ss -H -lnt | awk '{print $4}' | grep -Eq "^(0\.0\.0\.0|\[::\]):${EXPECTED_PORT}$"
docker exec -i "${CONTAINER_ID}" python - <<PY
import json
from pathlib import Path

health = json.loads(Path('/tmp/reliquary-signer-health.json').read_text())
assert health['status'] == 'ready'
assert health['signer_hotkey'] == '${EXPECTED_HOTKEY}'
assert health['netuid'] == int('${EXPECTED_NETUID}')
PY

echo "signer-01 validation passed"
