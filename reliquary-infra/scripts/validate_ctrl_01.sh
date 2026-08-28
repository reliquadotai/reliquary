#!/usr/bin/env bash
set -euo pipefail

EDGE_URL=${EDGE_URL:-http://127.0.0.1:8081}

curl --fail --silent --show-error "${EDGE_URL}/livez" | jq -e '.status == "alive"' >/dev/null
curl --fail --silent --show-error "${EDGE_URL}/readyz" | jq -e '.status == "ready"' >/dev/null
curl --fail --silent --show-error "${EDGE_URL}/state" | jq -e '.window_n | type == "number"' >/dev/null
curl --fail --silent --show-error "${EDGE_URL}/state?env=openmathinstruct" | jq -e '.window_n | type == "number"' >/dev/null
curl --fail --silent --show-error "${EDGE_URL}/health" | jq -e 'type == "object"' >/dev/null
curl --fail --silent --show-error "${EDGE_URL}/runtime-contract" | jq -e '.validator_profile | type == "object"' >/dev/null
curl --fail --silent --show-error "${EDGE_URL}/checkpoint" | jq -e '.checkpoint_n | type == "number"' >/dev/null

unknown_status=$(curl --silent --output /dev/null --write-out '%{http_code}' "${EDGE_URL}/state?env=invalid")
test "${unknown_status}" = 404
write_status=$(curl --silent --output /dev/null --write-out '%{http_code}' -X POST "${EDGE_URL}/submit")
test "${write_status}" = 503
verdict_status=$(curl --silent --output /dev/null --write-out '%{http_code}' "${EDGE_URL}/verdicts/test-hotkey")
test "${verdict_status}" = 503

curl --fail --silent --show-error -H 'Accept-Encoding: gzip' \
  --output /tmp/reliquary-state.gz "${EDGE_URL}/state"
gzip --test /tmp/reliquary-state.gz
rm -f /tmp/reliquary-state.gz

systemctl is-active --quiet reliquary-snapshot-sync nginx prometheus-node-exporter fail2ban auditd docker
ss -lntup | grep -Eq '127\.0\.0\.1:8081'
ss -lntup | grep -Eq '127\.0\.0\.1:9100'
! ss -lntup | grep -Eq '0\.0\.0\.0:(80|443|8081|9100)'
! swapon --show --noheadings | grep -q .
ufw status | grep -q 'Status: active'
! find /root /home -maxdepth 5 -type d -name wallets -path '*/.bittensor/*' -print -quit | grep -q .
test ! -e /etc/reliquary/executor-client/ca.key
test ! -e /etc/reliquary/signer-client/ca.key

echo "ctrl-01 validation passed"
