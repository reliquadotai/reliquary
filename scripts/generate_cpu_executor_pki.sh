#!/usr/bin/env bash
# Generate a small private CA plus one CPU-executor server identity and one
# validator-control client identity. Run on an operator workstation, not either
# production host. The CA private key is never copied to a server.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <output-dir> <executor-private-ip-or-dns>" >&2
  exit 2
fi

OUTPUT_DIR="$1"
EXECUTOR_HOST="$2"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! "${EXECUTOR_HOST}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "invalid executor host: ${EXECUTOR_HOST}" >&2
  exit 2
fi
if ! command -v openssl >/dev/null 2>&1; then
  echo "openssl is required" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT

mkdir -p \
  "${OUTPUT_DIR}/ca" \
  "${OUTPUT_DIR}/cpu-executor" \
  "${OUTPUT_DIR}/grader-client"

openssl genpkey \
  -algorithm EC \
  -pkeyopt ec_paramgen_curve:P-256 \
  -out "${OUTPUT_DIR}/ca/ca.key"
openssl req \
  -x509 \
  -new \
  -sha256 \
  -days 3650 \
  -key "${OUTPUT_DIR}/ca/ca.key" \
  -subj "/CN=Reliquary CPU Executor CA" \
  -out "${OUTPUT_DIR}/ca/ca.crt"

if [[ "${EXECUTOR_HOST}" == *:* || "${EXECUTOR_HOST}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  SAN_VALUE="IP:${EXECUTOR_HOST}"
else
  SAN_VALUE="DNS:${EXECUTOR_HOST}"
fi

cat > "${WORK_DIR}/server.ext" <<EOF
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature,keyEncipherment
extendedKeyUsage=serverAuth
subjectAltName=${SAN_VALUE}
EOF

openssl genpkey \
  -algorithm EC \
  -pkeyopt ec_paramgen_curve:P-256 \
  -out "${OUTPUT_DIR}/cpu-executor/server.key"
openssl req \
  -new \
  -sha256 \
  -key "${OUTPUT_DIR}/cpu-executor/server.key" \
  -subj "/CN=${EXECUTOR_HOST}" \
  -out "${WORK_DIR}/server.csr"
openssl x509 \
  -req \
  -sha256 \
  -days 365 \
  -in "${WORK_DIR}/server.csr" \
  -CA "${OUTPUT_DIR}/ca/ca.crt" \
  -CAkey "${OUTPUT_DIR}/ca/ca.key" \
  -CAcreateserial \
  -extfile "${WORK_DIR}/server.ext" \
  -out "${OUTPUT_DIR}/cpu-executor/server.crt"

cat > "${WORK_DIR}/client.ext" <<'EOF'
basicConstraints=critical,CA:FALSE
keyUsage=critical,digitalSignature
extendedKeyUsage=clientAuth
EOF

openssl genpkey \
  -algorithm EC \
  -pkeyopt ec_paramgen_curve:P-256 \
  -out "${OUTPUT_DIR}/grader-client/client.key"
openssl req \
  -new \
  -sha256 \
  -key "${OUTPUT_DIR}/grader-client/client.key" \
  -subj "/CN=reliquary-ctrl-01" \
  -out "${WORK_DIR}/client.csr"
openssl x509 \
  -req \
  -sha256 \
  -days 365 \
  -in "${WORK_DIR}/client.csr" \
  -CA "${OUTPUT_DIR}/ca/ca.crt" \
  -CAkey "${OUTPUT_DIR}/ca/ca.key" \
  -CAcreateserial \
  -extfile "${WORK_DIR}/client.ext" \
  -out "${OUTPUT_DIR}/grader-client/client.crt"

install -m 0644 \
  "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/cpu-executor/ca.crt"
install -m 0644 \
  "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/grader-client/ca.crt"
chmod 0600 \
  "${OUTPUT_DIR}/ca/ca.key" \
  "${OUTPUT_DIR}/cpu-executor/server.key" \
  "${OUTPUT_DIR}/grader-client/client.key"
chmod 0644 \
  "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/cpu-executor/ca.crt" \
  "${OUTPUT_DIR}/cpu-executor/server.crt" \
  "${OUTPUT_DIR}/grader-client/ca.crt" \
  "${OUTPUT_DIR}/grader-client/client.crt"

openssl verify \
  -CAfile "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/cpu-executor/server.crt" \
  "${OUTPUT_DIR}/grader-client/client.crt"

echo "PKI generated in ${OUTPUT_DIR}"
echo "keep offline: ${OUTPUT_DIR}/ca/ca.key"
echo "copy to cpu-exec-01: ${OUTPUT_DIR}/cpu-executor/"
echo "copy to ctrl-01: ${OUTPUT_DIR}/grader-client/"
