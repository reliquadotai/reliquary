#!/usr/bin/env bash
# Generate a small private CA plus one CPU-executor server identity and one
# validator-control client identity. Run on an operator workstation, not either
# production host. The CA private key is never copied to a server.
set -euo pipefail
umask 077

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <output-dir> <executor-private-ip-or-dns>" >&2
  exit 2
fi

OUTPUT_DIR="$1"
EXECUTOR_HOST="$2"
LEAF_DAYS="${RELIQUARY_PKI_LEAF_DAYS:-90}"

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
if [[ ! "${LEAF_DAYS}" =~ ^[0-9]+$ ]] || (( LEAF_DAYS < 1 || LEAF_DAYS > 397 )); then
  echo "RELIQUARY_PKI_LEAF_DAYS must be within [1, 397]" >&2
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
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
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
  -days "${LEAF_DAYS}" \
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
subjectKeyIdentifier=hash
authorityKeyIdentifier=keyid,issuer
subjectAltName=URI:spiffe://reliquary.internal/control/ctrl-01
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
  -days "${LEAF_DAYS}" \
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
  -purpose sslserver \
  -CAfile "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/cpu-executor/server.crt"
openssl verify \
  -purpose sslclient \
  -CAfile "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/grader-client/client.crt"
if [[ "${SAN_VALUE}" == IP:* ]]; then
  openssl verify \
    -verify_ip "${EXECUTOR_HOST}" \
    -CAfile "${OUTPUT_DIR}/ca/ca.crt" \
    "${OUTPUT_DIR}/cpu-executor/server.crt"
else
  openssl verify \
    -verify_hostname "${EXECUTOR_HOST}" \
    -CAfile "${OUTPUT_DIR}/ca/ca.crt" \
    "${OUTPUT_DIR}/cpu-executor/server.crt"
fi

echo "PKI generated in ${OUTPUT_DIR}"
echo "keep offline: ${OUTPUT_DIR}/ca/ca.key"
echo "copy to cpu-exec-01: ${OUTPUT_DIR}/cpu-executor/"
echo "copy to ctrl-01: ${OUTPUT_DIR}/grader-client/"
echo "leaf validity: ${LEAF_DAYS} days; rotate before expiry"
