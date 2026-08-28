#!/usr/bin/env bash
# Generate a dedicated signer CA, signer server leaf, and ctrl-01 client leaf.
# Run only on the operator workstation; never copy ca/ca.key to any server.
set -euo pipefail
umask 077

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <new-output-dir> <signer-private-ip-or-dns>" >&2
  exit 2
fi

OUTPUT_DIR="$1"
SIGNER_HOST="$2"
LEAF_DAYS="${RELIQUARY_PKI_LEAF_DAYS:-90}"

if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite existing output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ ! "${SIGNER_HOST}" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "invalid signer host: ${SIGNER_HOST}" >&2
  exit 2
fi
command -v openssl >/dev/null 2>&1 || {
  echo "openssl is required" >&2
  exit 2
}
if [[ ! "${LEAF_DAYS}" =~ ^[0-9]+$ ]] || (( LEAF_DAYS < 1 || LEAF_DAYS > 397 )); then
  echo "RELIQUARY_PKI_LEAF_DAYS must be within [1, 397]" >&2
  exit 2
fi

WORK_DIR="$(mktemp -d)"
trap 'rm -rf "${WORK_DIR}"' EXIT
mkdir -p "${OUTPUT_DIR}/ca" "${OUTPUT_DIR}/signer" "${OUTPUT_DIR}/signer-client"

openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
  -out "${OUTPUT_DIR}/ca/ca.key"
openssl req -x509 -new -sha256 -days 3650 \
  -key "${OUTPUT_DIR}/ca/ca.key" \
  -subj "/CN=Reliquary Signer CA" \
  -out "${OUTPUT_DIR}/ca/ca.crt"

if [[ "${SIGNER_HOST}" == *:* || "${SIGNER_HOST}" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
  SAN_VALUE="IP:${SIGNER_HOST}"
else
  SAN_VALUE="DNS:${SIGNER_HOST}"
fi

printf '%s\n' \
  'basicConstraints=critical,CA:FALSE' \
  'keyUsage=critical,digitalSignature,keyEncipherment' \
  'extendedKeyUsage=serverAuth' \
  'subjectKeyIdentifier=hash' \
  'authorityKeyIdentifier=keyid,issuer' \
  "subjectAltName=${SAN_VALUE}" > "${WORK_DIR}/server.ext"
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
  -out "${OUTPUT_DIR}/signer/server.key"
openssl req -new -sha256 \
  -key "${OUTPUT_DIR}/signer/server.key" \
  -subj "/CN=${SIGNER_HOST}" \
  -out "${WORK_DIR}/server.csr"
openssl x509 -req -sha256 -days "${LEAF_DAYS}" \
  -in "${WORK_DIR}/server.csr" \
  -CA "${OUTPUT_DIR}/ca/ca.crt" \
  -CAkey "${OUTPUT_DIR}/ca/ca.key" \
  -CAcreateserial \
  -extfile "${WORK_DIR}/server.ext" \
  -out "${OUTPUT_DIR}/signer/server.crt"

printf '%s\n' \
  'basicConstraints=critical,CA:FALSE' \
  'keyUsage=critical,digitalSignature' \
  'extendedKeyUsage=clientAuth' \
  'subjectKeyIdentifier=hash' \
  'authorityKeyIdentifier=keyid,issuer' \
  'subjectAltName=URI:spiffe://reliquary.internal/control/ctrl-01' \
  > "${WORK_DIR}/client.ext"
openssl genpkey -algorithm EC -pkeyopt ec_paramgen_curve:P-256 \
  -out "${OUTPUT_DIR}/signer-client/client.key"
openssl req -new -sha256 \
  -key "${OUTPUT_DIR}/signer-client/client.key" \
  -subj "/CN=reliquary-ctrl-01" \
  -out "${WORK_DIR}/client.csr"
openssl x509 -req -sha256 -days "${LEAF_DAYS}" \
  -in "${WORK_DIR}/client.csr" \
  -CA "${OUTPUT_DIR}/ca/ca.crt" \
  -CAkey "${OUTPUT_DIR}/ca/ca.key" \
  -CAcreateserial \
  -extfile "${WORK_DIR}/client.ext" \
  -out "${OUTPUT_DIR}/signer-client/client.crt"

install -m 0644 "${OUTPUT_DIR}/ca/ca.crt" "${OUTPUT_DIR}/signer/ca.crt"
install -m 0644 "${OUTPUT_DIR}/ca/ca.crt" "${OUTPUT_DIR}/signer-client/ca.crt"
chmod 0600 "${OUTPUT_DIR}/ca/ca.key" \
  "${OUTPUT_DIR}/signer/server.key" \
  "${OUTPUT_DIR}/signer-client/client.key"
chmod 0644 "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/signer/ca.crt" \
  "${OUTPUT_DIR}/signer/server.crt" \
  "${OUTPUT_DIR}/signer-client/ca.crt" \
  "${OUTPUT_DIR}/signer-client/client.crt"

openssl verify -purpose sslserver -CAfile "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/signer/server.crt"
openssl verify -purpose sslclient -CAfile "${OUTPUT_DIR}/ca/ca.crt" \
  "${OUTPUT_DIR}/signer-client/client.crt"
if [[ "${SAN_VALUE}" == IP:* ]]; then
  openssl verify -verify_ip "${SIGNER_HOST}" \
    -CAfile "${OUTPUT_DIR}/ca/ca.crt" "${OUTPUT_DIR}/signer/server.crt"
else
  openssl verify -verify_hostname "${SIGNER_HOST}" \
    -CAfile "${OUTPUT_DIR}/ca/ca.crt" "${OUTPUT_DIR}/signer/server.crt"
fi

echo "Signer PKI generated in ${OUTPUT_DIR}"
echo "keep offline: ${OUTPUT_DIR}/ca/ca.key"
echo "copy to signer-01: ${OUTPUT_DIR}/signer/"
echo "copy to ctrl-01: ${OUTPUT_DIR}/signer-client/"
echo "leaf validity: ${LEAF_DAYS} days; rotate before expiry"
