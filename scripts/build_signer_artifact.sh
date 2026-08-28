#!/usr/bin/env bash
# Build one immutable linux/amd64 signer image plus auditable evidence.
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 <new-output-directory> [40-character-git-revision]" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$1"
REVISION="${2:-}"

for command_name in docker gzip jq sha256sum; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "${command_name} is required" >&2
    exit 2
  }
done
if [[ -e "${OUTPUT_DIR}" ]]; then
  echo "refusing to overwrite output directory: ${OUTPUT_DIR}" >&2
  exit 2
fi
if [[ -z "${REVISION}" ]] && git -C "${REPO_DIR}" rev-parse HEAD >/dev/null 2>&1; then
  REVISION="$(git -C "${REPO_DIR}" rev-parse HEAD)"
fi
if [[ ! "${REVISION}" =~ ^[0-9a-f]{40}$ ]]; then
  echo "a full 40-character Git revision is required" >&2
  exit 2
fi
if git -C "${REPO_DIR}" rev-parse HEAD >/dev/null 2>&1; then
  if [[ "$(git -C "${REPO_DIR}" rev-parse HEAD)" != "${REVISION}" ]]; then
    echo "requested revision is not the checked-out revision" >&2
    exit 2
  fi
  if ! git -C "${REPO_DIR}" diff --quiet -- \
    docker/Dockerfile.signer \
    docker/signer-requirements.in \
    docker/signer-requirements.lock \
    reliquary/signer; then
    echo "refusing to build with uncommitted signer source changes" >&2
    exit 2
  fi
fi

mkdir -m 0750 "${OUTPUT_DIR}"
IMAGE_TAG="reliquary-signer:${REVISION}"
ARCHIVE_PATH="${OUTPUT_DIR}/reliquary-signer-${REVISION}.tar.gz"

docker build \
  --progress plain \
  --platform linux/amd64 \
  --network host \
  --build-arg "RELIQUARY_BUILD_REVISION=${REVISION}" \
  --tag "${IMAGE_TAG}" \
  --file "${REPO_DIR}/docker/Dockerfile.signer" \
  "${REPO_DIR}"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
if [[ ! "${IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "invalid image ID returned by Docker: ${IMAGE_ID}" >&2
  exit 1
fi

docker run --rm --network none --entrypoint python "${IMAGE_ID}" \
  -m pip freeze --all > "${OUTPUT_DIR}/python-packages.txt"
docker run --rm --network none --entrypoint python "${IMAGE_ID}" \
  -c 'from reliquary.signer.protocol import SIGNER_PROTOCOL_VERSION; print(SIGNER_PROTOCOL_VERSION)' \
  > "${OUTPUT_DIR}/protocol-version.txt"
docker run --rm --network none --read-only \
  --tmpfs /tmp:rw,nosuid,nodev,noexec,size=16m,mode=0700,uid=10001,gid=10001 \
  --entrypoint python "${IMAGE_ID}" \
  -c 'import bittensor; from reliquary.signer.backend import BittensorSignerBackend; print("signer-import-ok")' \
  > "${OUTPUT_DIR}/runtime-import.txt"
docker image inspect "${IMAGE_ID}" > "${OUTPUT_DIR}/image-inspect.json"
docker history --no-trunc "${IMAGE_ID}" > "${OUTPUT_DIR}/image-history.txt"

if grep -Eiq '(torch|cuda|transformers|huggingface-hub|boto3|aiobotocore|datasets|wandb)' \
  "${OUTPUT_DIR}/python-packages.txt"; then
  echo "forbidden validator/GPU/storage dependency found in signer image" >&2
  exit 1
fi
if docker image inspect --format '{{json .Config.Env}}' "${IMAGE_ID}" \
  | grep -Eiq '(TOKEN=|PASSWORD=|SECRET=|PRIVATE_KEY=|AWS_ACCESS_KEY)'; then
  echo "credential-shaped environment entry found in signer image" >&2
  exit 1
fi

docker save "${IMAGE_TAG}" | gzip -1 > "${ARCHIVE_PATH}"
ARCHIVE_SHA256="$(sha256sum "${ARCHIVE_PATH}" | awk '{print $1}')"
ARCHIVE_SIZE="$(stat -c '%s' "${ARCHIVE_PATH}")"
BUILT_AT="$(date --utc +'%Y-%m-%dT%H:%M:%SZ')"

jq -n \
  --arg schema_version "1" \
  --arg built_at "${BUILT_AT}" \
  --arg revision "${REVISION}" \
  --arg image_tag "${IMAGE_TAG}" \
  --arg image_id "${IMAGE_ID}" \
  --arg archive "$(basename "${ARCHIVE_PATH}")" \
  --arg archive_sha256 "${ARCHIVE_SHA256}" \
  --argjson archive_bytes "${ARCHIVE_SIZE}" \
  '{
    schema_version: ($schema_version | tonumber),
    built_at: $built_at,
    git_revision: $revision,
    image_tag: $image_tag,
    image_id: $image_id,
    signer_protocol_version: 1,
    archive: $archive,
    archive_sha256: $archive_sha256,
    archive_bytes: $archive_bytes,
    target_platform: "linux/amd64",
    role: "signer-01"
  }' > "${OUTPUT_DIR}/manifest.json"

sha256sum \
  "${OUTPUT_DIR}/manifest.json" \
  "${OUTPUT_DIR}/image-inspect.json" \
  "${OUTPUT_DIR}/image-history.txt" \
  "${OUTPUT_DIR}/python-packages.txt" \
  "${OUTPUT_DIR}/protocol-version.txt" \
  "${OUTPUT_DIR}/runtime-import.txt" \
  > "${OUTPUT_DIR}/evidence.sha256"
chmod -R go-w "${OUTPUT_DIR}"

echo "Signer artifact ready: ${OUTPUT_DIR}"
jq . "${OUTPUT_DIR}/manifest.json"
