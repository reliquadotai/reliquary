#!/usr/bin/env bash
# Build one immutable linux/amd64 CPU-executor artifact and evidence bundle.
# Run only from a trusted source checkout or a trusted ctrl-01 build directory.
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "usage: $0 <new-output-directory> [40-character-git-revision]" >&2
  exit 2
fi

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="$1"
REVISION="${2:-}"
RUNSC_RELEASE=20260817
RUNSC_SHA256=048b89aada69dc3333422e139d6e9d02f8ab06bda52398060e0fbdacca00074c
RUNSC_SHA512=84936438d583ec976800f464e75a83e1515f0890b451b9b4db219c4472b54ca9b106a6772ee683f1e64cce2128871d7637b14d800591f8451b8137f6c39fb2ef

for command_name in docker gzip jq sha256sum; do
  if ! command -v "${command_name}" >/dev/null 2>&1; then
    echo "${command_name} is required" >&2
    exit 2
  fi
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
    docker/Dockerfile.cpu-executor \
    docker/cpu-executor-requirements.in \
    docker/cpu-executor-requirements.lock \
    reliquary; then
    echo "refusing to build with uncommitted executor source changes" >&2
    exit 2
  fi
fi

mkdir -m 0750 "${OUTPUT_DIR}"
IMAGE_TAG="reliquary-cpu-executor:${REVISION}"
ARCHIVE_PATH="${OUTPUT_DIR}/reliquary-cpu-executor-${REVISION}.tar.gz"

docker build \
  --progress plain \
  --platform linux/amd64 \
  --build-arg "RELIQUARY_BUILD_REVISION=${REVISION}" \
  --build-arg "RUNSC_RELEASE=${RUNSC_RELEASE}" \
  --build-arg "RUNSC_SHA256=${RUNSC_SHA256}" \
  --build-arg "RUNSC_SHA512=${RUNSC_SHA512}" \
  --tag "${IMAGE_TAG}" \
  --file "${REPO_DIR}/docker/Dockerfile.cpu-executor" \
  "${REPO_DIR}"

IMAGE_ID="$(docker image inspect --format '{{.Id}}' "${IMAGE_TAG}")"
if [[ ! "${IMAGE_ID}" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "invalid image ID returned by Docker: ${IMAGE_ID}" >&2
  exit 1
fi

RUNTIME_ID="$(
  docker run --rm --network none --entrypoint python "${IMAGE_ID}" \
    -c 'from reliquary.environment.grader.executor import grader_runtime_id; print(grader_runtime_id())'
)"
RUNSC_VERSION="$(
  docker run --rm --network none --entrypoint runsc "${IMAGE_ID}" --version
)"
docker run --rm --network none --entrypoint python "${IMAGE_ID}" \
  -m pip freeze --all > "${OUTPUT_DIR}/python-packages.txt"
docker image inspect "${IMAGE_ID}" > "${OUTPUT_DIR}/image-inspect.json"
docker history --no-trunc "${IMAGE_ID}" > "${OUTPUT_DIR}/image-history.txt"
printf '%s\n' "${RUNSC_VERSION}" > "${OUTPUT_DIR}/runsc-version.txt"

if grep -Eiq '(bittensor|torch|cuda|huggingface-hub|boto3|aiobotocore)' \
  "${OUTPUT_DIR}/python-packages.txt"; then
  echo "forbidden validator/GPU/storage dependency found in executor image" >&2
  exit 1
fi
if docker image inspect --format '{{json .Config.Env}}' "${IMAGE_ID}" \
  | grep -Eiq '(TOKEN=|PASSWORD=|SECRET=|PRIVATE_KEY=|AWS_ACCESS_KEY)'; then
  echo "credential-shaped environment entry found in executor image" >&2
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
  --arg runtime_id "${RUNTIME_ID}" \
  --arg archive "$(basename "${ARCHIVE_PATH}")" \
  --arg archive_sha256 "${ARCHIVE_SHA256}" \
  --argjson archive_bytes "${ARCHIVE_SIZE}" \
  --arg runsc_release "${RUNSC_RELEASE}" \
  --arg runsc_sha256 "${RUNSC_SHA256}" \
  --arg runsc_sha512 "${RUNSC_SHA512}" \
  '{
    schema_version: ($schema_version | tonumber),
    built_at: $built_at,
    git_revision: $revision,
    image_tag: $image_tag,
    image_id: $image_id,
    grader_runtime_id: $runtime_id,
    archive: $archive,
    archive_sha256: $archive_sha256,
    archive_bytes: $archive_bytes,
    target_platform: "linux/amd64",
    sandbox_backend: "gvisor-runsc",
    runsc_release: $runsc_release,
    runsc_sha256: $runsc_sha256,
    runsc_sha512: $runsc_sha512
  }' > "${OUTPUT_DIR}/manifest.json"

sha256sum \
  "${OUTPUT_DIR}/manifest.json" \
  "${OUTPUT_DIR}/image-inspect.json" \
  "${OUTPUT_DIR}/image-history.txt" \
  "${OUTPUT_DIR}/python-packages.txt" \
  "${OUTPUT_DIR}/runsc-version.txt" \
  > "${OUTPUT_DIR}/evidence.sha256"
chmod -R go-w "${OUTPUT_DIR}"

echo "CPU executor artifact ready: ${OUTPUT_DIR}"
jq . "${OUTPUT_DIR}/manifest.json"
