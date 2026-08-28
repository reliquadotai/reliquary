#!/usr/bin/env bash
# Verify an artifact bundle without starting its service or executing miner code.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "usage: $0 <artifact-directory>" >&2
  exit 2
fi

ARTIFACT_DIR="$1"
MANIFEST="${ARTIFACT_DIR}/manifest.json"
test -f "${MANIFEST}"
jq -e '
  .schema_version == 1 and
  (.git_revision | test("^[0-9a-f]{40}$")) and
  (.image_id | test("^sha256:[0-9a-f]{64}$")) and
  (.grader_runtime_id | test("^grader-sha256:[0-9a-f]{64}$")) and
  .target_platform == "linux/amd64" and
  .sandbox_backend == "gvisor-runsc"
' "${MANIFEST}" >/dev/null

ARCHIVE="${ARTIFACT_DIR}/$(jq -r '.archive' "${MANIFEST}")"
EXPECTED="$(jq -r '.archive_sha256' "${MANIFEST}")"
ACTUAL="$(sha256sum "${ARCHIVE}" | awk '{print $1}')"
test "${ACTUAL}" = "${EXPECTED}"

(
  cd "${ARTIFACT_DIR}"
  sha256sum --check evidence.sha256
)

echo "CPU executor artifact verification passed: ${ARTIFACT_DIR}"
