#!/usr/bin/env bash
# One entry point for all three CPU roles. The inventory is the only switch.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "usage: $0 <ctrl-01|cpu-exec-01|signer-01> <inventory.yml> [ansible arguments...]" >&2
  exit 2
fi

ROLE="$1"
INVENTORY_INPUT="$2"
shift 2
INFRA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

case "${ROLE}" in
  ctrl-01)
    PLAYBOOK="${INFRA_DIR}/playbooks/ctrl-01.yml"
    ;;
  cpu-exec-01)
    PLAYBOOK="${INFRA_DIR}/playbooks/cpu-exec-01.yml"
    ;;
  signer-01)
    PLAYBOOK="${INFRA_DIR}/playbooks/signer-01.yml"
    ;;
  *)
    echo "unknown role: ${ROLE}" >&2
    exit 2
    ;;
esac

test -f "${INVENTORY_INPUT}" || {
  echo "inventory does not exist: ${INVENTORY_INPUT}" >&2
  exit 2
}
INVENTORY_DIR="$(cd "$(dirname "${INVENTORY_INPUT}")" && pwd)"
INVENTORY="${INVENTORY_DIR}/$(basename "${INVENTORY_INPUT}")"
for command_name in ansible-inventory ansible-playbook; do
  command -v "${command_name}" >/dev/null 2>&1 || {
    echo "${command_name} is required" >&2
    exit 2
  }
done

cd "${INFRA_DIR}"
ANSIBLE_CONFIG="${INFRA_DIR}/ansible.cfg" \
  ansible-inventory --inventory "${INVENTORY}" --host "${ROLE}" >/dev/null
ANSIBLE_CONFIG="${INFRA_DIR}/ansible.cfg" \
  ansible-playbook --inventory "${INVENTORY}" --syntax-check "${PLAYBOOK}"
ANSIBLE_CONFIG="${INFRA_DIR}/ansible.cfg" \
  exec ansible-playbook --inventory "${INVENTORY}" --limit "${ROLE}" \
    "${PLAYBOOK}" "$@"
