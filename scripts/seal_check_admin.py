#!/usr/bin/env python
"""Read-only seal check of the single-turn evaluation cards against the grader installed here.

Run it where single-turn rows are graded: in the running admin API container and, before switching to a rebuilt
image, in that image, both runs reading the script from the same Core ``<commit>`` (docs/runbooks/order-control.md,
"Before switching"). ``git show`` writes it to ``SEAL``, and both runs read it from that file, never through a pipe:
bash 3.2 judges a Ctrl-C by a pipeline's first command, and ``git show`` has already exited 0, so it would run no
``INT`` trap and the check would go on.

In the rebuilt image, run it only in a one-shot container that replaces the entrypoint, starts no service and is
given nothing but ``R2_ENV``: a file holding exactly what the ``grep`` below keeps, of the six variables
``R2_ACCOUNT_ID``, ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``, ``R2_BUCKET_ID``, ``R2_ENDPOINT_URL`` and
``R2_REGION`` those the admin sets, and never an ``R2_PUBLIC_*`` one. ``R2_ENV`` and ``SEAL`` are mode-600 files
created right before the check, and the ``EXIT`` trap removes both, interrupted or not: the ``INT``, ``TERM`` and
``HUP`` traps make the shell run it on an interrupt and end the check with status 130, 143 or 129, not 0 (in bash, a
Ctrl-C that the running command survives runs no ``INT`` trap, and the check goes on if that command exits 0). Run
the subshell on its own, never chained with the switch: chained with ``&&``, ``||`` or ``if``, the shell ignores its
``set -e``. That ``grep`` copies the admin's own keys, so it runs on the admin's host only; anywhere else, write the
same variables by hand, and the keys must be those of an R2 token that can only read the bucket. ``IMG`` is the
rebuilt image by the digest its one push printed, the image the switch then runs::

    (
      set -e -o pipefail
      R2_ENV="$(mktemp)"; SEAL="$(mktemp)"; trap 'rm -f "$R2_ENV" "$SEAL"' EXIT
      trap 'exit 130' INT; trap 'exit 143' TERM; trap 'exit 129' HUP
      git show <commit>:scripts/seal_check_admin.py > "$SEAL"
      docker exec <admin-container> env |
        grep -E '^R2_(ACCOUNT_ID|ACCESS_KEY_ID|SECRET_ACCESS_KEY|BUCKET_ID|ENDPOINT_URL|REGION)=' > "$R2_ENV"
      docker exec -i <admin-container> python - < "$SEAL"
      docker run --rm -i --entrypoint python --env-file "$R2_ENV" "$IMG" - < "$SEAL"
    )

Never start that image with production settings before the switch: its entrypoint is the order control itself, and
``admin serve`` would be a second admin (a second cap resizer with ``RELIQUARY_ORDER_CAP_SIZING=on``). No setting
makes the bucket read-only: the store uses the ``R2_*`` credentials it is given.

It only calls ``SubnetEvalStore.get_bytes`` on each set's ``set.json`` and reads installed package metadata and
files: no write, no network call of its own beyond the store's read, no subprocess. One JSON line per set:
``grader_ok`` compares the card's sealed ``protocol.source.grader_sha256`` with ``grader_sha256()`` here (checked
first at grading, ``eval/protocol.py`` ``validate_set``); ``taskset_ok`` compares the card's pinned package version,
verifiers version and artifact hash with what is installed. A card with ``"sealed": false`` has no protocol: neither
comparison applies to it, both read ``null``, and only grading one row tells whether that set grades. This is the
field-level check; the sufficient check scores one row of ``grading.jsonl`` through the grader's own entry points.

``SETS`` is the single-turn catalog as of the commit the script is read from. Set ids given as arguments replace it
(``python - <set id>... < "$SEAL"``): a set published since, or a set that commit does not list.
"""
import asyncio, json, sys
from importlib.metadata import version
from reliquary.eval.protocol import grader_sha256
from reliquary.eval.storage import SubnetEvalStore, subnet_key
from reliquary.eval.verifiers_source import installed_artifacts

# AIME 2024, Apex Shortlist and MATH-500 under new ids sealed with grader c0b08e92, their earlier cards having
# drifted (runbook, step 4); GraphWalks and OpenAI MRCR withdrawn from the catalog.
SETS = """verifiers-aime24-r0-n30-gc0b08e92 verifiers-aime25-r0-n30 verifiers-aime26-r0-n30
verifiers-apex_shortlist-r0-n48-gc0b08e92 verifiers-gpqa-ab6f143e1-r0-n198 verifiers-ifbench-r0-n300
verifiers-livecodebench-r0-n454-k300-s7 verifiers-math500-r0-n500-gc0b08e92 verifiers-mmlu-pro-r0-n12032-k500-s7
verifiers-scicode-steps-a024c0a1e-r0-n64""".split()


async def main(set_ids):
    store, here = SubnetEvalStore(), grader_sha256()
    print(json.dumps({"grader_sha256_here": here, "verifiers_here": version("verifiers")}))
    for set_id in set_ids:
        body = await store.get_bytes(subnet_key(set_id, "set.json"))
        if body is None:
            print(json.dumps({"set": set_id, "card": None}))
            continue
        card = json.loads(body)
        taskset = card.get("taskset") or {}
        sealed = card.get("protocol") is not None
        card_grader = ((card.get("protocol") or {}).get("source") or {}).get("grader_sha256")
        want = [taskset.get("package_version"), taskset.get("verifiers_version"), taskset.get("artifact_sha256")]
        try:
            installed = [version(taskset["package"]), version("verifiers"), installed_artifacts(taskset["package"])]
        except Exception as exc:
            installed = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"set": set_id, "sealed": sealed, "package": taskset.get("package"),
                          "card_grader_sha256": card_grader, "grader_ok": card_grader == here if sealed else None,
                          "card_taskset": want, "installed": installed,
                          "taskset_ok": installed == want if sealed else None}))


asyncio.run(main(sys.argv[1:] or SETS))
