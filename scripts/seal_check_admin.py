#!/usr/bin/env python
"""Read-only seal check of the single-turn evaluation cards against the grader installed here.

Run it where single-turn rows are graded, every run reading the script from the same Core ``<commit>``
(docs/runbooks/order-control.md, "Before switching"). In the running admin API container::

    git show <commit>:scripts/seal_check_admin.py | docker exec -i <admin-container> python -

In a rebuilt image, before switching to it, run it only in a one-shot container that replaces the entrypoint,
starts no service and is given nothing but ``R2_ENV``: a mode-600 file holding exactly what the ``grep`` below
keeps, of the six variables ``R2_ACCOUNT_ID``, ``R2_ACCESS_KEY_ID``, ``R2_SECRET_ACCESS_KEY``, ``R2_BUCKET_ID``,
``R2_ENDPOINT_URL`` and ``R2_REGION`` those the admin sets, and never an ``R2_PUBLIC_*`` one. It is created right
before the check, and the traps remove it on exit, interrupted or not. That ``grep`` copies the admin's own keys, so
it runs on the admin's host only; anywhere else, write the same variables by hand, and the keys must be those of an
R2 token that can only read the bucket. ``IMG`` is the rebuilt image by the digest its one push printed, the image
the switch then runs::

    (
      set -e -o pipefail
      R2_ENV="$(mktemp)"; trap 'rm -f "$R2_ENV"' EXIT; trap exit INT TERM HUP
      docker exec <admin-container> env |
        grep -E '^R2_(ACCOUNT_ID|ACCESS_KEY_ID|SECRET_ACCESS_KEY|BUCKET_ID|ENDPOINT_URL|REGION)=' > "$R2_ENV"
      git show <commit>:scripts/seal_check_admin.py |
        docker run --rm -i --entrypoint python --env-file "$R2_ENV" "$IMG" -
    )

Never start that image with production settings before the switch: its entrypoint is the order control itself, and
``admin serve`` would be a second admin (a second cap resizer with ``RELIQUARY_ORDER_CAP_SIZING=on``). No setting
makes the bucket read-only: the store uses the ``R2_*`` credentials it is given.

It only calls ``SubnetEvalStore.get_bytes`` on each set's ``set.json`` and reads installed package metadata and
files: no write, no network call of its own beyond the store's read, no subprocess. One JSON line per set:
``grader_ok`` compares the card's sealed ``protocol.source.grader_sha256`` with ``grader_sha256()`` here (checked
first at grading, ``eval/protocol.py`` ``validate_set``); ``taskset_ok`` compares the card's pinned package version,
verifiers version and artifact hash with what is installed. A card with ``"sealed": false`` has no protocol: neither
comparison applies to it, and only grading one row tells whether that set grades. This is the field-level check;
the sufficient check scores one row of ``grading.jsonl`` through the grader's own entry points.
"""
import asyncio, json
from importlib.metadata import version
from reliquary.eval.protocol import grader_sha256
from reliquary.eval.storage import SubnetEvalStore, subnet_key
from reliquary.eval.verifiers_source import installed_artifacts

SETS = """verifiers-aime24-r0-n30 verifiers-aime25-r0-n30 verifiers-aime26-r0-n30 verifiers-apex_shortlist-r0-n48
verifiers-gpqa-ab6f143e1-r0-n198 verifiers-graphwalks-single-turn-a896ff7f1-r0-n650-k650-s7 verifiers-ifbench-r0-n300
verifiers-livecodebench-r0-n454-k300-s7 verifiers-math500-r0-n500 verifiers-mmlu-pro-r0-n12032-k500-s7
verifiers-openai-mrcr-single-turn-r0-n500-k500-s7 verifiers-scicode-steps-a024c0a1e-r0-n64""".split()


async def main():
    store, here = SubnetEvalStore(), grader_sha256()
    print(json.dumps({"grader_sha256_here": here, "verifiers_here": version("verifiers")}))
    for set_id in SETS:
        body = await store.get_bytes(subnet_key(set_id, "set.json"))
        if body is None:
            print(json.dumps({"set": set_id, "card": None}))
            continue
        card = json.loads(body)
        taskset = card.get("taskset") or {}
        sealed = ((card.get("protocol") or {}).get("source") or {}).get("grader_sha256")
        want = [taskset.get("package_version"), taskset.get("verifiers_version"), taskset.get("artifact_sha256")]
        try:
            installed = [version(taskset["package"]), version("verifiers"), installed_artifacts(taskset["package"])]
        except Exception as exc:
            installed = f"{type(exc).__name__}: {exc}"
        print(json.dumps({"set": set_id, "sealed": card.get("protocol") is not None, "package": taskset.get("package"),
                          "card_grader_sha256": sealed, "grader_ok": sealed == here,
                          "card_taskset": want, "installed": installed, "taskset_ok": installed == want}))


asyncio.run(main())
