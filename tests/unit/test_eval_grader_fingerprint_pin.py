"""The grader fingerprint is part of every sealed set card the platform offers.

``grader_sha256()`` (``eval/protocol.py``) hashes the grading code. An admin running another fingerprint refuses a
new order on a card sealed under this one at job creation (#547), so a change to any file it hashes makes every
sealed single-turn set of the catalog unorderable from the next admin deploy, until each one is sealed again under a
new set id (``docs/runbooks/order-control.md``, "Before switching", step 4). The pin makes that change deliberate.
"""

from reliquary.eval.protocol import grader_sha256

# The fingerprint the sealed single-turn cards of the catalog carry (scripts/seal_check_admin.py, SETS).
CATALOG_GRADER_SHA256 = "c0b08e92a5a1eab62255fac031af2dc44e815d85323c3e8be57d3326ef1a260d"


def test_the_grader_fingerprint_is_the_one_the_catalog_cards_are_sealed_with():
    assert grader_sha256() == CATALOG_GRADER_SHA256, (
        "This change alters grader_sha256(). Deployed to the admin, it gets every sealed single-turn set the platform "
        "offers refused at job creation (set protocol grader drift). If the grading change is meant, update "
        "CATALOG_GRADER_SHA256 and plan to seal each sealed catalog set again under a new id once the admin runs it "
        "(docs/runbooks/order-control.md, step 4); otherwise leave the files grader_sha256() hashes as they are.")
