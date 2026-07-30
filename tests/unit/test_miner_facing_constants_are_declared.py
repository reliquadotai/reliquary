"""Rules that decide whether a miner is accepted, or how much it is paid, must
be readable in the code.

A miner cannot see the validator's environment. An env-only rule is therefore
invisible to the people it governs — and it silently reverts on a clean
redeploy, which is exactly how `DRAND_ROUND_BACKWARD_TOLERANCE` ended up
rejecting honest submissions with `stale_round` while the fix lived only in a
`.env`.

The asymmetry that makes an env override acceptable: it may only ever make the
validator MORE permissive (disable a gate in an emergency). An env var that
*enables* new scoring behaviour can surprise a miner into being ranked or paid
differently than the published rule, so those must be declared constants.
"""

import ast

import pytest

CONSTANTS_PATH = "reliquary/constants.py"

# Values a miner must be able to read to know what it is producing, and under
# what rule it is scored. None of these may be environment-backed.
MINER_FACING = [
    # what the miner must generate
    "MAX_NEW_TOKENS_PROTOCOL_CAP",
    "BFT_THINKING_BUDGET",
    "BFT_ANSWER_BUDGET",
    "BFT_ENABLED",
    "BFT_FORCE_ANSWER",
    "M_ROLLOUTS",
    "T_PROTO",
    "TOP_P_PROTO",
    "TOP_K_PROTO",
    # how long it has to produce it
    "WINDOW_COLLECTION_SECONDS",
    # whether the submission is admitted
    "MAX_TRUNCATED_PER_SUBMISSION",
    "MAX_TRUNCATED_PER_SUBMISSION_BY_ENV",
    "SIGMA_MIN",
    "B_BATCH",
    # how it is scored and ranked
    "CONSERVATIVE_TRUNCATION_VALUE",
    "DIFFICULTY_AUCTION_DELTA",
]


def _assignments(source: str) -> dict[str, str]:
    """Map module constants to their exact right-hand-side source."""
    out: dict[str, str] = {}
    tree = ast.parse(source)
    for node in tree.body:
        name = None
        value = None
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            name = node.targets[0].id
            value = node.value
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
        ):
            name = node.target.id
            value = node.value
        if (
            name is None
            or value is None
            or not name.isupper()
            or len(name) < 3
        ):
            continue
        segment = ast.get_source_segment(source, value)
        if segment is not None:
            out[name] = segment
    return out


@pytest.fixture(scope="module")
def assignments():
    return _assignments(open(CONSTANTS_PATH).read())


@pytest.mark.parametrize("name", MINER_FACING)
def test_miner_facing_constant_is_not_environment_backed(name, assignments):
    assert name in assignments, f"{name} not found in {CONSTANTS_PATH}"
    assert "environ" not in assignments[name], (
        f"{name} decides what a miner produces, whether it is admitted, or how "
        "it is paid — it must be declared in code, not read from the "
        "validator's environment, which miners cannot see"
    )


def test_the_audit_actually_reads_multiline_definitions(assignments):
    """Guard the guard: the parser must see parenthesised definitions, or an
    env-backed constant would pass this suite unnoticed."""
    multiline = [n for n, v in assignments.items() if "\n" in v]
    assert multiline, "parser never captured a multi-line assignment"
    env_backed = [n for n, v in assignments.items() if "environ" in v]
    assert env_backed, "parser found no env-backed constants at all — suspicious"
