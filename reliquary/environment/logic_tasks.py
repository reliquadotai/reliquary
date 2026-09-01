"""Deterministic logic tasks for ``reliquarylogic_v1``.

Task designs follow the SynLogic families (MIT); the generators here are
Reliquary-authored so the consensus manifest can bind them by file digest.
Every generator is *total*: an index always yields a task, never a retry
loop, because ``get_problem`` has no way to report failure.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

# Reused rather than duplicated. ``records_tasks`` is frozen by its own
# contract, so binding it in this manifest costs nothing.
from reliquary.environment.records_tasks import HashCounterRng
from reliquary.environment.structured_output import canonical_json


ENVIRONMENT_NAME = "reliquarylogic_v1"
TASK_FAMILY = "logic_v1"
GENERATOR_VERSION = "logic-v1"
CHECKER_VERSION = "logic-checker-v1"
VIRTUAL_LENGTH = 1 << 31

_ANSWER_SHAPE = (
    'Return the final value in exactly this answer shape, using JSON types '
    'and no additional keys:\n```json\n{"result": <final value>}\n```'
)


@dataclass(frozen=True, slots=True)
class GeneratedLogicTask:
    prompt: str
    expected: Any
    operation_id: str
    family: str
    difficulty: int
    # "equality" compares the parsed answer to ``expected``; "numbrix_path"
    # accepts any grid satisfying the clues and the path invariants, so the
    # generator never has to prove its puzzle has a unique solution.
    check: str
    constraints: dict[str, Any]


def _rng_for_index(index: int) -> HashCounterRng:
    normalized = int(index) % VIRTUAL_LENGTH
    seed = hashlib.sha256(
        f"{ENVIRONMENT_NAME}\0{GENERATOR_VERSION}\0{normalized}".encode("ascii")
    ).digest()
    return HashCounterRng(seed)


# ────────────────  boolean_expressions  ────────────────

_VARIABLE_NAMES = ("p", "q", "r", "s", "t", "u")


def _boolean_node(
    rng: HashCounterRng,
    depth: int,
    variables: tuple[tuple[str, bool], ...],
) -> tuple[str, bool, int]:
    """Build one expression node, returning (text, value, operator count)."""
    if depth <= 0:
        name, value = variables[rng.randbelow(len(variables))]
        return name, value, 0
    kind = rng.randbelow(3)
    if kind == 0:
        inner, value, ops = _boolean_node(rng, depth - 1, variables)
        return f"not {inner}", not value, ops + 1
    left, left_value, left_ops = _boolean_node(rng, depth - 1, variables)
    right, right_value, right_ops = _boolean_node(rng, depth - 1, variables)
    operator = "and" if kind == 1 else "or"
    value = (
        left_value and right_value if operator == "and"
        else left_value or right_value
    )
    return (
        f"( {left} {operator} {right} )",
        value,
        left_ops + right_ops + 1,
    )


def _boolean_expressions(rng: HashCounterRng) -> GeneratedLogicTask:
    # Named variables rather than bare True/False literals. With literals the
    # reachable expression space is a few thousand strings, which produced a
    # 31% duplicate-prompt rate — and duplicate prompt content is burned
    # silently at seal.
    names = list(_VARIABLE_NAMES)
    rng.shuffle(names)
    count = 3 + rng.randbelow(3)
    variables = tuple(
        (name, rng.randbelow(2) == 1) for name in sorted(names[:count])
    )
    depth = 2 + rng.randbelow(3)
    text, value, operators = _boolean_node(rng, depth, variables)
    assignments = ", ".join(
        f"{name} = {'True' if bound else 'False'}" for name, bound in variables
    )
    prompt = (
        "Evaluate the boolean expression under the given assignment.\n\n"
        f"Assignment: {assignments}\n\n"
        f"```\n{text}\n```\n\n"
        "Answer with the JSON boolean true or false.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=value,
        operation_id="boolean-expression-v1",
        family="boolean_expressions",
        difficulty=min(5, 1 + operators // 2),
        check="equality",
        constraints={},
    )


# ────────────────  numbrix  ────────────────


def _serpentine(size: int) -> list[tuple[int, int]]:
    """Row-major boustrophedon order: always a Hamiltonian path."""
    cells: list[tuple[int, int]] = []
    for row in range(size):
        columns = range(size) if row % 2 == 0 else range(size - 1, -1, -1)
        for column in columns:
            cells.append((row, column))
    return cells


def _neighbours(cell: tuple[int, int], size: int) -> list[tuple[int, int]]:
    row, column = cell
    candidates = ((row - 1, column), (row + 1, column),
                  (row, column - 1), (row, column + 1))
    return [
        (r, c) for r, c in candidates if 0 <= r < size and 0 <= c < size
    ]


def _backbite(
    rng: HashCounterRng, path: list[tuple[int, int]], size: int, moves: int
) -> list[tuple[int, int]]:
    """Randomise a Hamiltonian path without ever searching.

    One backbite joins the head to a random grid neighbour already on the
    path and reverses the prefix in between. The result is always another
    Hamiltonian path, so the move cannot fail and needs no retry loop — the
    property that made the upstream generator unusable here.
    """
    cells = list(path)
    for _ in range(moves):
        if rng.randbelow(2) == 1:
            cells.reverse()
        options = _neighbours(cells[0], size)
        target = options[rng.randbelow(len(options))]
        index = cells.index(target)
        if index > 1:
            cells = cells[:index][::-1] + cells[index:]
    return cells


def _numbrix(rng: HashCounterRng) -> GeneratedLogicTask:
    # 4.7% in band on the base model. Shrinking to 3x3 duplicated 30% of
    # prompts — too few Hamiltonian paths — so difficulty comes down by
    # revealing more clues instead, which leaves the path space untouched.
    size = 4 + rng.randbelow(2)
    # A serpentine alone spans only 16 shapes under the grid symmetries, which
    # left 1.75% of prompts duplicated; backbite mixing removes that.
    order = _backbite(rng, _serpentine(size), size, 4 * size * size)

    solution = [[0] * size for _ in range(size)]
    for step, (row, column) in enumerate(order):
        solution[row][column] = step + 1

    # Reveal both endpoints plus a random interior subset; the endpoints
    # anchor the numbering so the puzzle stays solvable by reasoning.
    revealed = {order[0], order[-1]}
    interior = [cell for cell in order[1:-1]]
    rng.shuffle(interior)
    for cell in interior[: (size * size * 2) // 3]:
        revealed.add(cell)

    grid = [
        [solution[row][column] if (row, column) in revealed else 0
         for column in range(size)]
        for row in range(size)
    ]
    rendered = "\n".join(
        " ".join("." if value == 0 else str(value) for value in row)
        for row in grid
    )
    prompt = (
        f"Complete the {size}x{size} Numbrix grid below. Fill every dot so "
        f"the grid contains each integer from 1 to {size * size} exactly "
        "once, and so consecutive integers sit in orthogonally adjacent "
        "cells. Keep every given number where it is.\n\n"
        f"```\n{rendered}\n```\n\n"
        "Answer with the completed grid as a JSON array of "
        f"{size} arrays of {size} integers.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=solution,
        operation_id="numbrix-path-v1",
        family="numbrix",
        difficulty=size - 1,
        check="numbrix_path",
        constraints={"size": size, "clues": grid},
    )


# ────────────────  cipher (inactive)  ────────────────

_WORDS = (
    "amber", "anchor", "bridge", "candle", "cavern", "cinder", "clover",
    "copper", "cypher", "ember", "falcon", "garnet", "harbor", "hollow",
    "indigo", "ivory", "jasper", "kernel", "lantern", "marble", "meadow",
    "nectar", "obsidian", "onyx", "pewter", "quartz", "quiver", "ripple",
    "saffron", "silver", "summit", "thicket", "timber", "velvet", "walnut",
    "willow", "zenith", "zephyr",
)


def _shift(text: str, amount: int) -> str:
    return "".join(
        chr((ord(character) - 97 + amount) % 26 + 97) for character in text
    )


def _cipher(rng: HashCounterRng) -> GeneratedLogicTask:
    words = list(_WORDS)
    rng.shuffle(words)
    plain = " ".join(words[: 3 + rng.randbelow(3)])
    amount = 1 + rng.randbelow(25)
    encoding = rng.randbelow(2) == 1
    shifted = _shift(plain, amount if encoding else -amount)
    given, expected = (plain, shifted) if encoding else (shifted, plain)
    direction = "Encode" if encoding else "Decode"
    prompt = (
        f"{direction} the message below with a Caesar shift of {amount} "
        f"{'forward' if encoding else 'backward'} through the alphabet. "
        "Spaces are unchanged.\n\n"
        f"```\n{given}\n```\n\n"
        "Answer with the resulting string.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=expected,
        operation_id="caesar-cipher-v1",
        family="cipher",
        difficulty=2 + len(plain.split()) // 2,
        check="equality",
        constraints={},
    )

# ────────────────  dyck_language  ────────────────

_BRACKETS = (("(", ")"), ("[", "]"), ("{", "}"))


def _dyck_language(rng: HashCounterRng) -> GeneratedLogicTask:
    steps = 10 + rng.randbelow(11)
    stack: list[int] = []
    sequence: list[str] = []
    for step in range(steps):
        remaining = steps - step
        # Leave room to stay open at the end; never pop an empty stack.
        if stack and (rng.randbelow(2) == 1 or len(stack) >= remaining):
            sequence.append(_BRACKETS[stack.pop()][1])
        else:
            kind = rng.randbelow(len(_BRACKETS))
            stack.append(kind)
            sequence.append(_BRACKETS[kind][0])
    if not stack:  # degenerate: force one open so the answer is non-empty
        kind = rng.randbelow(len(_BRACKETS))
        stack.append(kind)
        sequence.append(_BRACKETS[kind][0])
    expected = "".join(_BRACKETS[kind][1] for kind in reversed(stack))
    prompt = (
        # Instruction before the content, not after: the same task with the
        # instruction moved below the code block measured 42.2% in band
        # against 67.2% here.
        "Complete the sequence below so every bracket is properly closed, "
        "in the correct order. Answer with only the closing brackets you "
        "would append, and nothing else.\n\n"
        f"```\n{''.join(sequence)}\n```\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=expected,
        operation_id="dyck-completion-v1",
        family="dyck_language",
        difficulty=min(5, 1 + len(expected) // 2),
        check="equality",
        constraints={},
    )


# ────────────────  web_of_lies  ────────────────

_PEOPLE = (
    "Ada", "Bo", "Cleo", "Dmitri", "Elena", "Farid", "Greta", "Hugo",
    "Ines", "Jonas", "Kira", "Liam", "Mira", "Noor", "Osric", "Petra",
)


def _web_of_lies(rng: HashCounterRng) -> GeneratedLogicTask:
    names = list(_PEOPLE)
    rng.shuffle(names)
    count = 4 + rng.randbelow(3)
    chosen = names[:count]
    honest = [rng.randbelow(2) == 1 for _ in chosen]

    lines = [
        f"{chosen[0]} {'tells the truth' if honest[0] else 'lies'}."
    ]
    # Each speaker reports on the next, so the chain is fully determined by
    # the anchor above; a liar states the opposite of what is true.
    for index in range(count - 1):
        claim = honest[index + 1] if honest[index] else not honest[index + 1]
        lines.append(
            f"{chosen[index]} says that {chosen[index + 1]} "
            f"{'tells the truth' if claim else 'lies'}."
        )
    target = 1 + rng.randbelow(count - 1)
    prompt = (
        "Each person either always tells the truth or always lies.\n\n"
        + "\n".join(lines)
        + f"\n\nDoes {chosen[target]} tell the truth? Answer with the JSON "
        "boolean true or false.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=honest[target],
        operation_id="web-of-lies-v1",
        family="web_of_lies",
        difficulty=min(5, count - 1),
        check="equality",
        constraints={},
    )


# ────────────────  cryptarithm  ────────────────

_LETTERS = "ABCDEFGHIJ"


def _cryptarithm(rng: HashCounterRng) -> GeneratedLogicTask:
    # 3-4 digit addends measured 1.6% in band on the base model.
    width = 2 + rng.randbelow(2)
    # Build from a true sum, so the puzzle is solvable without any search.
    left = rng.randbelow(9 * 10 ** (width - 1)) + 10 ** (width - 1)
    right = rng.randbelow(9 * 10 ** (width - 1)) + 10 ** (width - 1)
    total = left + right
    digits = sorted({int(c) for c in f"{left}{right}{total}"})
    mapping = {}
    pool = list(_LETTERS)
    rng.shuffle(pool)
    for position, digit in enumerate(digits):
        mapping[digit] = pool[position]

    def encode(value: int) -> str:
        return "".join(mapping[int(c)] for c in str(value))

    words = [encode(left), encode(right), encode(total)]
    prompt = (
        "Each letter below stands for a distinct decimal digit, and the "
        "addition holds as written. No number has a leading zero.\n\n"
        f"```\n  {words[0]}\n+ {words[1]}\n= {words[2]}\n```\n\n"
        "Answer with a JSON object mapping every letter to its digit.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected={letter: digit for digit, letter in mapping.items()},
        operation_id="cryptarithm-sum-v1",
        family="cryptarithm",
        difficulty=min(5, width),
        check="cryptarithm_sum",
        constraints={"addends": words[:2], "sum": words[2]},
    )


# ────────────────  dyck_language_errors  ────────────────


def _balanced_sequence(rng: HashCounterRng, pairs: int) -> list[str]:
    """Emit a fully balanced sequence; the stack always empties."""
    stack: list[int] = []
    out: list[str] = []
    opened = 0
    while len(out) < pairs * 2:
        remaining = pairs * 2 - len(out)
        must_close = len(stack) >= remaining
        can_open = opened < pairs and not must_close
        if can_open and (not stack or rng.randbelow(2) == 1):
            kind = rng.randbelow(len(_BRACKETS))
            stack.append(kind)
            out.append(_BRACKETS[kind][0])
            opened += 1
        else:
            out.append(_BRACKETS[stack.pop()][1])
    return out


def _dyck_language_errors(rng: HashCounterRng) -> GeneratedLogicTask:
    pairs = 5 + rng.randbelow(6)
    sequence = _balanced_sequence(rng, pairs)
    closers = [
        position for position, character in enumerate(sequence)
        if character in (pair[1] for pair in _BRACKETS)
    ]
    # Swap one closer for a different type: the sequence first becomes
    # invalid at exactly that position, so the answer is unambiguous.
    target = closers[rng.randbelow(len(closers))]
    others = [
        pair[1] for pair in _BRACKETS if pair[1] != sequence[target]
    ]
    sequence[target] = others[rng.randbelow(len(others))]
    prompt = (
        "The bracket sequence below contains exactly one wrong closing "
        "bracket. Reading left to right, give the 1-based position of the "
        "first character at which the sequence becomes invalid.\n\n"
        f"```\n{''.join(sequence)}\n```\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=target + 1,
        operation_id="dyck-error-position-v1",
        family="dyck_language_errors",
        difficulty=min(5, 1 + pairs // 3),
        check="equality",
        constraints={},
    )


# ────────────────  object_properties  ────────────────

_ITEMS = (
    "basket", "bottle", "crate", "flask", "jar", "kettle", "lantern",
    "pouch", "satchel", "tin", "urn", "vase",
)
_COLOURS = ("amber", "blue", "green", "grey", "red", "violet")
_SIZES = ("large", "small", "tiny", "wide")
_TEXTURES = ("rough", "smooth", "sticky", "worn")


def _object_properties(rng: HashCounterRng) -> GeneratedLogicTask:
    items = list(_ITEMS)
    rng.shuffle(items)
    count = 4 + rng.randbelow(3)
    chosen = items[:count]
    # Build the answer first, then give every other item a different colour
    # or size, so exactly one item matches the query by construction.
    target = rng.randbelow(count)
    colour = rng.choice(_COLOURS)
    size = rng.choice(_SIZES)
    rows = []
    for position, item in enumerate(chosen):
        if position == target:
            rows.append((item, colour, size, rng.choice(_TEXTURES)))
            continue
        if rng.randbelow(2) == 1:
            others = tuple(c for c in _COLOURS if c != colour)
            rows.append(
                (item, rng.choice(others), size, rng.choice(_TEXTURES))
            )
        else:
            others = tuple(z for z in _SIZES if z != size)
            rows.append(
                (item, colour, rng.choice(others), rng.choice(_TEXTURES))
            )
    order = list(range(count))
    rng.shuffle(order)
    lines = [
        f"The {rows[i][0]} is {rows[i][1]}, {rows[i][2]} and {rows[i][3]}."
        for i in order
    ]
    prompt = (
        f"Exactly one item below is both {colour} and {size}. Name it.\n\n"
        + "\n".join(lines)
        + f"\n\nAnswer with the item name as a JSON string.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=chosen[target],
        operation_id="object-properties-v1",
        family="object_properties",
        difficulty=min(5, count - 2),
        check="equality",
        constraints={},
    )


# ────────────────  operation  ────────────────


def _operation(rng: HashCounterRng) -> GeneratedLogicTask:
    # Widened from 6/6/11/9: the non-nested branch spanned only ~32k
    # distinct prompts and duplicated 2.35%. Operands stay small so the
    # arithmetic itself does not get harder.
    left_coefficient = 2 + rng.randbelow(10)
    right_coefficient = 2 + rng.randbelow(10)
    offset = rng.randbelow(19) - 9

    def apply(a: int, b: int) -> int:
        return left_coefficient * a + right_coefficient * b + offset

    values = [1 + rng.randbelow(12) for _ in range(3)]
    nested = rng.randbelow(2) == 1
    if nested:
        inner = apply(values[0], values[1])
        expected = apply(inner, values[2])
        expression = f"({values[0]} @ {values[1]}) @ {values[2]}"
    else:
        expected = apply(values[0], values[1])
        expression = f"{values[0]} @ {values[1]}"
    sign = "+" if offset >= 0 else "-"
    prompt = (
        "A new operator is defined as "
        f"a @ b = {left_coefficient}a + {right_coefficient}b "
        f"{sign} {abs(offset)}.\n\n"
        f"Compute {expression}.\n\n"
        "Answer with the resulting integer.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=expected,
        operation_id="custom-operator-v1",
        family="operation",
        difficulty=3 if nested else 2,
        check="equality",
        constraints={},
    )


# ────────────────  time_sequence  ────────────────

_ARRIVERS = (
    "Ada", "Bo", "Cleo", "Dmitri", "Elena", "Farid", "Greta", "Hugo",
    "Ines", "Jonas", "Kira", "Liam",
)


def _time_sequence(rng: HashCounterRng) -> GeneratedLogicTask:
    names = list(_ARRIVERS)
    rng.shuffle(names)
    count = 3 + rng.randbelow(3)
    chosen = names[:count]
    minutes = [7 * 60 + rng.randbelow(180)]
    lines = [
        f"{chosen[0]} arrived at "
        f"{minutes[0] // 60:02d}:{minutes[0] % 60:02d}."
    ]
    for index in range(1, count):
        delta = 5 + rng.randbelow(56)
        after = rng.randbelow(2) == 1
        minutes.append(minutes[index - 1] + (delta if after else -delta))
        lines.append(
            f"{chosen[index]} arrived {delta} minutes "
            f"{'after' if after else 'before'} {chosen[index - 1]}."
        )
    target = 1 + rng.randbelow(count - 1)
    total = minutes[target] % (24 * 60)
    prompt = (
        "\n".join(lines)
        + f"\n\nAt what time did {chosen[target]} arrive? Answer with a "
        'JSON string in 24-hour HH:MM form.\n\n'
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=f"{total // 60:02d}:{total % 60:02d}",
        operation_id="time-sequence-v1",
        family="time_sequence",
        difficulty=min(5, count),
        check="equality",
        constraints={},
    )


# ────────────────  math_path  ────────────────


def _math_path(rng: HashCounterRng) -> GeneratedLogicTask:
    """Multi-step arithmetic with a scalar answer.

    Deliberately deep but flat: it separates whether numbrix and cryptarithm
    scored low because of reasoning depth or because of answer shape.
    """
    size = 4 + rng.randbelow(2)
    grid = [
        [1 + rng.randbelow(20) for _ in range(size)] for _ in range(size)
    ]
    row, column = rng.randbelow(size), rng.randbelow(size)
    steps = 5 + rng.randbelow(4)
    moves, visited = [], [(row, column)]
    for _ in range(steps):
        options = [
            (name, r, c) for name, r, c in (
                ("up", row - 1, column), ("down", row + 1, column),
                ("left", row, column - 1), ("right", row, column + 1),
            ) if 0 <= r < size and 0 <= c < size
        ]
        name, row, column = options[rng.randbelow(len(options))]
        moves.append(name)
        visited.append((row, column))
    total = sum(grid[r][c] for r, c in visited)
    rendered = "\n".join(" ".join(f"{value:2d}" for value in line) for line in grid)
    start = visited[0]
    prompt = (
        f"Start on the cell at row {start[0] + 1}, column {start[1] + 1} of "
        f"the {size}x{size} grid below, then make these moves in order: "
        f"{', '.join(moves)}. Rows are numbered from the top and columns "
        "from the left. Add up every cell you stand on, including the "
        "starting cell and the cell after each move.\n\n"
        f"```\n{rendered}\n```\n\n"
        "Answer with the total as an integer.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=total,
        operation_id="math-path-sum-v1",
        family="math_path",
        difficulty=min(5, 1 + steps // 2),
        check="equality",
        constraints={},
    )


# ────────────────  space_reasoning  ────────────────

_SHELF_ITEMS = (
    "atlas", "almanac", "diary", "folio", "gazette", "journal", "ledger",
    "manual", "novel", "primer", "register", "songbook",
)


def _space_reasoning(rng: HashCounterRng) -> GeneratedLogicTask:
    items = list(_SHELF_ITEMS)
    rng.shuffle(items)
    count = 4 + rng.randbelow(3)
    shelf = items[:count]
    # The arrangement is chosen first; the clues are read off it, so the
    # puzzle is consistent by construction and never needs solving.
    lines = [f"The {shelf[0]} is at the far left."]
    for index in range(count - 1):
        lines.append(
            f"The {shelf[index + 1]} is immediately to the right of "
            f"the {shelf[index]}."
        )
    order = list(range(len(lines)))
    rng.shuffle(order)
    position = 1 + rng.randbelow(count)
    prompt = (
        f"{count} books sit in a row on a shelf.\n\n"
        + "\n".join(lines[i] for i in order)
        + f"\n\nWhich book is number {position} counting from the left? "
        "Answer with the book name as a JSON string.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=shelf[position - 1],
        operation_id="shelf-order-v1",
        family="space_reasoning",
        difficulty=min(5, count - 2),
        check="equality",
        constraints={},
    )


# ────────────────  word_sorting_mistake  ────────────────


def _word_sorting_mistake(rng: HashCounterRng) -> GeneratedLogicTask:
    pool = list(_WORDS)
    rng.shuffle(pool)
    count = 6 + rng.randbelow(5)
    words = sorted(pool[:count])
    # Swapping one adjacent pair makes exactly one word sit before a word
    # that should precede it, so the first out-of-order word is unique.
    cut = rng.randbelow(count - 1)
    words[cut], words[cut + 1] = words[cut + 1], words[cut]
    prompt = (
        "The words below should be in alphabetical order, but exactly one "
        "adjacent pair has been swapped. Reading left to right, name the "
        "first word that appears out of order.\n\n"
        f"```\n{', '.join(words)}\n```\n\n"
        "Answer with that word as a JSON string.\n\n"
        f"{_ANSWER_SHAPE}"
    )
    return GeneratedLogicTask(
        prompt=prompt,
        expected=words[cut],
        operation_id="word-sorting-mistake-v1",
        family="word_sorting_mistake",
        difficulty=min(5, 1 + count // 3),
        check="equality",
        constraints={},
    )


@dataclass(frozen=True, slots=True)
class Family:
    """One task family and whether the contract currently draws from it.

    ``band`` records the share of 16-rollout groups clearing SIGMA_MIN on
    Qwen3-4B-Base under the profile's sampling contract, so the roster
    carries the evidence behind each decision. ``-1.0`` means not yet
    measured.
    """

    name: str
    generator: Any
    active: bool
    band: float


# Toggling ``active`` REMAPS THE WHOLE INDEX SPACE: the family is drawn with
# a modulo over the active list, so a different roster is a different corpus.
# Never compare band measurements taken under different rosters.
_FAMILY_REGISTRY = (
    Family("boolean_expressions", _boolean_expressions, True, 0.984),
    Family("cipher", _cipher, False, 0.000),
    Family("cryptarithm", _cryptarithm, True, 0.156),
    Family("dyck_language", _dyck_language, True, 0.641),
    Family("dyck_language_errors", _dyck_language_errors, True, 0.531),
    Family("math_path", _math_path, True, 0.438),
    Family("numbrix", _numbrix, True, 0.219),
    Family("object_properties", _object_properties, True, 0.969),
    Family("operation", _operation, True, 0.984),
    Family("space_reasoning", _space_reasoning, True, 0.984),
    Family("time_sequence", _time_sequence, True, 0.984),
    Family("web_of_lies", _web_of_lies, True, 1.000),
    Family("word_sorting_mistake", _word_sorting_mistake, True, 0.688),
)

# cipher is inactive: measured 0.0% in band with 76% valid JSON and 99.5%
# EOS, i.e. well-formed and always wrong. Character-level shifting resists
# tokenisation. Kept implemented because that verdict is against the BASE
# model; a trained checkpoint may reach it.

_GENERATORS = tuple(
    family.generator for family in _FAMILY_REGISTRY if family.active
)
if not _GENERATORS:
    raise ValueError("at least one logic family must be active")


def active_families() -> tuple[str, ...]:
    """Roster identity, recorded beside any measurement taken under it."""
    return tuple(
        family.name for family in _FAMILY_REGISTRY if family.active
    )


def generate_logic_task(index: int) -> GeneratedLogicTask:
    rng = _rng_for_index(index)
    generator = _GENERATORS[rng.randbelow(len(_GENERATORS))]
    return generator(rng)


def verifier_spec(task: GeneratedLogicTask) -> str:
    """Minimum the checker needs. Constraint checks carry no answer at all.

    ``canonical_json`` bounds the document the same way it bounds a model
    answer, so shipping a reference solution beside the clues would blow the
    item budget — and a constraint checker never reads it.
    """
    spec: dict[str, Any] = {
        "schema": "reliquary/logic-verifier/v1",
        "checker_version": CHECKER_VERSION,
        "operation_id": task.operation_id,
        "family": task.family,
        "difficulty": task.difficulty,
        "check": task.check,
        "constraints": task.constraints,
    }
    if task.check == "equality":
        spec["expected"] = task.expected
    return canonical_json(spec)


def _check_numbrix(answer: Any, constraints: dict[str, Any]) -> bool:
    size = int(constraints["size"])
    clues = constraints["clues"]
    if not isinstance(answer, list) or len(answer) != size:
        return False
    position: dict[int, tuple[int, int]] = {}
    for row_index, row in enumerate(answer):
        if not isinstance(row, list) or len(row) != size:
            return False
        for column_index, value in enumerate(row):
            if not isinstance(value, int) or isinstance(value, bool):
                return False
            if not 1 <= value <= size * size or value in position:
                return False
            given = clues[row_index][column_index]
            if given and given != value:
                return False
            position[value] = (row_index, column_index)
    for value in range(1, size * size):
        row, column = position[value]
        next_row, next_column = position[value + 1]
        if abs(row - next_row) + abs(column - next_column) != 1:
            return False
    return True


def _check_cryptarithm(answer: Any, constraints: dict[str, Any]) -> bool:
    """Accept any assignment that makes the sum hold, not one stored answer."""
    words = list(constraints["addends"]) + [constraints["sum"]]
    if not isinstance(answer, dict):
        return False
    letters = sorted({character for word in words for character in word})
    if sorted(answer) != letters:
        return False
    digits = []
    for letter in letters:
        value = answer[letter]
        if not isinstance(value, int) or isinstance(value, bool):
            return False
        if not 0 <= value <= 9:
            return False
        digits.append(value)
    if len(set(digits)) != len(digits):
        return False
    values = []
    for word in words:
        if answer[word[0]] == 0:
            return False
        values.append(int("".join(str(answer[c]) for c in word)))
    return values[0] + values[1] == values[2]


def check_answer(spec: dict[str, Any], answer: Any) -> bool:
    """Total checker: never raises, returns False on anything unexpected."""
    check = spec.get("check")
    if check == "equality":
        expected = spec.get("expected")
        # Reject the bool/int conflation JSON allows through.
        if isinstance(expected, bool) != isinstance(answer, bool):
            return False
        return answer == expected
    constraints = spec.get("constraints")
    if not isinstance(constraints, dict):
        return False
    if check == "numbrix_path":
        return _check_numbrix(answer, constraints)
    if check == "cryptarithm_sum":
        return _check_cryptarithm(answer, constraints)
    return False


__all__ = [
    "CHECKER_VERSION",
    "ENVIRONMENT_NAME",
    "GENERATOR_VERSION",
    "GeneratedLogicTask",
    "TASK_FAMILY",
    "VIRTUAL_LENGTH",
    "active_families",
    "check_answer",
    "generate_logic_task",
    "verifier_spec",
]
