"""Which part of a completion is the program.

Only the answer is searched: when the completion contains `</think>`, the text
after the last one. In it, the last fenced block tagged `python`, `py` or
`python3` (any case) is the submission; an untagged block counts only when no
tagged block exists, and a block in another language never does, so sample
output or a C++ sketch after the program does not replace it. When the last
tagged block is blank the completion has no code: an earlier block is not
promoted, because the policy's final answer was empty.

Fences are matched line by line: an opener is a line starting with ``` (after
optional spaces), a closer a line that is exactly ``` (spaces allowed). A
tagged opener met inside an open block abandons that block, so a fence left
open earlier (in prose or reasoning) cannot misalign the ones after it, and a
block left open by truncation is not a program: it scores as no code rather
than running half a solution.

The core's sandboxed grader imports this function, as it imports
`outputs_match`: one source of truth for what the submission is.
"""

from __future__ import annotations

_FENCE = "```"
_THINK_CLOSE = "</think>"
_PYTHON_TAGS = frozenset({"python", "python3", "py"})


def _blocks(text: str) -> list[tuple[str, str]]:
    """Closed fenced blocks as (lowercased first word of the tag, body)."""
    blocks: list[tuple[str, str]] = []
    tag: str | None = None  # the open block's tag, None outside a block
    body: list[str] = []
    for line in text.splitlines(keepends=True):
        bare = line.rstrip("\r\n")
        if tag is not None and bare.strip(" ") == _FENCE:
            blocks.append((tag, "".join(body)))
            tag = None
            continue
        stripped = bare.lstrip(" ")
        if stripped.startswith(_FENCE):
            words = stripped[len(_FENCE):].split()
            opener_tag = words[0].lower() if words else ""
            if tag is None or opener_tag:
                tag, body = opener_tag, []
                continue
        if tag is not None:
            body.append(line)
    return blocks


def extract_program(completion: str) -> str | None:
    text = completion or ""
    if _THINK_CLOSE in text:
        text = text.rsplit(_THINK_CLOSE, 1)[1]
    blocks = _blocks(text)
    tagged = [body for tag, body in blocks if tag in _PYTHON_TAGS]
    candidates = tagged or [body for tag, body in blocks if tag == ""]
    if not candidates or not candidates[-1].strip():
        return None
    return candidates[-1]
