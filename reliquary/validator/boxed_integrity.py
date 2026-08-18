r"""Structural validity of a rollout's final boxed/fboxed answer.

The OMI reward scores the LAST ``\boxed{...}`` / ``\fbox{...}`` in the
completion. A reward=0 rollout whose final box is malformed (empty,
special-token, or unclosed) did not produce a parseable answer — it is a "fake
negative" used to manufacture a group reward vector (k=4 / sigma=0.5) that
passes the zone filter. Examples: appending ``\boxed{<|im_end|>`` after a
correct ``\boxed{121}``, an empty ``\boxed{}``, or spamming boxes to the token
cap so the final one is cut off.

This check is purely structural and aligned with what the env scores (the last
box): it does NOT compare to the ground truth and does NOT judge intent. A
well-formed final box that is simply wrong is a legitimate negative and is not
flagged here (forced wrong answers are covered by the boxed-answer probability
check). Pure, side-effect-free; called by the batcher before GRAIL.
"""
from __future__ import annotations

from dataclasses import dataclass

# Stop/special tokens that must never appear inside a final answer box.
SPECIAL_TOKENS = ("<|im_end|>", "<|endoftext|>", "<|im_start|>")

_MARKERS = (r"\boxed{", r"\fbox{")


@dataclass(frozen=True)
class BoxedSpan:
    marker: str
    content: str
    well_formed: bool


def extract_boxed_spans(text: str) -> list[BoxedSpan]:
    r"""All ``\boxed{...}`` / ``\fbox{...}`` occurrences with a flag.

    Malformed when unclosed, empty/whitespace, or containing a special token.
    """
    spans: list[BoxedSpan] = []
    i = 0
    while True:
        matches = [
            (pos, marker)
            for marker in _MARKERS
            for pos in [text.find(marker, i)]
            if pos != -1
        ]
        if not matches:
            break
        j, marker = min(matches, key=lambda item: item[0])
        k = j + len(marker)
        depth = 1
        buf: list[str] = []
        closed = False
        while k < len(text):
            c = text[k]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    closed = True
                    k += 1
                    break
            buf.append(c)
            k += 1
        content = "".join(buf)
        well_formed = (
            closed
            and content.strip() != ""
            and not any(tok in content for tok in SPECIAL_TOKENS)
        )
        spans.append(
            BoxedSpan(marker=marker, content=content, well_formed=well_formed)
        )
        i = k if k > j + len(marker) else j + len(marker)
    return spans


def is_missing_final_answer_box(text: str) -> bool:
    r"""True when the completion contains no ``\boxed``/``\fbox`` marker.

    Under the v4 boxed-only reward contract this is an off-format answer. It
    still receives reward zero, but its outcome must be valued as uncertain by
    the auction: otherwise deleting a naturally sampled box could manufacture
    a useful zero and move a degenerate group into the sigma zone.

    A present-but-malformed final marker is deliberately not classified as
    missing; ``has_malformed_final_answer`` rejects that stronger condition.
    """

    return not extract_boxed_spans(text)


def has_malformed_final_answer(
    reward: float,
    text: str,
    completion_length: int | None = None,
    cap: int | None = None,
) -> tuple[bool, str | None]:
    r"""True when a reward=0 rollout's final ``\boxed{}`` is malformed.

    Aligned with the env (which scores the last box). Only evaluated for
    reward < 0.5. Returns ``(False, None)`` when there is no box at all (a clean
    give-up), or when the last box is well-formed (a legitimate wrong answer).

    ``completion_length`` and ``cap`` are accepted for call-site symmetry with
    termination checks, but they do not exempt cap hits: a cap-cut malformed
    final answer is still not a trustworthy negative for GRPO.
    """
    if reward is not None and reward >= 0.5:
        return False, None
    spans = extract_boxed_spans(text)
    if not spans:
        return False, None
    if not spans[-1].well_formed:
        return True, "malformed_final_boxed"
    return False, None
