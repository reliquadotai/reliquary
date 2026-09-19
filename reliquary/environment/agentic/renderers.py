"""Which renderer an episode is written in, chosen by its declared id.

The id is not decoration. A miner renders an episode into the tokens it
generates and proves; a validator re-renders the same episode and requires the
two to match byte for byte. If the two sides picked renderers independently —
or if one of them always picked the same one, as both used to — an environment
declaring any other dialect would fail every replay on a transcript mismatch.
So there is one lookup, and both sides go through it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from reliquary.environment.agentic.chatml import (
    CHATML_RENDERER_ID,
    ChatMLEpisodeRenderer,
)
from reliquary.environment.agentic.renderer import (
    EPISODE_RENDERER_ID,
    CanonicalEpisodeRenderer,
)

_RENDERERS: dict[str, type] = {
    EPISODE_RENDERER_ID: CanonicalEpisodeRenderer,
    CHATML_RENDERER_ID: ChatMLEpisodeRenderer,
}

RENDERER_IDS = frozenset(_RENDERERS)


def renderer_for(renderer_id: str, encode: Callable[[str], list[int]]) -> Any:
    """The renderer an environment declared, bound to a tokenizer's encoder.

    An unknown id is refused rather than defaulted: falling back to the JSONL
    renderer would render a ChatML-trained policy's episodes in a dialect it has
    never seen, and nothing would say so until the band came back empty.
    """

    try:
        renderer_type = _RENDERERS[renderer_id]
    except KeyError:
        raise ValueError(
            f"unknown episode renderer {renderer_id!r}; "
            f"expected one of {sorted(_RENDERERS)}"
        ) from None
    return renderer_type(encode)


__all__ = ["RENDERER_IDS", "renderer_for"]
