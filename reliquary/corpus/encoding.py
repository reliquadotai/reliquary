"""Encodings a miner and a validator must compute identically.

Kept in one module on purpose: the auditor prefills the prompt tokens this
module produces, and a miner that tokenized differently would fail every audit
while being honest.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path


def prompt_token_ids(tokenizer, rendered_prompt: str) -> list[int]:
    encoded = tokenizer.encode(rendered_prompt, add_special_tokens=False)
    return [int(i) for i in getattr(encoded, "ids", encoded)]


def completion_text(tokenizer, tokens: Sequence[int], eos_token_id: int) -> str:
    """The only spelling the route accepts: the trailing terminator dropped, nothing else."""
    body = list(tokens[:-1]) if tokens and tokens[-1] == eos_token_id else list(tokens)
    return tokenizer.decode(body, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def checkpoint_fingerprint(directory: str | Path) -> str:
    shards = sorted(Path(directory).glob("*.safetensors"))
    if not shards:
        raise ValueError(f"no .safetensors shards in {directory}")
    outer = hashlib.sha256()
    for shard in shards:
        inner = hashlib.sha256()
        with shard.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                inner.update(block)
        outer.update(shard.name.encode() + b"\0" + inner.hexdigest().encode() + b"\n")
    return outer.hexdigest()
