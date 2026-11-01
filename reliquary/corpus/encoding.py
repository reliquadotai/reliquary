"""Encodings a miner and a validator must compute identically.

Kept in one module on purpose: the auditor prefills the prompt tokens this
module produces, and a miner that tokenized differently would fail every audit
while being honest.
"""

from __future__ import annotations

from collections.abc import Sequence
import hashlib
from pathlib import Path


# A prompt given as token ids (a prompt set row's ``prompt_token_ids``) travels as this canonical text in place of a
# rendered prompt: miner and validator build the same string, compare it as any rendered prompt, and read the ids
# back exactly (``prompt_token_ids``), never through the tokenizer. No real chat template starts with a NUL.
TOKEN_IDS_PREFIX = "\x00reliquary-token-ids:"
MAX_PROMPT_TOKEN_IDS = 131072


def token_ids_text(ids: Sequence[int]) -> str:
    if not ids or len(ids) > MAX_PROMPT_TOKEN_IDS:
        raise ValueError(f"a token-id prompt holds 1 to {MAX_PROMPT_TOKEN_IDS} ids")
    if any(isinstance(i, bool) or not isinstance(i, int) or i < 0 for i in ids):
        raise ValueError("a token-id prompt holds non-negative whole numbers")
    return TOKEN_IDS_PREFIX + ",".join(str(int(i)) for i in ids)


def is_token_ids_text(text: str) -> bool:
    return isinstance(text, str) and text.startswith(TOKEN_IDS_PREFIX)


def prompt_token_ids(tokenizer, rendered_prompt: str) -> list[int]:
    if is_token_ids_text(rendered_prompt):
        body = rendered_prompt[len(TOKEN_IDS_PREFIX):]
        ids = [int(part) for part in body.split(",")]
        if token_ids_text(ids) != rendered_prompt:
            raise ValueError("a token-id prompt is not in its canonical form")
        return ids
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
