"""Wire bounds for TOPLOC proofs, shared by every submission that carries them.

One honest 128-point proof is 344 base64 characters; at the finest deployed
chunking (32 tokens) that is under 11 characters per token. Proofs are bounded
by the length of what they prove, plus one chunk of rounding; a contract with
finer chunks or a larger topk must raise these.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated

from pydantic import StringConstraints

MAX_PROOF_BYTES = 2 + 2 * 1024
MAX_PROOF_B64_CHARS = 4 * ((MAX_PROOF_BYTES + 2) // 3)
MAX_PROOF_CHARS_PER_TOKEN = 11
PROOF_CHARS_SLACK = 344
ProofB64 = Annotated[
    str,
    StringConstraints(pattern=r"^[A-Za-z0-9+/]*={0,2}$", max_length=MAX_PROOF_B64_CHARS),
]


def proof_volume_error(proofs: Sequence[str], token_count: int) -> str | None:
    if len(proofs) > token_count:
        return "more proofs than tokens"
    budget = token_count * MAX_PROOF_CHARS_PER_TOKEN + PROOF_CHARS_SLACK
    if sum(len(proof) for proof in proofs) > budget:
        return f"proofs exceed {budget} characters"
    return None
