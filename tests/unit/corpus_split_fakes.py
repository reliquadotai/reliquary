"""Stand-ins shared by the split-process tests: a toploc proof, a tokenizer, a
GPU score function, and records. Not a test module itself."""

from __future__ import annotations

from reliquary.protocol.profiles import ProofProfile
from reliquary.protocol.toploc import ChunkResult

PROOF = ProofProfile(scheme="toploc-v1", mode="enforce", chunk_tokens=32, topk=128,
                     exp_mismatch_threshold=60, mant_mean_threshold=40.0,
                     mant_median_threshold=40.0, min_allowed_failures=0,
                     ratio_allowed_failures=0.0)
VOCAB = 200_000
# A completion carrying this token id was not produced by the model.
FORGED = 13
HONEST_CHUNK = ChunkResult(0, 0.0125, 0.01)
FORGED_CHUNK = ChunkResult(1000, 99.5, 99.25)


class Tokenizer:
    """Digits in, digits out: a prompt encodes to one id per character."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 1000 + 1 for c in text]

    def decode(self, ids, **kwargs):
        return "".join(str(i) for i in ids)


def score_rows(rows):
    """``score_sequences`` without a model: per row, one chunk per proof,
    failing when the completion carries FORGED."""
    scores = []
    for tokens, prompt_len, proofs in rows:
        chunk = FORGED_CHUNK if FORGED in tokens[prompt_len:] else HONEST_CHUNK
        scores.append(("ok", tuple(chunk for _ in proofs)))
    return scores


def score_sequences(model, sequences, *, chunk_tokens, topk, batch_tokens):
    return score_rows(sequences), 0.0, 0.0


class Model:
    """What the auditor and the route read off a loaded model."""

    class _Embeddings:
        num_embeddings = VOCAB

    def get_input_embeddings(self):
        return self._Embeddings()

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


def record(hotkey, received_at, *, tokens=(5, 6, 7, 8), forged=False, proofs=None):
    body = list(tokens) + ([FORGED] if forged else [])
    return {"hotkey": hotkey, "received_at": received_at, "token_count": len(body),
            "rendered_prompt": "p", "completions": [
                {"tokens": body, "proofs": proofs or ["A" * 8] * max(1, len(body) // 32 + 1)}]}
