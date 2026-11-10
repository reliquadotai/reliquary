"""Smoke test: submission → batch → cooldown → weights over 3 windows.

No on-chain dependencies — all verifiers are stubbed. The goal is to
prove the pieces fit together end-to-end.
"""

import hashlib

import pytest

from reliquary.constants import B_BATCH, CHALLENGE_K, M_ROLLOUTS
from reliquary.protocol.submission import RejectReason
from collections import defaultdict


class FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx): return {"prompt": f"q{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c): return 1.0 if "WIN" in c else 0.0


class _ModelStub:
    """Minimal model stub so verify_tokens can resolve vocab/length limits."""
    class config:
        vocab_size = 10_000
        max_position_embeddings = 4096


def _make_commit(
    *,
    tokens: list[int] | None = None,
    prompt_length: int = 4,
    success: bool = False,
    total_reward: float = 0.0,
) -> dict:
    """Build a minimal commit that passes CommitModel.model_validate."""
    if tokens is None:
        tokens = list(range(CHALLENGE_K + prompt_length))
    seq_len = len(tokens)
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v7",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "success": success,
            "total_reward": total_reward,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


def _merkle(n: int) -> str:
    return hashlib.sha256(str(n).encode()).hexdigest()


def _run_ema_windows(hotkey_counts_per_window: list[dict[str, int]]) -> dict:
    """Replay EMA over a synthetic archive sequence; return final EMA dict.

    Builds R2-style archive records out of the per-window hotkey counts and
    feeds them through ``WeightOnlyValidator._replay_ema`` — the single
    canonical scoring path used by both trainer and weight-only modes.
    """
    from reliquary.validator.weight_only import WeightOnlyValidator

    archives = []
    for window_idx, counts in enumerate(hotkey_counts_per_window):
        batch_entries = []
        rewards_by_hotkey = {}
        for hk, n in counts.items():
            for _ in range(n):
                batch_entries.append({"hotkey": hk, "prompt_idx": 0})
            rewards_by_hotkey[hk] = n / B_BATCH
        archives.append({
            "window_start": window_idx,
            "batch": batch_entries,
            "rewards_by_hotkey": rewards_by_hotkey,
        })
    return WeightOnlyValidator._replay_ema(archives)


def test_weights_for_full_batch():
    """Full batch every window — after convergence, EMA sum → 1.0, burn ≈ 0."""
    counts = {f"hk{i}": 1 for i in range(B_BATCH)}
    ema = _run_ema_windows([counts] * 500)
    total = sum(ema.values())
    burn = max(0.0, 1.0 - total)
    assert abs(total + burn - 1.0) < 1e-9
    assert abs(burn) < 0.005
    assert len(ema) == B_BATCH


def test_weights_for_partial_batch_burns_rest():
    """Partial batch (filled/B_BATCH) every window — burn → (B_BATCH - filled) / B_BATCH."""
    filled = 5
    counts = {f"hk{i}": 1 for i in range(filled)}
    ema = _run_ema_windows([counts] * 500)
    total = sum(ema.values())
    burn = max(0.0, 1.0 - total)
    assert abs(burn - (B_BATCH - filled) / B_BATCH) < 0.005


