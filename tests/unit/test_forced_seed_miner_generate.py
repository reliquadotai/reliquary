"""End-to-end coverage of the forced-seed wiring inside the real
``MiningEngine._generate_m_rollouts`` phase-1 path: with a fake generation
model that applies the passed ``logits_processor`` greedily (exactly as HF
does), the produced completion tokens must equal the protocol forced picks
for each rollout index and position. Proves the processor is constructed with
the right rollout_indices / base_offsets / start_len and that HF's own warpers
are stripped.
"""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from reliquary.constants import M_ROLLOUTS, T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment import forced_sampling as fs
from reliquary.miner.engine import MiningEngine

_VOCAB = 5
_PROMPT = [1, 2, 3]
_RAW_ROW = torch.tensor([0.2, 0.1, 0.0, 0.15, 0.05])  # flat -> stochastic
_RANDOMNESS = "rr"
_HOTKEY = "hk"
_PROMPT_IDX = 7
_CKPT = "sha:ck"
_N_NEW = 3


class _FakeGenModel:
    """Minimal HF-generate stand-in: greedy decode that applies the
    logits_processor each step, like transformers' sample loop with
    do_sample=False."""

    device = "cpu"

    def generate(self, input_ids, *, max_new_tokens, logits_processor=None,
                 **kwargs):
        seq = input_ids
        for _ in range(max_new_tokens):
            b = seq.shape[0]
            scores = _RAW_ROW.unsqueeze(0).repeat(b, 1)
            if logits_processor is not None:
                scores = logits_processor(seq, scores)
            nxt = scores.argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq


def _expected_completion(rollout_index):
    return [
        fs.pick(
            fs.warp(_RAW_ROW, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO),
            fs.u_at(_RANDOMNESS, _HOTKEY, _PROMPT_IDX, _CKPT, rollout_index, t),
        )
        for t in range(_N_NEW)
    ]


@patch("reliquary.shared.hf_compat.resolve_hidden_size", return_value=128)
@patch("reliquary.shared.modeling.resolve_eos_token_ids", return_value={999})
@patch("reliquary.protocol.tokens.encode_prompt", return_value=list(_PROMPT))
def test_generate_m_rollouts_phase1_follows_forced_stream(_enc, _eos, _hid):
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address=_HOTKEY))
    engine = MiningEngine(
        vllm_model=_FakeGenModel(), hf_model=MagicMock(), tokenizer=MagicMock(),
        wallet=wallet, envs={"opencode": SimpleNamespace(name="opencode")},
        mix=[("opencode", 1)], max_new_tokens=_N_NEW,
    )

    rollouts = engine._generate_m_rollouts(
        {"prompt": "p"}, _RANDOMNESS, env_name="opencode",
        prompt_idx=_PROMPT_IDX, checkpoint_hash=_CKPT,
    )

    assert len(rollouts) == M_ROLLOUTS
    for r, rollout in enumerate(rollouts):
        completion = rollout["tokens"][rollout["prompt_length"]:]
        assert completion == _expected_completion(r), f"rollout {r} not forced"
