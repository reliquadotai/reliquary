import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from reliquary.constants import M_ROLLOUTS, T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, warp
from reliquary.miner.engine import MiningEngine
from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor
from reliquary.miner.vllm_generation import ForcedSeedVLLMProcessor, forced_seed_extra_args
from reliquary.protocol.seed_pool import SeedPool
from reliquary.protocol.service_contract import ServiceContract


def _contract(exploration=False):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["policies"]["sampling"] = {"kind": "public-group-pool/v1", "group_size": M_ROLLOUTS,
                                     "pool_groups": 3, "renewal_windows": 2}
    if exploration:
        value["service_kind"] = "adaptive_training"
        value["policies"]["checkpoint"] = {"kind": "trainer-driven/v1", "task_scoped": 1}
        value["policies"]["reward"] = {"kind": "exploration-discount/v1", "divisor": 4,
                                       "budget_bps": 1000, "refresh_windows": 2, "max_tokens_per_group": 1000000}
    return ServiceContract.from_dict(value)


def _pool():
    return SeedPool.from_contract(_contract(), prompt_idx=7, checkpoint_hash="d" * 40,
                                  pool_epoch=3, randomness="ab" * 32)


def test_hf_and_vllm_draw_the_same_public_candidate_after_resumption():
    pool = _pool()
    logits = torch.tensor([[0.2, 0.1, 0.0, 0.15]] * 2)
    hf = ForcedSeedLogitsProcessor(randomness="current-window", hotkey="miner-A", prompt_idx=7,
                                  checkpoint_hash="d" * 40, rollout_indices=[3, 5], base_offsets=[9, 10],
                                  start_len=4, seed_pool=pool, candidate_id=2)
    other = ForcedSeedLogitsProcessor(randomness="another-window", hotkey="miner-B", prompt_idx=7,
                                     checkpoint_hash="d" * 40, rollout_indices=[3, 5], base_offsets=[9, 10],
                                     start_len=4, seed_pool=pool, candidate_id=2)
    vllm = ForcedSeedVLLMProcessor()
    requests = [SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness="current-window", prompt_idx=7, checkpoint_hash="d" * 40,
        rollout_index=index, base_offset=offset, seed_pool=pool, candidate_id=2))
        for index, offset in ((3, 9), (5, 10))]
    vllm.update_state(SimpleNamespace(removed=[], moved=[], added=[
        (row, params, [], []) for row, params in enumerate(requests)]))
    for step in range(3):
        tokens = torch.zeros(2, 4 + step, dtype=torch.long)
        expected = torch.tensor([pick(warp(logits[row], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO),
                                      pool.uniform(2, index, offset + step))
                                 for row, (index, offset) in enumerate(((3, 9), (5, 10)))])
        assert torch.equal(hf(tokens, logits.clone()).argmax(-1), expected)
        assert torch.equal(other(tokens, logits.clone()).argmax(-1), expected)
        assert torch.equal(vllm.apply(logits.clone()).argmax(-1), expected)


def test_mismatched_public_generation_context_is_refused():
    with pytest.raises(ValueError, match="context differs"):
        ForcedSeedLogitsProcessor(randomness="r", hotkey="h", prompt_idx=8,
                                  checkpoint_hash="d" * 40, rollout_indices=[0], base_offsets=[0],
                                  start_len=1, seed_pool=_pool(), candidate_id=0)
    processor = ForcedSeedVLLMProcessor()
    args = forced_seed_extra_args(randomness="r", prompt_idx=8, checkpoint_hash="d" * 40,
                                  rollout_index=0, seed_pool=_pool(), candidate_id=0)
    with pytest.raises(ValueError, match="context differs"):
        processor.update_state(SimpleNamespace(removed=[], moved=[], added=[
            (0, SimpleNamespace(extra_args=args), [], [])]))


def _engine():
    engine = object.__new__(MiningEngine)
    engine.max_new_tokens = 2
    engine.tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens[0]))
    return engine


def test_cherry_pick_keeps_complete_group_and_original_candidate_ids():
    engine, pool = _engine(), _pool()
    env = SimpleNamespace(name="opencodeinstruct", compute_reward=lambda problem, text: float(text))
    candidates = [[0.0] * M_ROLLOUTS, [i % 2 for i in range(M_ROLLOUTS)], [1.0] * M_ROLLOUTS]
    seen = []
    def generate(problem, randomness, **kwargs):
        candidate = kwargs["candidate_id"]
        seen.append(candidate)
        return [{"tokens": [99, reward], "prompt_length": 1,
                 "seed_pool": pool.selection(candidate).rollout_binding(index)}
                for index, reward in enumerate(candidates[candidate])]
    engine._generate_m_rollouts = generate
    result = engine._generate_public_pool_rollouts({}, "current", env=env, prompt_idx=7,
                                                   checkpoint_hash="d" * 40, seed_pool=pool,
                                                   max_exploration_tokens=1000000)
    assert seen == [0, 1, 2]
    assert len(result) == M_ROLLOUTS
    assert [g["seed_pool"]["candidate_id"] for g in result] == [1] * M_ROLLOUTS
    assert [g["seed_pool"]["rollout_index"] for g in result] == list(range(M_ROLLOUTS))
    with pytest.raises(ValueError, match="cannot cover"):
        engine._generate_public_pool_rollouts({}, "current", env=env, prompt_idx=7,
                                              checkpoint_hash="d" * 40, seed_pool=pool,
                                              max_exploration_tokens=1)


@pytest.mark.parametrize("rewards,purpose", [([0.0] * M_ROLLOUTS, "exploration"),
                                            ([1.0] * M_ROLLOUTS, "exploration"),
                                            ([0.3] * M_ROLLOUTS, "exploration"),
                                            ([0.0, 0.1] * (M_ROLLOUTS // 2), "training"),
                                            ([0.0, 1.0] * (M_ROLLOUTS // 2), "training")])
def test_service_purpose_distinguishes_uniform_from_diverse_below_threshold(rewards, purpose):
    engine = _engine()
    env = SimpleNamespace(compute_reward=lambda problem, text: float(text))
    generations = [{"tokens": [99, r], "prompt_length": 1} for r in rewards]
    result = engine._service_submission_binding({"contract": _contract(exploration=True).to_dict()},
                                                 generations, {}, env)
    assert result["purpose"] == purpose
    legacy = engine._service_submission_binding({"contract": _contract().to_dict()}, generations, {}, env)
    assert legacy["purpose"] == "training"


def test_unknown_reward_is_never_marked_exploration():
    engine = _engine()
    env = SimpleNamespace(compute_reward=lambda problem, text: None)
    result = engine._service_submission_binding({"contract": _contract(exploration=True).to_dict()},
                                                 [{"tokens": [99, 0], "prompt_length": 1}] * M_ROLLOUTS, {}, env)
    assert result["purpose"] == "training"


def test_reference_hf_generator_preserves_public_stream_and_candidate_identity():
    pool = _pool()
    raw = torch.tensor([0.2, 0.1, 0.0, 0.15])
    class Model:
        device = "cpu"
        def generate(self, input_ids, *, max_new_tokens, logits_processor, **kwargs):
            assert kwargs["do_sample"] is False
            tokens = input_ids
            for _ in range(max_new_tokens):
                scores = raw.repeat(tokens.shape[0], 1)
                choice = logits_processor(tokens, scores).argmax(-1, keepdim=True)
                tokens = torch.cat((tokens, choice), dim=1)
            return tokens
    engine = _engine()
    engine.vllm_model = Model()
    engine.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner"))
    engine.tokenizer = SimpleNamespace(pad_token_id=0)
    with patch("reliquary.protocol.tokens.encode_prompt", return_value=[1, 2]), \
            patch("reliquary.shared.modeling.resolve_eos_token_ids", return_value={99}):
        generations = engine._generate_m_rollouts({"prompt": "p"}, "current", env_name="opencodeinstruct",
                                                  prompt_idx=7, checkpoint_hash="d" * 40,
                                                  seed_pool=pool, candidate_id=2)
    probs = warp(raw, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    assert len(generations) == M_ROLLOUTS
    for index, generation in enumerate(generations):
        assert generation["tokens"][2:] == [pick(probs, pool.uniform(2, index, t)) for t in range(2)]
        assert generation["seed_pool"] == pool.selection(2).rollout_binding(index)


def test_public_bft_phase_two_keeps_original_rows_and_offsets_after_padding():
    from reliquary.miner.engine import _bft_assemble_rollouts

    pool = _pool()
    raw = torch.tensor([0.2, 0.1, 0.0, 0.15])
    class Model:
        device = "cpu"
        def generate(self, tokens, *, logits_processor, **kwargs):
            picks = logits_processor(tokens, raw.repeat(tokens.shape[0], 1)).argmax(-1, keepdim=True)
            return torch.cat((tokens, picks, torch.full_like(picks, 99)), dim=1)
    phase_one = torch.tensor([[1, 1, 5, 777, 6, 99],
                              [1, 1, 5, 777, 7, 8],
                              [1, 1, 5, 6, 7, 8]])
    generations = _bft_assemble_rollouts(model=Model(), phase1_tensor=phase_one,
        prompt_tokens=[1, 1], think_close_ids={777}, force_ids=[777, 7, 8], eos_ids={99},
        answer_budget=2, randomness="current", hotkey="miner", prompt_idx=7, checkpoint_hash="d" * 40,
        seed_pool=pool, candidate_id=2)
    probs = warp(raw, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    assert generations[1]["tokens"][-2] == pick(probs, pool.uniform(2, 1, 4))
    assert generations[2]["tokens"][-2] == pick(probs, pool.uniform(2, 2, 7))
    assert generations[2]["force_span"] == (6, 9)
