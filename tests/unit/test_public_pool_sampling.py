from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from reliquary.constants import M_ROLLOUTS, T_PROTO, TOP_K_PROTO, TOP_P_PROTO
from reliquary.environment.forced_sampling import pick, warp
from reliquary.miner.engine import MiningEngine, bind_public_seed_group
from reliquary.miner.forced_seed_sampler import ForcedSeedLogitsProcessor
from reliquary.miner.vllm_generation import FORCED_SEED_KEY, ForcedSeedVLLMProcessor, forced_seed_extra_args
from reliquary.protocol.seed_pool import SeedPool
from tests.unit.service_v2_fixtures import MATH, contract_v2


POOL_SEEDS = 2 * M_ROLLOUTS
FIRST = tuple(range(M_ROLLOUTS))
ODD = tuple(range(1, POOL_SEEDS, 2))
LAST = tuple(range(M_ROLLOUTS, POOL_SEEDS))


def _contract(exploration=False):
    return contract_v2(exploration=1 if exploration else 0)


def _pool():
    return SeedPool.from_contract(_contract(), environment=MATH, prompt_idx=7, checkpoint_hash="d" * 40,
                                  pool_epoch=3, randomness="ab" * 32)


def test_hf_and_vllm_draw_the_seed_stream_whatever_the_rank_hotkey_or_window():
    pool = _pool()
    logits = torch.tensor([[0.2, 0.1, 0.0, 0.15]] * 2)
    rows = ((3, 9), (5, 10))                     # (rollout rank, completion offset)
    hf = ForcedSeedLogitsProcessor(randomness="current-window", hotkey="miner-A", prompt_idx=7,
                                  checkpoint_hash="d" * 40, rollout_indices=[3, 5], base_offsets=[9, 10],
                                  start_len=4, seed_pool=pool, seeds=ODD)
    # Another miner, another window, another subset in which the same two seeds sit at other ranks.
    shifted = (0, *ODD[:-1]) if M_ROLLOUTS > 6 else None
    assert shifted is not None and shifted[4] == ODD[3] and shifted[6] == ODD[5]
    other = ForcedSeedLogitsProcessor(randomness="another-window", hotkey="miner-B", prompt_idx=7,
                                     checkpoint_hash="d" * 40, rollout_indices=[4, 6], base_offsets=[9, 10],
                                     start_len=4, seed_pool=pool, seeds=shifted)
    vllm = ForcedSeedVLLMProcessor()
    requests = [SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness="current-window", prompt_idx=7, checkpoint_hash="d" * 40,
        rollout_index=index, base_offset=offset, seed_pool=pool, seed_index=ODD[index]))
        for index, offset in rows]
    vllm.update_state(SimpleNamespace(removed=[], moved=[], added=[
        (row, params, [], []) for row, params in enumerate(requests)]))
    for step in range(3):
        tokens = torch.zeros(2, 4 + step, dtype=torch.long)
        expected = torch.tensor([pick(warp(logits[row], t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO),
                                      pool.uniform(ODD[index], offset + step))
                                 for row, (index, offset) in enumerate(rows)])
        assert torch.equal(hf(tokens, logits.clone()).argmax(-1), expected)
        assert torch.equal(other(tokens, logits.clone()).argmax(-1), expected)
        assert torch.equal(vllm.apply(logits.clone()).argmax(-1), expected)


def test_mismatched_public_generation_context_is_refused():
    with pytest.raises(ValueError, match="context differs"):
        ForcedSeedLogitsProcessor(randomness="r", hotkey="h", prompt_idx=8,
                                  checkpoint_hash="d" * 40, rollout_indices=[0], base_offsets=[0],
                                  start_len=1, seed_pool=_pool(), seeds=FIRST)
    processor = ForcedSeedVLLMProcessor()
    args = forced_seed_extra_args(randomness="r", prompt_idx=8, checkpoint_hash="d" * 40,
                                  rollout_index=0, seed_pool=_pool(), seed_index=0)
    with pytest.raises(ValueError, match="context differs"):
        processor.update_state(SimpleNamespace(removed=[], moved=[], added=[
            (0, SimpleNamespace(extra_args=args), [], [])]))


def test_samplers_refuse_seeds_outside_the_pool_and_seeds_without_a_pool():
    kw = dict(randomness="r", hotkey="h", prompt_idx=7, checkpoint_hash="d" * 40, base_offsets=[0], start_len=1)
    for seeds, indices in (((POOL_SEEDS,), [0]), ((True,), [0]), ((0,), [1]), (None, [0])):
        with pytest.raises(ValueError):
            ForcedSeedLogitsProcessor(rollout_indices=indices, seed_pool=_pool(), seeds=seeds, **kw)
    with pytest.raises(ValueError):
        ForcedSeedLogitsProcessor(rollout_indices=[0], seeds=FIRST, **kw)
    for seed in (POOL_SEEDS, -1, True, None):
        with pytest.raises(ValueError):
            forced_seed_extra_args(randomness="r", prompt_idx=7, checkpoint_hash="d" * 40,
                                   rollout_index=0, seed_pool=_pool(), seed_index=seed)
    with pytest.raises(ValueError):
        forced_seed_extra_args(randomness="r", prompt_idx=7, checkpoint_hash="d" * 40,
                               rollout_index=0, seed_index=0)
    # the legacy (no pool) request is unchanged
    assert forced_seed_extra_args(randomness="r", prompt_idx=7, checkpoint_hash="d" * 40, rollout_index=2) == {
        FORCED_SEED_KEY: {"randomness": "r", "prompt_idx": 7, "checkpoint_hash": "d" * 40,
                                  "rollout_index": 2, "base_offset": 0}}


def _engine():
    engine = object.__new__(MiningEngine)
    engine.max_new_tokens = 2
    engine.tokenizer = SimpleNamespace(decode=lambda tokens: str(tokens[0]))
    return engine


def _fake_generation(engine, pool, calls):
    """A generator whose completion for a seed is that seed, whatever its companions."""
    def generate(problem, randomness, **kwargs):
        assert kwargs["seed_pool"] is pool
        seeds = kwargs["seeds"]
        calls.append(seeds)
        selection = pool.selection(seeds)
        return [{"tokens": [99, seed], "prompt_length": 1, "seed_pool": selection.rollout_binding(index)}
                for index, seed in enumerate(seeds)]
    engine._generate_m_rollouts = generate


def _public_group(engine, pool, env):
    return engine._generate_public_pool_rollouts({}, "current", env=env, prompt_idx=7,
                                                 checkpoint_hash="d" * 40, seed_pool=pool)


def test_default_seed_policy_is_the_first_m_seeds_without_over_generation():
    engine, pool, calls = _engine(), _pool(), []
    _fake_generation(engine, pool, calls)
    env = SimpleNamespace(name=MATH, compute_reward=lambda problem, text: pytest.fail("default never grades"))
    result = _public_group(engine, pool, env)
    assert calls == [FIRST]
    assert [g["seed_pool"] for g in result] == [pool.selection(FIRST).rollout_binding(i) for i in range(M_ROLLOUTS)]


def test_a_replaced_hook_over_generates_the_pool_and_keeps_any_subset():
    engine, pool, calls = _engine(), _pool(), []
    _fake_generation(engine, pool, calls)
    env = SimpleNamespace(name=MATH)
    keep = (0, *range(POOL_SEEDS - M_ROLLOUTS + 1, POOL_SEEDS))     # one seed of the low half, the rest high

    def hook(seed_pool, generate, *, problem, env):
        everything = generate(FIRST) + generate(LAST)
        chosen = [g for g in everything if g["seed_pool"]["seed_index"] in keep]
        return chosen[::-1]                                         # order returned does not matter
    engine.choose_public_seed_group = hook
    result = _public_group(engine, pool, env)
    assert calls == [FIRST, LAST]
    selection = pool.selection(keep)
    assert [g["seed_pool"] for g in result] == [selection.rollout_binding(i) for i in range(M_ROLLOUTS)]
    assert [g["tokens"][1] for g in result] == list(keep)           # each rollout is its own seed's completion
    from reliquary.protocol.seed_pool import validate_rollout_selection
    validate_rollout_selection(pool, selection, [{"rollout": {"seed_pool": g["seed_pool"]}} for g in result])


@pytest.mark.parametrize("broken", ["duplicate", "short", "long", "unbound", "foreign"])
def test_a_hook_cannot_return_an_invalid_group(broken):
    engine, pool, calls = _engine(), _pool(), []
    _fake_generation(engine, pool, calls)

    def hook(seed_pool, generate, *, problem, env):
        group = generate(FIRST)
        if broken == "duplicate":
            return group[:-1] + group[:1]
        if broken == "short":
            return group[:-1]
        if broken == "long":
            return group + generate(LAST)[:1]
        if broken == "unbound":
            return [{k: v for k, v in g.items() if k != "seed_pool"} for g in group]
        other = SeedPool.from_dict({**pool.to_dict(), "prompt_idx": 8})
        return [dict(g, seed_pool=other.selection(FIRST).rollout_binding(i)) for i, g in enumerate(group)]
    engine.choose_public_seed_group = hook
    with pytest.raises(ValueError):
        _public_group(engine, pool, SimpleNamespace(name=MATH))


def test_request_selection_is_rebuilt_from_the_rollouts_seeds():
    engine, pool = _engine(), _pool()
    engine.wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="miner"))
    engine._build_rollout_submission = lambda generation, problem, randomness, env: SimpleNamespace(
        seed=generation.get("seed_pool"))
    selection = pool.selection(ODD)
    group = [{"tokens": [1], "seed_pool": selection.rollout_binding(i)} for i in range(M_ROLLOUTS)]
    captured = {}

    class Request:
        def __init__(self, **kwargs):
            captured.update(kwargs)
    with patch("reliquary.protocol.submission.BatchSubmissionRequest", Request), \
            patch("reliquary.miner.engine._compute_merkle_root", return_value="00"), \
            patch("reliquary.miner.engine._initial_runtime_bound_nonce", return_value="n"):
        build = lambda generations: engine.build_batch_request_from_generations(
            generations=generations, problem={}, randomness="r", prompt_idx=7, window_number=1,
            checkpoint_revision="d" * 40, runtime_fingerprint=None, environment=SimpleNamespace(name=MATH))
        build(group)
        assert captured["pool_selection"] == selection.to_dict()
        for bad in (group[::-1], [group[1], group[0], *group[2:]], group[:1] * M_ROLLOUTS,
                    [dict(group[0], seed_pool={**group[0]["seed_pool"], "pool_sha256": "ee" * 32}), *group[1:]],
                    [{"tokens": [1]}, *group[1:]]):
            with pytest.raises(ValueError):
                build(bad)
        # unsorted seeds at the right ranks: not a canonical subset
        forged = [dict(g, seed_pool={**g["seed_pool"], "seed_index": ODD[M_ROLLOUTS - 1 - i]}) for i, g in enumerate(group)]
        with pytest.raises(ValueError):
            build(forged)


@pytest.mark.parametrize("rewards,purpose", [([0.0] * M_ROLLOUTS, "exploration"),
                                            ([1.0] * M_ROLLOUTS, "exploration"),
                                            ([0.3] * M_ROLLOUTS, "exploration"),
                                            ([0.0, 0.1] * (M_ROLLOUTS // 2), "training"),
                                            ([0.0, 1.0] * (M_ROLLOUTS // 2), "training")])
def test_service_purpose_distinguishes_uniform_from_diverse_below_threshold(rewards, purpose):
    engine = _engine()
    env = SimpleNamespace(name=MATH, compute_reward=lambda problem, text: float(text))
    generations = [{"tokens": [99, r], "prompt_length": 1} for r in rewards]
    result = engine._service_submission_binding({"contract": _contract(exploration=True).to_dict()},
                                                 generations, {}, env)
    assert result["purpose"] == purpose
    legacy = engine._service_submission_binding({"contract": _contract().to_dict()}, generations, {}, env)
    assert legacy["purpose"] == "training"


def test_unknown_reward_is_never_marked_exploration():
    engine = _engine()
    env = SimpleNamespace(name=MATH, compute_reward=lambda problem, text: None)
    result = engine._service_submission_binding({"contract": _contract(exploration=True).to_dict()},
                                                 [{"tokens": [99, 0], "prompt_length": 1}] * M_ROLLOUTS, {}, env)
    assert result["purpose"] == "training"


def test_reference_hf_generator_draws_each_rollout_from_its_chosen_seed():
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
                                                  seed_pool=pool, seeds=ODD)
        for invalid in (ODD[::-1], ODD[:-1], (0, 0, *ODD[2:]), None):
            with pytest.raises(ValueError):
                engine._generate_m_rollouts({"prompt": "p"}, "current", env_name="opencodeinstruct",
                                            prompt_idx=7, checkpoint_hash="d" * 40, seed_pool=pool, seeds=invalid)
        with pytest.raises(ValueError):
            engine._generate_m_rollouts({"prompt": "p"}, "current", env_name="opencodeinstruct",
                                        prompt_idx=7, checkpoint_hash="d" * 40, seeds=ODD)
    probs = warp(raw, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    assert len(generations) == M_ROLLOUTS
    for index, generation in enumerate(generations):
        assert generation["tokens"][2:] == [pick(probs, pool.uniform(ODD[index], t)) for t in range(2)]
        assert generation["seed_pool"] == pool.selection(ODD).rollout_binding(index)


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
        seed_pool=pool, seeds=ODD)
    probs = warp(raw, t=T_PROTO, top_k=TOP_K_PROTO, top_p=TOP_P_PROTO)
    assert generations[1]["tokens"][-2] == pick(probs, pool.uniform(ODD[1], 4))
    assert generations[2]["tokens"][-2] == pick(probs, pool.uniform(ODD[2], 7))
    assert generations[2]["force_span"] == (6, 9)
