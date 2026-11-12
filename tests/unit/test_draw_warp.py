"""The Gumbel draw over a warped distribution (temperature, top-k, top-p, min-p): the pick is
argmax over the warp's support of (logit / T + G), an exact sample of the warped distribution.
The validator judges a token just outside its own support (the cut moved under drift) as an
ordinary miss, and a token far outside it as impossible."""
from __future__ import annotations

import math

import pytest
import torch

from reliquary.environment import forced_sampling as fs
from reliquary.environment.forced_sampling import Warp


def _us(n, seed="77"):
    return [fs.u_at(seed * 16, 5, "sha:warp", 0, t) for t in range(n)]


# ------------------------------------------------------------------ support


def test_the_full_warp_keeps_every_token():
    x = torch.randn(3, 50)
    assert Warp().identity
    assert bool(fs.warp_support(x, Warp()).all())


def test_top_k_keeps_the_k_highest():
    x = torch.tensor([[0.0, 3.0, 1.0, 2.0, -1.0]])
    assert fs.warp_support(x, Warp(top_k=2)).tolist() == [[False, True, False, True, False]]


def test_top_p_keeps_the_crossing_token_over_the_top_k_set():
    probs = torch.tensor([[0.5, 0.3, 0.15, 0.05]])
    x = probs.log()
    # mass before each sorted token: 0, .5, .8, .95
    assert fs.warp_support(x, Warp(top_p=0.79)).tolist() == [[True, True, False, False]]
    assert fs.warp_support(x, Warp(top_p=0.81)).tolist() == [[True, True, True, False]]
    # top-k first, then top-p renormalised over the 2 kept: .625 / .375 -> top_p .6 keeps only the first
    assert fs.warp_support(x, Warp(top_k=2, top_p=0.6)).tolist() == [[True, False, False, False]]


def test_min_p_is_relative_to_the_most_likely_token():
    probs = torch.tensor([[0.6, 0.3, 0.06, 0.04]])
    assert fs.warp_support(probs.log(), Warp(min_p=0.1)).tolist() == [[True, True, True, False]]


def test_the_top_token_always_survives():
    x = torch.tensor([[5.0, 0.0, 0.0]])
    assert fs.warp_support(x, Warp(top_p=1e-6)).tolist() == [[True, False, False]]


def test_a_tightened_min_p_never_empties_the_support():
    x = torch.tensor([[1.0, 0.99, -3.0]])
    tight = Warp(min_p=0.95).widened(-0.15)
    assert tight.min_p > 1.0
    assert fs.warp_support(x, tight).tolist() == [[True, False, False]]


def test_widening_moves_every_cut_by_the_margin():
    w = Warp(t=0.7, top_k=20, top_p=0.8, min_p=0.05)
    loose, tight = w.widened(0.15), w.widened(-0.15)
    assert (loose.top_k, tight.top_k) == (23, 17)
    assert (loose.top_p, tight.top_p) == (pytest.approx(0.95), pytest.approx(0.65))
    assert (loose.min_p, tight.min_p) == (pytest.approx(0.0425), pytest.approx(0.0575))
    assert loose.t == tight.t == 0.7
    assert Warp(top_p=0.95).widened(0.15).top_p == 1.0
    assert Warp(top_k=1).widened(-0.15).top_k == 1


@pytest.mark.parametrize("warp", [Warp(t=0.0), Warp(t=-1.0), Warp(t=math.inf),
                                  Warp(top_p=0.0), Warp(top_p=1.5), Warp(min_p=1.0), Warp(min_p=-0.1)])
def test_a_bad_warp_is_refused(warp):
    with pytest.raises(ValueError):
        fs.check_draw(fs.DRAW_GUMBEL_MAX, t=warp.t, top_k=warp.top_k, top_p=warp.top_p, min_p=warp.min_p)


def test_the_inverse_cdf_draw_has_no_min_p():
    with pytest.raises(ValueError, match="min_p"):
        fs.check_draw(fs.DRAW_INVERSE_CDF, min_p=0.05)


def test_a_warp_round_trips_as_a_dict():
    w = Warp(t=0.6, top_k=20, top_p=0.95, min_p=0.05)
    assert Warp.from_dict(w.to_dict()) == w
    with pytest.raises(ValueError):
        Warp.from_dict({**w.to_dict(), "beam": 2})
    with pytest.raises(ValueError):
        Warp.from_dict({"t": 0.6})
    with pytest.raises(ValueError):
        Warp.from_dict({**w.to_dict(), "top_k": -1})


# ------------------------------------------------------------------ pick


def test_the_warped_gumbel_pick_samples_the_warped_distribution():
    probs = torch.tensor([0.4, 0.25, 0.15, 0.1, 0.06, 0.04])
    warp = Warp(t=0.5, top_k=4, top_p=0.9)
    logits = probs.log() + 3.0
    target = fs.warp(logits, t=warp.t, top_k=warp.top_k, top_p=warp.top_p).double()
    n = 30000
    picks = fs.pick_rows(logits.expand(n, -1).contiguous(), _us(n), draw=fs.DRAW_GUMBEL_MAX,
                         t=warp.t, top_k=warp.top_k, top_p=warp.top_p)
    counts = torch.bincount(picks, minlength=6).double()
    assert counts[target == 0].sum() == 0
    kept = target > 0
    expected = target[kept] * n
    chi2 = float(((counts[kept] - expected) ** 2 / expected).sum())
    assert int(kept.sum()) == 3
    assert chi2 < 13.82  # df=2, alpha=0.001


def test_the_warped_pick_is_argmax_of_scaled_logits_plus_noise_over_the_support():
    torch.manual_seed(0)
    logits = torch.randn(6, 40) * 3
    us = _us(6)
    warp = Warp(t=0.6, top_k=10, top_p=0.9, min_p=0.02)
    picks = fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX, t=warp.t, top_k=warp.top_k,
                         top_p=warp.top_p, min_p=warp.min_p)
    x = logits / warp.t
    keys = torch.tensor([fs.gumbel_key(u) for u in us])
    scores = x + fs.gumbel_noise(keys, torch.arange(40))
    scores[~fs.warp_support(x, warp)] = -math.inf
    assert picks.tolist() == scores.argmax(dim=-1).tolist()


def test_the_full_warp_pick_is_unchanged():
    torch.manual_seed(1)
    logits = torch.randn(5, 30)
    us = _us(5)
    keys = torch.tensor([fs.gumbel_key(u) for u in us])
    assert (fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX).tolist()
            == fs.gumbel_argmax_reference(logits, keys)[0].tolist())


# ------------------------------------------------------------------ diagnostics


def _diag(logits, tokens, us, warp, margin=0.15):
    return fs.seed_consistency_diagnostics(
        logits, tokens, us, t=warp.t, top_k=warp.top_k, top_p=warp.top_p, min_p=warp.min_p,
        edge_margin=margin, stochastic_threshold=0.99, boundary_epsilon=0.0, draw=fs.DRAW_GUMBEL_MAX)


def _picks(logits, us, warp):
    return fs.pick_rows(logits, us, draw=fs.DRAW_GUMBEL_MAX, t=warp.t, top_k=warp.top_k,
                        top_p=warp.top_p, min_p=warp.min_p).tolist()


def test_honest_warped_picks_are_exact():
    torch.manual_seed(2)
    logits = torch.randn(64, 100) * 2
    us = _us(64)
    warp = Warp(t=0.7, top_k=20, top_p=0.8, min_p=0.05)
    d = _diag(logits, _picks(logits, us, warp), us, warp)
    assert (d.n_positions, d.n_drawn_exact_match, d.max_cdf_miss) == (64, 64, 0.0)


def test_a_token_far_outside_the_support_is_impossible():
    """The lowest-ranked token with the biggest noise would beat the pick unmasked: the cut, not the
    score, refuses it."""
    torch.manual_seed(3)
    logits = torch.randn(1, 200) * 2
    us = _us(1)
    warp = Warp(top_k=5)
    keys = torch.tensor([fs.gumbel_key(us[0])])
    noise = fs.gumbel_noise(keys, torch.arange(200))[0]
    ranked = logits[0].argsort()
    outsider = int(ranked[:150][noise[ranked[:150]].argmax()])
    d = _diag(logits, [outsider], us, warp)
    assert d.n_drawn_exact_match == 0
    assert d.max_cdf_miss == 1.0


def test_an_edge_token_the_miner_kept_is_an_ordinary_miss():
    """The miner's cut kept one more token than the validator's (its top_p a hair higher): the
    validator scores that token against its tight support, a near-tie, not an impossible token."""
    x = torch.tensor([[2.0, 1.5, 1.0, 0.99, -5.0, -6.0]])
    validator, miner = Warp(top_p=0.84), Warp(top_p=0.86)
    assert fs.warp_support(x, validator).tolist() != fs.warp_support(x, miner).tolist()
    for seed in [f"{i:02d}" for i in range(1, 60)]:
        us = _us(1, seed)
        token = _picks(x, us, miner)
        if token == _picks(x, us, validator):
            continue
        d = _diag(x, token, us, validator)
        assert d.n_drawn_exact_match == 0
        assert d.max_cdf_miss < 1.0
        return
    pytest.fail("no draw where the cuts disagree")


def test_a_boundary_best_the_miner_cut_does_not_make_a_large_gap():
    """The validator kept a boundary token the miner cut; when its noise made it the validator's
    pick, the miner's token is judged against the tight support, so the gap stays a near-tie's."""
    x = torch.tensor([[2.0, 1.5, 1.0, 0.99, -5.0, -6.0]])
    validator, miner = Warp(top_p=0.86), Warp(top_p=0.84)
    for seed in [f"{i:02d}" for i in range(1, 60)]:
        us = _us(1, seed)
        token, best = _picks(x, us, miner), _picks(x, us, validator)
        if token == best:
            continue
        d = _diag(x, token, us, validator)
        gap = -math.log1p(-d.max_cdf_miss)
        assert gap < 1.0
        return
    pytest.fail("no draw where the cuts disagree")


def test_no_margin_judges_against_the_exact_support():
    x = torch.tensor([[2.0, 1.5, 1.0, 0.99, -5.0, -6.0]])
    validator, miner = Warp(top_p=0.84), Warp(top_p=0.86)
    for seed in [f"{i:02d}" for i in range(1, 60)]:
        us = _us(1, seed)
        token = _picks(x, us, miner)
        if token != _picks(x, us, validator) and not fs.warp_support(x, validator)[0, token[0]]:
            assert _diag(x, token, us, validator, margin=0.0).max_cdf_miss == 1.0
            return
    pytest.fail("no draw where the miner picked outside the validator's support")


def test_full_warp_diagnostics_are_unchanged_by_the_margin():
    torch.manual_seed(4)
    logits = torch.randn(16, 50)
    us = _us(16)
    tokens = torch.randint(0, 50, (16,)).tolist()
    a = _diag(logits, tokens, us, Warp(), margin=0.0)
    b = _diag(logits, tokens, us, Warp(), margin=0.3)
    assert a == b


# ------------------------------------------------------------------ miner (vLLM processor)


def _job_params(rollout_index, warp=None, prompt_idx=7):
    from types import SimpleNamespace

    from reliquary.miner.vllm_generation import forced_seed_extra_args
    from tests.unit.test_vllm_forced_seed import CHECKPOINT, RANDOMNESS

    return SimpleNamespace(extra_args=forced_seed_extra_args(
        randomness=RANDOMNESS, prompt_idx=prompt_idx, checkpoint_hash=CHECKPOINT, rollout_index=rollout_index,
        domain="job-domain", draw=fs.DRAW_GUMBEL_MAX, warp=None if warp is None else warp.to_dict()))


def _job_expected(row_logits, rollout_index, position, warp):
    from tests.unit.test_vllm_forced_seed import CHECKPOINT, RANDOMNESS

    u = fs.u_at(RANDOMNESS, 7, CHECKPOINT, rollout_index, position, domain="job-domain")
    return int(fs.pick_rows(row_logits.reshape(1, -1), [u], draw=fs.DRAW_GUMBEL_MAX, t=warp.t, top_k=warp.top_k,
                            top_p=warp.top_p, min_p=warp.min_p)[0])


def test_the_runner_draws_each_requests_own_warp(monkeypatch):
    from reliquary import constants
    from tests.unit.test_vllm_forced_seed import VOCAB, _Engine

    monkeypatch.setattr(constants, "FORCED_DRAW", fs.DRAW_GUMBEL_MAX)
    torch.manual_seed(30)
    warps = [Warp(), Warp(t=0.6, top_k=5, top_p=0.9), Warp(t=0.8, min_p=0.1)]
    engine = _Engine(monkeypatch)
    for row, warp in enumerate(warps):
        engine.add(f"r{row}", row, _job_params(row, None if warp.identity else warp))
    for step in range(6):
        logits = torch.randn(3, VOCAB) * 3
        picks = engine.step(logits)
        for row, warp in enumerate(warps):
            assert int(picks[row]) == _job_expected(logits[row], row, step, warp)


def test_a_warp_without_its_draw_is_refused():
    from reliquary.miner.vllm_generation import FORCED_SEED_KEY, _forced_state

    forced = _job_params(0, Warp(t=0.6)).extra_args[FORCED_SEED_KEY]
    del forced["draw"]
    with pytest.raises(ValueError, match="warp"):
        _forced_state(forced)
    with pytest.raises(ValueError):
        _forced_state({**_job_params(0).extra_args[FORCED_SEED_KEY], "warp": {"t": 0.0, "top_k": 0, "top_p": 1.0,
                                                                              "min_p": 0.0}})


def test_a_request_without_a_warp_is_unchanged_on_the_wire():
    from reliquary.miner.vllm_generation import FORCED_SEED_KEY

    assert "warp" not in _job_params(0).extra_args[FORCED_SEED_KEY]


# ------------------------------------------------------------------ job


def _job_raw(**sampling):
    from tests.unit.test_eval_draw import _raw

    raw = _raw()
    raw["sampling"] = {**raw["sampling"], **sampling}
    return raw


def test_a_draw_job_samples_its_own_warp():
    from reliquary.corpus import eval_draw
    from reliquary.corpus.job import parse_job

    job = parse_job(_job_raw(temperature=0.6, top_k=20, top_p=0.95, min_p=0.05))
    assert eval_draw.job_warp(job) == Warp(t=0.6, top_k=20, top_p=0.95, min_p=0.05)
    assert job.to_contract()["sampling"]["min_p"] == 0.05
    assert parse_job(job.to_contract()) == job
    assert eval_draw.draw_context(job, 3, 1)["warp"] == {"t": 0.6, "top_k": 20, "top_p": 0.95, "min_p": 0.05}


def test_a_full_warp_job_is_unchanged():
    """No min_p in the contract and no warp on the wire: every existing draw job hashes and draws as before."""
    from reliquary.corpus import eval_draw
    from reliquary.corpus.job import parse_job

    job = parse_job(_job_raw())
    assert "min_p" not in job.to_contract()["sampling"]
    assert parse_job(_job_raw(min_p=0.0)).to_contract() == job.to_contract()
    assert "warp" not in eval_draw.draw_context(job, 3, 1)
    assert eval_draw.job_warp(job).identity


@pytest.mark.parametrize("sampling", [{"min_p": 1.0}, {"min_p": -0.1}, {"min_p": "0.1"}, {"min_p": True}])
def test_a_bad_min_p_is_refused(sampling):
    from reliquary.corpus.job import JobError, parse_job

    with pytest.raises(JobError):
        parse_job(_job_raw(**sampling))


def test_min_p_needs_the_draw():
    """Free-sampling miners do not apply min_p: only a draw job, whose pick the miner computes itself, takes it."""
    from reliquary.corpus.job import JobError, parse_job

    raw = _job_raw(min_p=0.05)
    del raw["sampling"]["draw"]
    raw.pop("seed")
    with pytest.raises(JobError, match="min_p"):
        parse_job(raw)


def test_the_edge_margin_is_a_judge_parameter():
    from reliquary.corpus.job import DEFAULT_DRAW_JUDGE, draw_judge_from, parse_job

    assert DEFAULT_DRAW_JUDGE.edge_margin == 0.15
    assert draw_judge_from({"edge_margin": 0.3}).edge_margin == 0.3
    raw = _job_raw(temperature=0.7, top_p=0.8)
    raw["judge"] = {"edge_margin": 0.25}
    with pytest.raises(Exception):
        parse_job(raw)                       # a manifest's judge is complete
    raw["judge"] = {**DEFAULT_DRAW_JUDGE.to_contract(), "edge_margin": 0.25}
    assert parse_job(raw).judge.edge_margin == 0.25
    for bad in (-0.1, 0.6):
        with pytest.raises(Exception):
            draw_judge_from({"edge_margin": bad})


def test_the_auditor_diagnoses_with_the_jobs_warp_and_margin():
    """Honest picks under the job's warp are exact; diagnosed as T=1 over the full vocabulary they would not be."""
    from reliquary.corpus import eval_draw
    from reliquary.corpus.job import parse_job

    job = parse_job(_job_raw(temperature=0.5, top_k=8, top_p=0.9))
    torch.manual_seed(5)
    logits = torch.randn(40, 64) * 2
    u = eval_draw.uniforms(job, 3, 1, 0, 40)
    tokens = fs.pick_rows(logits, u, draw=fs.DRAW_GUMBEL_MAX, t=0.5, top_k=8, top_p=0.9).tolist()
    assert eval_draw.diagnostics(logits, tokens, u, job).n_drawn_exact_match == 40
    assert eval_draw.diagnostics(logits, tokens, u).n_drawn_exact_match < 40


# ------------------------------------------------------------------ manifest and schedule


def test_a_manifest_carries_min_p_only_when_set(monkeypatch):
    from reliquary.cli.main import build_job_manifest

    monkeypatch.setattr("reliquary.validator.corpus_service.prompt_job_for_spec", lambda *a, **k: None)
    from tests.unit.test_eval_draw import SOURCE

    common = dict(job_id="j", checkpoint_repo="r", checkpoint_revision="v", checkpoint_sha256="a" * 64,
                  prompt_source=SOURCE, prompt_count=1, seed=7, renderer_id="chat-template-thinking-v1", eos_token_id=1, slots_per_prompt=1,
                  temperature=0.6, top_p=0.95, top_k=20, min_new_tokens=2, max_new_tokens=8, n=1, grader_id=None,
                  threshold=None, prompt_order="free", deadline_round=None, draw="gumbel_max")
    assert "min_p" not in build_job_manifest(**common)["sampling"]
    assert build_job_manifest(**common, min_p=0.05)["sampling"]["min_p"] == 0.05


def test_the_schedule_passes_its_warp_to_each_job(tmp_path):
    from tests.unit.test_eval_rl_schedule import World, scheduler

    plain = World()
    scheduler(tmp_path / "a", plain).tick(execute=True)
    assert all("--top-k" not in c and "--min-p" not in c for c in plain.commands)
    warped = World()
    scheduler(tmp_path / "b", warped, temperature=0.6, top_p=0.95, top_k=20, min_p=0.05).tick(execute=True)
    command = warped.commands[0]
    assert command[command.index("--top-k") + 1] == "20"
    assert command[command.index("--min-p") + 1] == "0.05"
    assert command[command.index("--temperature") + 1] == "0.6"


def test_warped_chunks_follow_their_own_workspace_and_agree(monkeypatch):
    torch.manual_seed(5)
    logits = torch.randn(9, 60) * 2
    us = _us(9)
    # No top-k: the whole-vocabulary path, which keeps its own workspace.
    warp = Warp(t=0.7, top_p=0.9)
    tokens = _picks(logits, us, warp)
    tokens[4] = (tokens[4] + 1) % 60
    whole = _diag(logits, tokens, us, warp)

    seen = []
    original = fs._gumbel_warp_chunk_stats

    def recording(chunk, *args):
        seen.append(int(chunk.shape[0]))
        return original(chunk, *args)

    monkeypatch.setattr(fs, "_gumbel_warp_chunk_stats", recording)
    monkeypatch.setattr(fs, "_WARP_WORKSPACE_BYTES", 4 * 60 * 4 * 8)
    # The general budget no longer caps a warped chunk.
    monkeypatch.setattr(fs, "_DIAGNOSTIC_FLOAT_WORKSPACE_BYTES", 1)
    assert _diag(logits, tokens, us, warp) == whole
    assert seen == [4, 4, 1]


@pytest.mark.parametrize("warp", [Warp(t=1.0, top_k=20, top_p=0.95), Warp(t=0.6, top_k=8),
                                  Warp(t=1.3, top_k=30, top_p=0.8, min_p=0.05), Warp(t=1.0, top_k=1)])
@pytest.mark.parametrize("margin", [0.0, 0.05, 0.2])
def test_the_top_k_slice_judges_like_the_whole_vocabulary(warp, margin):
    torch.manual_seed(11)
    logits = torch.randn(64, 500) * 3
    logits[::7] *= 0.05  # near-flat rows: many near-ties at the cut
    us = _us(64)
    tokens = _picks(logits, us, warp)
    for row in range(0, 64, 3):  # edits: an edge token, a far token, a near one
        tokens[row] = int(torch.argsort(logits[row], descending=True)[(row // 3) % 40])
    args = (torch.tensor(tokens), us, warp, margin, 0.95)
    sliced = fs._gumbel_warp_chunk_stats(logits, *args)
    dense = fs._gumbel_warp_chunk_stats_dense(logits, *args)
    assert torch.equal(sliced[0], dense[0])
    assert torch.equal(sliced[2], dense[2])
    assert torch.allclose(sliced[1], dense[1], atol=1e-6, rtol=0)


def test_a_top_k_slice_takes_bigger_chunks_from_the_warp_workspace(monkeypatch):
    seen = []
    original = fs._gumbel_warp_chunk_stats

    def recording(chunk, *args):
        seen.append(int(chunk.shape[0]))
        return original(chunk, *args)

    monkeypatch.setattr(fs, "_gumbel_warp_chunk_stats", recording)
    # Two [rows, V] float blocks a row: the slice needs neither the masks nor the score copies.
    monkeypatch.setattr(fs, "_WARP_WORKSPACE_BYTES", 4 * 60 * 4 * 2)
    torch.manual_seed(5)
    logits = torch.randn(9, 60) * 2
    us = _us(9)
    warp = Warp(t=0.7, top_k=12, top_p=0.9)
    _diag(logits, _picks(logits, us, warp), us, warp)
    assert seen == [4, 4, 1]


@pytest.mark.parametrize("warp", [Warp(t=1.0, top_p=0.95), Warp(t=1.0, top_k=20, top_p=0.95), Warp()])
def test_far_misses_are_counted_on_every_path(warp):
    """Each token written far from the draw is one far miss (the edit pattern); honest picks are none."""
    torch.manual_seed(3)
    logits = torch.randn(40, 300) * 4
    us = _us(40)
    tokens = _picks(logits, us, warp)
    assert _diag(logits, tokens, us, warp).n_far_miss == 0
    edited = list(tokens)
    for row in (5, 17, 29):  # the least likely token: far past any near-tie
        edited[row] = int(torch.argmin(logits[row]))
    d = _diag(logits, edited, us, warp)
    assert d.n_far_miss == 3 and fs._FAR_MISS < d.max_cdf_miss


def test_one_shared_ranking_gives_every_warp_the_masks_warp_support_gives():
    torch.manual_seed(0)
    x = torch.randn(7, 300) * 3
    x[0, :5] = 2.0  # ties
    base = Warp(t=1.0, top_p=0.9, min_p=0.02)
    for warps in ([base, base.widened(-0.05), base.widened(0.05)],
                  [Warp(top_p=0.97), Warp(top_p=0.97).widened(0.05)],  # the loose cut reaches 1
                  [Warp(min_p=0.1), Warp()]):
        assert all(torch.equal(got, fs.warp_support(x, w)) for got, w in zip(fs._warp_supports(x, warps), warps))


def test_a_top_k_cut_keeps_its_own_ranking():
    x = torch.randn(3, 50)
    warps = [Warp(top_k=5, top_p=0.9), Warp(top_k=7, top_p=0.95)]
    assert all(torch.equal(got, fs.warp_support(x, w)) for got, w in zip(fs._warp_supports(x, warps), warps))


def test_the_forward_packs_only_under_flash_attention_without_linear_layers(monkeypatch):
    from types import SimpleNamespace

    from reliquary.validator.corpus_audit import _packed

    def model(attn, layers=None):
        return SimpleNamespace(config=SimpleNamespace(_attn_implementation=attn, layer_types=layers))

    assert not _packed(model("flash_attention_2"))  # off unless asked
    monkeypatch.setenv("RELIQUARY_PACKED_FORWARD", "1")
    assert _packed(model("flash_attention_2"))
    assert not _packed(model("sdpa"))
    assert not _packed(model("flash_attention_2", ["linear_attention", "full_attention"]))
