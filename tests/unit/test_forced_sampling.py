import pytest
import torch
from reliquary.environment import forced_sampling as fs


def test_pick_inverse_cdf_boundaries():
    probs = torch.tensor([0.5, 0.5])          # token 0 -> [0,0.5), token 1 -> [0.5,1)
    assert fs.pick(probs, 0.0) == 0
    assert fs.pick(probs, 0.49) == 0
    assert fs.pick(probs, 0.5) == 1
    assert fs.pick(probs, 0.999) == 1


def test_pick_matches_probs_device_no_mismatch_error():
    # pick must build its comparison tensor on probs.device (CUDA-or-CPU) so
    # the GPU-resident verifier path never round-trips logits through PCIe.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    probs = torch.tensor([0.5, 0.5], device=device)
    assert fs.pick(probs, 0.5) == 1


def test_warp_topk_topp_masks():
    logits = torch.tensor([10.0, 9.0, 1.0, 1.0])
    probs = fs.warp(logits, t=0.6, top_k=2, top_p=1.0)
    assert probs[2] == 0.0 and probs[3] == 0.0          # top_k=2 masks tail
    assert torch.isclose(probs.sum(), torch.tensor(1.0))


def test_u_at_deterministic_and_field_sensitive():
    a = fs.u_at("cd" * 16, 7, "sha:abc", 0, 3)
    b = fs.u_at("cd" * 16, 7, "sha:abc", 0, 3)
    assert a == b and 0.0 <= a < 1.0
    assert fs.u_at("cd" * 16, 7, "sha:abc", 0, 4) != a   # position changes it
    assert fs.u_at("cd" * 16, 8, "sha:abc", 0, 3) != a   # prompt changes it
    assert fs.u_at("ce" * 16, 7, "sha:abc", 0, 3) != a   # randomness changes it
    assert fs.u_at("cd" * 16, 7, "sha:xyz", 0, 3) != a   # checkpoint changes it
    assert fs.u_at("cd" * 16, 7, "sha:abc", 1, 3) != a   # rollout changes it


def test_u_at_is_identity_free_to_kill_variance_farming():
    """v2 anti-farming: the forced stream no longer takes a hotkey, so the group
    for a prompt is identical for every miner in the window. One operator's N
    hotkeys therefore get N copies of the SAME draw — there is nothing to farm.
    Passing a hotkey must be a TypeError, not a silently-ignored argument."""
    import pytest
    with pytest.raises(TypeError):
        fs.u_at("cd" * 16, "hk1", 7, "sha:abc", 0, 3)



def _seed_consistency(logits, token_ids, u_values, *, t, top_k, top_p,
                      stochastic_threshold):
    """(n_stochastic, n_exact_match) via the production diagnostics path."""
    d = fs.seed_consistency_diagnostics(
        logits, token_ids, u_values, t=t, top_k=top_k, top_p=top_p,
        stochastic_threshold=stochastic_threshold, boundary_epsilon=0.0)
    return d.n_stochastic, d.n_exact_match

def test_seed_consistency_perfect_when_tokens_follow_u():
    # two peaked positions (argmax ~1 -> not stochastic) + two flat positions (stochastic)
    logits = torch.tensor([[10.0, 0.0, 0.0],      # argmax token 0
                           [0.2, 0.1, 0.0],       # flat -> stochastic
                           [10.0, 0.0, 0.0],       # argmax token 0
                           [0.1, 0.2, 0.15]])      # flat -> stochastic
    u = [fs.u_at("r", 0, "c", 0, t) for t in range(4)]
    # tokens = what the forced u actually picks (honest miner)
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i]) for i in range(4)]
    n_stoch, n_match = _seed_consistency(
        logits, tokens, u, t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    assert n_stoch >= 1
    assert n_match == n_stoch                       # honest -> every stochastic pos matches


def test_seed_consistency_low_when_tokens_ignore_u():
    logits = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.2, 0.15], [0.0, 0.1, 0.2]])
    u = [fs.u_at("r", 0, "c", 0, t) for t in range(3)]
    wrong = [fs.u_at("OTHER", 0, "c", 0, t) for t in range(3)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), wrong[i]) for i in range(3)]
    n_stoch, n_match = _seed_consistency(
        logits, tokens, u, t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    assert n_stoch >= 1
    assert n_match < n_stoch                         # ignoring u -> mismatches appear


def _seed_consistency_reference(logits, token_ids, u_values, *, t, top_k, top_p,
                                stochastic_threshold):
    """Explicit per-position reference (the original loop) the vectorized
    implementation must match bit-for-bit."""
    n_stoch = n_match = 0
    n = min(len(token_ids), len(u_values), logits.shape[0])
    for i in range(n):
        probs = fs.warp(logits[i], t=t, top_k=top_k, top_p=top_p)
        if float(probs.max()) < stochastic_threshold:
            n_stoch += 1
            if fs.pick(probs, u_values[i]) == int(token_ids[i]):
                n_match += 1
    return n_stoch, n_match


def test_seed_consistency_matches_per_position_reference_on_batch():
    # Vectorized seed_consistency must equal the per-position reference on a
    # varied batch: peaked (excluded) + flat rows, ~1/3 corrupted tokens.
    torch.manual_seed(0)
    n, vocab = 16, 64
    logits = torch.randn(n, vocab)
    for i in (2, 7, 11):
        logits[i, 0] = 50.0                          # peaked -> not stochastic
    u = [fs.u_at("rand", 3, "ckpt", 0, i) for i in range(n)]
    tokens = []
    for i in range(n):
        p = fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i])
        tokens.append(p if i % 3 else (p + 1) % vocab)   # corrupt ~1/3
    kw = dict(t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    got = _seed_consistency(logits, tokens, u, **kw)
    ref = _seed_consistency_reference(logits, tokens, u, **kw)
    assert got == ref
    assert ref[0] > 0 and ref[1] < ref[0]            # meaningful: stochastic + mismatches


def test_seed_consistency_empty_batch_returns_zeros():
    assert _seed_consistency(
        torch.zeros(0, 5), [], [], t=0.6, top_k=20, top_p=0.95,
        stochastic_threshold=0.99) == (0, 0)


def test_seed_consistency_truncates_to_shortest_input():
    # n = min(len(tokens), len(u), rows); extra logits rows are ignored.
    torch.manual_seed(1)
    logits = torch.randn(6, 32)
    u = [fs.u_at("z", 0, "c", 0, i) for i in range(4)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i])
              for i in range(4)]
    kw = dict(t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    assert _seed_consistency(logits, tokens, u, **kw) == \
        _seed_consistency_reference(logits, tokens, u, **kw)


def test_cdf_diagnostics_accepts_only_calibrated_boundary_distance():
    logits = torch.log(torch.tensor([[0.5, 0.5]]))

    near = fs.seed_consistency_diagnostics(
        logits,
        [0],
        [0.5005],
        t=1.0,
        top_k=0,
        top_p=1.0,
        stochastic_threshold=0.99,
        boundary_epsilon=0.001,
    )
    hard = fs.seed_consistency_diagnostics(
        logits,
        [0],
        [0.5005],
        t=1.0,
        top_k=0,
        top_p=1.0,
        stochastic_threshold=0.99,
        boundary_epsilon=0.0001,
    )

    assert near.n_exact_match == 0
    assert near.n_boundary_match == 1
    assert near.n_hard_mismatch == 0
    assert near.max_cdf_miss == pytest.approx(0.0005, abs=1e-6)
    assert hard.n_boundary_match == 0
    assert hard.n_hard_mismatch == 1
    assert hard.n_miss_gt_0_01 == 0
    assert hard.n_miss_gt_0_05 == 0
    assert hard.n_miss_gt_0_10 == 0


def test_cdf_diagnostics_checks_near_deterministic_positions_too():
    logits = torch.tensor([[10.0, 0.0]])
    diagnostics = fs.seed_consistency_diagnostics(
        logits,
        [1],
        [0.5],
        t=1.0,
        top_k=0,
        top_p=1.0,
        stochastic_threshold=0.99,
        boundary_epsilon=0.001,
    )

    assert diagnostics.n_stochastic == 0
    assert diagnostics.n_exact_match == 0
    assert diagnostics.n_hard_mismatch == 1
    assert diagnostics.n_deterministic_hard_mismatch == 1
    assert diagnostics.n_miss_gt_0_01 == 1
    assert diagnostics.n_miss_gt_0_05 == 1
    assert diagnostics.n_miss_gt_0_10 == 1


def test_cdf_diagnostics_reports_completion_offset_of_first_hard_mismatch():
    logits = torch.log(torch.tensor([[0.5, 0.5], [0.5, 0.5]]))
    diagnostics = fs.seed_consistency_diagnostics(
        logits,
        [0, 0],
        [0.25, 0.75],
        t=1.0,
        top_k=0,
        top_p=1.0,
        stochastic_threshold=0.99,
        boundary_epsilon=0.0,
        position_offsets=[4, 11],
    )

    assert diagnostics.n_hard_mismatch == 1
    assert diagnostics.first_hard_mismatch_offset == 11


def test_cdf_diagnostics_chunks_selected_logit_rows(monkeypatch):
    torch.manual_seed(7)
    logits = torch.randn(9, 17)
    positions = [8, 2, 6, 1, 4]
    selected = logits[positions]
    u_values = [0.11, 0.32, 0.53, 0.74, 0.95]
    token_ids = [
        fs.pick(fs.warp(row, t=0.6, top_k=8, top_p=0.95), u)
        for row, u in zip(selected, u_values)
    ]
    kwargs = dict(
        t=0.6,
        top_k=8,
        top_p=0.95,
        stochastic_threshold=0.99,
        boundary_epsilon=0.001,
        position_offsets=[10, 20, 30, 40, 50],
    )
    expected = fs.seed_consistency_diagnostics(
        selected, token_ids, u_values, **kwargs,
    )

    seen_rows = []
    original_intervals = fs._interval_stats_sparse

    def _recording_intervals(chunk, *args, **inner_kwargs):
        seen_rows.append(int(chunk.shape[0]))
        return original_intervals(chunk, *args, **inner_kwargs)

    monkeypatch.setattr(fs, "_interval_stats_sparse", _recording_intervals)
    monkeypatch.setattr(
        fs,
        "_DIAGNOSTIC_FLOAT_WORKSPACE_BYTES",
        2 * logits.shape[-1] * 4,
    )
    actual = fs.seed_consistency_diagnostics(
        logits,
        token_ids,
        u_values,
        logit_positions=positions,
        **kwargs,
    )

    assert actual == expected
    assert seen_rows == [2, 2, 1]


# ---------------------------------------------------------------------------
# Sparse (top-k window) interval path vs the dense reference
# ---------------------------------------------------------------------------

def _diag_kwargs(**overrides):
    kwargs = dict(t=0.6, top_k=20, top_p=0.95,
                  stochastic_threshold=0.95, boundary_epsilon=1e-4)
    kwargs.update(overrides)
    return kwargs


def _random_case(seed, rows=64, vocab=1024):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(rows, vocab, generator=g)
    toks = torch.randint(0, vocab, (rows,), generator=g).tolist()
    u = torch.rand(rows, generator=g).tolist()
    return logits, toks, u


def _counts(diag):
    return (diag.n_stochastic, diag.n_exact_match, diag.n_boundary_match,
            diag.n_hard_mismatch, diag.n_deterministic_hard_mismatch,
            diag.n_miss_gt_0_01, diag.n_miss_gt_0_05, diag.n_miss_gt_0_10,
            diag.first_hard_mismatch_offset)


def test_sparse_intervals_match_dense_reference():
    for seed in range(8):
        logits, toks, u = _random_case(seed)
        tok_t = torch.tensor(toks)
        d = fs._interval_stats_dense(logits, tok_t, t=0.6, top_k=20, top_p=0.95)
        s = fs._interval_stats_sparse(logits, tok_t, t=0.6, top_k=20, top_p=0.95)
        for dense, sparse in zip(d, s):
            assert torch.allclose(dense, sparse, atol=1e-6), seed


def test_sparse_diagnostics_counts_identical_to_reference():
    # End-to-end: the shipped function (sparse+fallback) must classify every
    # position exactly like a dense-only run.
    for seed in range(8):
        logits, toks, u = _random_case(seed)
        got = fs.seed_consistency_diagnostics(logits, toks, u, **_diag_kwargs())
        # dense-only reference: force every row through the fallback path
        # by shrinking the tie window to nothing via a huge top_k.
        probs_ref = fs._warp_batch(logits, t=0.6, top_k=20, top_p=0.95)
        cdf = torch.cumsum(probs_ref, dim=-1)
        row = torch.arange(logits.shape[0])
        tok_t = torch.tensor(toks)
        upper = cdf[row, tok_t]
        mass = probs_ref[row, tok_t]
        lower = upper - mass
        uu = torch.tensor(u, dtype=upper.dtype)
        exact = (uu >= lower) & (uu < upper)
        assert got.n_exact_match == int(
            ((probs_ref.max(-1).values < 0.95) & exact).sum()
        )


def test_sparse_boundary_epsilon_classification_matches():
    # Uniforms placed exactly at interval edges +- tiny offsets.
    logits, toks, _ = _random_case(99, rows=32)
    tok_t = torch.tensor(toks)
    _, lower, upper, _ = fs._interval_stats_dense(
        logits, tok_t, t=0.6, top_k=20, top_p=0.95)
    for eps in (0.0, 5e-5, 2e-4):
        u = (lower - eps).clamp(0, 1).tolist()
        a = fs.seed_consistency_diagnostics(logits, toks, u, **_diag_kwargs())
        assert a.n_positions == 32
        # miss must equal eps (within fp) -> boundary iff eps <= 1e-4
        expected_boundary = 32 if eps <= 1e-4 else a.n_boundary_match
        assert a.n_boundary_match == expected_boundary


def test_sparse_tie_overflow_falls_back_to_dense():
    # All logits identical: every token ties at the kth value, the window
    # cannot contain the tie set, and the fallback must produce the dense
    # answer instead of a poisoned NaN.
    rows, vocab = 4, 256
    logits = torch.zeros(rows, vocab)
    toks = [0, 1, 128, 255]
    u = [0.0, 0.5, 0.5, 0.999]
    got = fs.seed_consistency_diagnostics(logits, toks, u, **_diag_kwargs())
    assert got.n_positions == rows
    ref_probs = fs._warp_batch(logits, t=0.6, top_k=20, top_p=0.95)
    assert torch.isfinite(ref_probs).all()


def test_sparse_handles_token_outside_topk_window():
    # Submitted token far below the top-k cut: mass 0, interval empty, and
    # the miss distance must match the dense computation.
    logits = torch.zeros(1, 512)
    logits[0, :21] = 10.0          # 21-way tie at the top -> overflow+fallback
    logits2 = torch.zeros(1, 512)
    logits2[0, :8] = torch.arange(8, 0, -1).float() * 3
    for lg in (logits, logits2):
        toks = [500]               # never a survivor
        got = fs.seed_consistency_diagnostics(lg, toks, [0.5], **_diag_kwargs())
        assert got.n_positions == 1
        assert got.n_exact_match == 0
        assert got.n_hard_mismatch == 1


def test_full_support_diagnostics_verify_honest_picks_exactly():
    """v4 sampling (top_k=0): every honest pick must verify, wherever it lands.

    The sparse window is sized by top_k; at 0 it degenerates to 33 tokens with
    the dense fallback unreachable, so honest full-support picks outside the
    window (or in-window picks whose interval misses the tail mass) would be
    misclassified. Full support must route to the full-vocab intervals.
    """
    g = torch.Generator().manual_seed(5)
    rows, vocab = 48, 4096
    logits = torch.randn(rows, vocab, generator=g)   # flat: honest picks spread
    u = torch.rand(rows, generator=g).tolist()
    for top_k in (0, -1):                            # profile and verl sentinels
        kw = dict(t=1.0, top_k=top_k, top_p=1.0)
        toks = [fs.pick(fs.warp(logits[i], **kw), u[i]) for i in range(rows)]
        got = fs.seed_consistency_diagnostics(
            logits, toks, u, **kw,
            stochastic_threshold=0.99, boundary_epsilon=1e-6,
        )
        assert got.n_positions == rows
        assert got.n_stochastic == rows              # randn(4096) is never peaked
        assert got.n_exact_match == rows, "honest full-support pick rejected"
        assert got.n_hard_mismatch == 0


def test_full_support_diagnostics_match_dense_reference_counts():
    # Same contract as test_sparse_diagnostics_counts_identical_to_reference,
    # at the v4 envelope: ~1/3 corrupted tokens must classify exactly as a
    # dense-only computation says, no more and no fewer.
    logits, toks, u = _random_case(11, rows=64, vocab=2048)
    kw = dict(t=1.0, top_k=0, top_p=1.0)
    got = fs.seed_consistency_diagnostics(
        logits, toks, u, **kw, stochastic_threshold=0.99, boundary_epsilon=1e-6)
    probs_ref = fs._warp_batch(logits, **kw)
    cdf = torch.cumsum(probs_ref, dim=-1)
    row = torch.arange(logits.shape[0])
    tok_t = torch.tensor(toks)
    upper = cdf[row, tok_t]
    lower = upper - probs_ref[row, tok_t]
    uu = torch.tensor(u, dtype=upper.dtype)
    exact = (uu >= lower) & (uu < upper)
    assert got.n_exact_match == int(
        ((probs_ref.max(-1).values < 0.99) & exact).sum()
    )


# ── v4 sampling: warp() is the identity, so the PPO ratio lives in the space
# the samples actually came from. Task 7 (clip-higher) is only interpretable
# because of this. These are regression pins, not change drivers: they pass on
# the current code and must fail before the trainer can silently drift.

_V4 = dict(t=1.0, top_k=0, top_p=1.0)
_V3 = dict(t=0.6, top_k=20, top_p=0.95)


def test_warp_is_plain_softmax_at_v4_sampling_values():
    torch.manual_seed(0)
    logits = torch.randn(1000) * 5.0

    out = fs.warp(logits, **_V4)

    assert torch.allclose(out, torch.softmax(logits.float(), dim=-1), atol=1e-6)
    assert torch.all(out > 0), "full support: v4 must not mask any token to zero"


def test_warp_top_k_zero_and_minus_one_both_disable():
    """0 is the profile's disable sentinel; -1 is verl's. They must agree."""
    torch.manual_seed(1)
    logits = torch.randn(50)

    assert torch.allclose(
        fs.warp(logits, t=1.0, top_k=0, top_p=1.0),
        fs.warp(logits, t=1.0, top_k=-1, top_p=1.0),
    )


def _log_ratio(z_new, z_old, token, **warp_kwargs):
    """log π_new(token) − log π_old(token), measured through warp()."""
    new = fs.warp(z_new, **warp_kwargs)[token]
    old = fs.warp(z_old, **warp_kwargs)[token]
    return float(torch.log(new) - torch.log(old))


def test_v4_sampling_makes_ppo_ratio_space_match_sampling_space():
    """The importance ratio on raw log-probs IS the ratio the samples came from.

    The trainer forms r = π_θ/π_old from raw log-softmax values while the miner
    samples through warp(). At v4 values those coincide exactly, which is what
    makes an epsilon of 0.28 mean 0.28.
    """
    torch.manual_seed(2)
    z_old = torch.randn(500) * 3.0
    z_new = z_old + torch.randn(500) * 0.01  # one small optimiser step apart
    token = int(torch.argmax(z_old))

    raw = float(
        torch.log_softmax(z_new.float(), -1)[token]
        - torch.log_softmax(z_old.float(), -1)[token]
    )

    assert _log_ratio(z_new, z_old, token, **_V4) == pytest.approx(raw, abs=1e-6)


def test_v3_sampling_distorts_the_ppo_ratio_unevenly_across_tokens():
    """The contrast that makes the pin above meaningful.

    Under v3 the trainer forms the ratio on raw log-probs while the miner
    samples through a temperature-0.6, top-20 warp, so the two spaces disagree.
    The disagreement is NOT the clean r_raw^(1/T) rescaling the 08-03 divergence
    audit assumed: that approximation needs Z_old/Z_new ~= 1, which fails on a
    peaked distribution because logsumexp(z/T) tracks the top logit. Measured
    here, the per-token factor between the two spaces ranges from negative to
    well above 1/T — it can even flip the ratio's direction.

    That is worse than a mis-scaled band, and it is why the sampling change has
    to land before clip-higher: epsilon would be tuned against a factor that
    varies token by token rather than a constant exponent.
    """
    torch.manual_seed(3)
    z_old = torch.randn(500) * 3.0
    z_new = z_old + torch.randn(500) * 0.01
    nucleus = torch.topk(z_old, _V3["top_k"]).indices.tolist()

    factors = []
    for token in nucleus:
        raw = float(
            torch.log_softmax(z_new.float(), -1)[token]
            - torch.log_softmax(z_old.float(), -1)[token]
        )
        if abs(raw) < 1e-4:  # ratio undefined against numerical noise
            continue
        factors.append(_log_ratio(z_new, z_old, token, **_V3) / raw)

    assert len(factors) >= 5
    # A clean exponent would put every factor at 1/T. Instead they spread.
    assert max(factors) - min(factors) > 1.0
