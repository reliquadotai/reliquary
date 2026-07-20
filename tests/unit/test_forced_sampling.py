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
    original_warp_batch = fs._warp_batch

    def _recording_warp_batch(chunk, *args, **inner_kwargs):
        seen_rows.append(int(chunk.shape[0]))
        return original_warp_batch(chunk, *args, **inner_kwargs)

    monkeypatch.setattr(fs, "_warp_batch", _recording_warp_batch)
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
