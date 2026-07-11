import hashlib
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
    a = fs.u_at("cd" * 16, "hk1", 7, "sha:abc", 0, 3)
    b = fs.u_at("cd" * 16, "hk1", 7, "sha:abc", 0, 3)
    assert a == b and 0.0 <= a < 1.0
    assert fs.u_at("cd" * 16, "hk1", 7, "sha:abc", 0, 4) != a   # position changes it
    assert fs.u_at("cd" * 16, "hk2", 7, "sha:abc", 0, 3) != a   # hotkey changes it


def test_seed_consistency_perfect_when_tokens_follow_u():
    # two peaked positions (argmax ~1 -> not stochastic) + two flat positions (stochastic)
    logits = torch.tensor([[10.0, 0.0, 0.0],      # argmax token 0
                           [0.2, 0.1, 0.0],       # flat -> stochastic
                           [10.0, 0.0, 0.0],       # argmax token 0
                           [0.1, 0.2, 0.15]])      # flat -> stochastic
    u = [fs.u_at("r", "h", 0, "c", 0, t) for t in range(4)]
    # tokens = what the forced u actually picks (honest miner)
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i]) for i in range(4)]
    n_stoch, n_match = fs.seed_consistency(
        logits, tokens, u, t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    assert n_stoch >= 1
    assert n_match == n_stoch                       # honest -> every stochastic pos matches


def test_seed_consistency_low_when_tokens_ignore_u():
    logits = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.2, 0.15], [0.0, 0.1, 0.2]])
    u = [fs.u_at("r", "h", 0, "c", 0, t) for t in range(3)]
    wrong = [fs.u_at("OTHER", "h", 0, "c", 0, t) for t in range(3)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), wrong[i]) for i in range(3)]
    n_stoch, n_match = fs.seed_consistency(
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
    u = [fs.u_at("rand", "hk", 3, "ckpt", 0, i) for i in range(n)]
    tokens = []
    for i in range(n):
        p = fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i])
        tokens.append(p if i % 3 else (p + 1) % vocab)   # corrupt ~1/3
    kw = dict(t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    got = fs.seed_consistency(logits, tokens, u, **kw)
    ref = _seed_consistency_reference(logits, tokens, u, **kw)
    assert got == ref
    assert ref[0] > 0 and ref[1] < ref[0]            # meaningful: stochastic + mismatches


def test_seed_consistency_empty_batch_returns_zeros():
    assert fs.seed_consistency(
        torch.zeros(0, 5), [], [], t=0.6, top_k=20, top_p=0.95,
        stochastic_threshold=0.99) == (0, 0)


def test_seed_consistency_truncates_to_shortest_input():
    # n = min(len(tokens), len(u), rows); extra logits rows are ignored.
    torch.manual_seed(1)
    logits = torch.randn(6, 32)
    u = [fs.u_at("z", "h", 0, "c", 0, i) for i in range(4)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i])
              for i in range(4)]
    kw = dict(t=0.6, top_k=20, top_p=0.95, stochastic_threshold=0.99)
    assert fs.seed_consistency(logits, tokens, u, **kw) == \
        _seed_consistency_reference(logits, tokens, u, **kw)
