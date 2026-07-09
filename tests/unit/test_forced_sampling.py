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
