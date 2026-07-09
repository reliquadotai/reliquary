import torch
from reliquary.validator import verifier
from reliquary.environment import forced_sampling as fs


def test_gpu_completion_seed_counts_perfect_for_forced_tokens():
    # 3 completion positions, flat distributions -> all stochastic
    logits = torch.tensor([[0.2, 0.1, 0.0], [0.1, 0.2, 0.15], [0.0, 0.1, 0.2]])
    u = [fs.u_at("r", "h", 0, "c", 0, t) for t in range(3)]
    tokens = [fs.pick(fs.warp(logits[i], t=0.6, top_k=20, top_p=0.95), u[i]) for i in range(3)]
    n_stoch, n_match = verifier._gpu_seed_consistency(logits, tokens, u)
    assert n_stoch == 3 and n_match == 3


def test_proof_result_has_seed_fields():
    p = verifier.ProofResult(all_passed=True, passed=0, checked=0, has_sparse_outputs=False)
    assert p.seed_n_stochastic == 0 and p.seed_n_match == 0
