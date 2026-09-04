"""All-token authenticity telemetry for full-support rollouts.

Covers the detector semantics without treating a legal low-probability draw as
proof of tampering.
"""

import os
import subprocess
import sys

import torch

from reliquary.environment.forced_sampling import pick, warp
from reliquary.validator.verifier import ProofResult, evaluate_all_token_auth_shadow


def _proof(chosen, argmax):
    return ProofResult(
        all_passed=True, passed=1, checked=1,
        has_sparse_outputs=True,
        completion_chosen_probs=chosen,
        completion_argmax_probs=argmax,
    )


def test_flags_legal_v5_inverse_cdf_tail_pick_for_telemetry():
    probs = warp(torch.tensor([0.0, -12.0]), t=1.0, top_k=0, top_p=1.0)
    u = float(probs[0]) + float(probs[1]) / 2
    token = pick(probs, u)
    assert token == 1
    assert float(probs[token]) < 1e-5

    proof = _proof([float(probs[token])], [float(probs.max())])
    ok, metrics = evaluate_all_token_auth_shadow(proof)
    assert ok is False
    assert metrics["findings"] == 1


def test_honest_confident_tokens_pass():
    proof = _proof([0.99, 0.98, 0.97, 0.99], [0.99, 0.98, 0.97, 0.99])
    ok, metrics = evaluate_all_token_auth_shadow(proof)
    assert ok is True
    assert metrics["findings"] == 0


def test_low_prob_at_uncertain_position_not_flagged():
    # chosen below threshold but the model was NOT confident (argmax < conf):
    # a genuine decision point, legitimately sampled -> no finding.
    proof = _proof([0.99, 1e-7, 0.99], [0.99, 0.30, 0.99])
    ok, metrics = evaluate_all_token_auth_shadow(proof)
    assert ok is True
    assert metrics["findings"] == 0


def test_forced_span_is_exempt():
    # Same injected-looking token, but the position is inside the exempted
    # forced-termination span -> must not count as a false positive.
    proof = _proof([0.99, 0.99, 1e-7, 0.99], [0.99, 0.99, 0.999, 0.99])
    ok, metrics = evaluate_all_token_auth_shadow(proof, exempt_positions={2})
    assert ok is True
    assert metrics["findings"] == 0


def test_v5_all_token_auth_is_shadow_only():
    env = dict(os.environ)
    env["RELIQUARY_PROTOCOL_PROFILE"] = "qwen3-4b-base-dapo-reasoning-v5"
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from reliquary.constants import ALL_TOKEN_AUTH_ENFORCE; "
            "raise SystemExit(ALL_TOKEN_AUTH_ENFORCE)",
        ],
        check=True,
        env=env,
    )
