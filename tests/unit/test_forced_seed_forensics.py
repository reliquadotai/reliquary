import json

from reliquary.validator.auth_forensics import record_forced_seed_shadow


def test_forced_seed_forensics_records_boundary_and_reject_diagnostics(tmp_path):
    path = tmp_path / "forced-seed.jsonl"
    per_rollout = [
        {
            "rollout_idx": 0,
            "n_positions": 100,
            "n_stochastic": 90,
            "n_exact_match": 89,
            "n_boundary_match": 99,
            "n_hard_mismatch": 1,
            "n_deterministic_hard_mismatch": 0,
            "max_cdf_miss": 0.2,
        }
    ]

    record_forced_seed_shadow(
        "hk",
        42,
        90,
        89,
        per_rollout=per_rollout,
        n_positions=100,
        n_boundary_match=99,
        n_hard_mismatch=1,
        max_cdf_miss=0.2,
        cdf_boundary_epsilon=0.002,
        ratio_group_would_reject=False,
        ratio_rollout_would_reject=False,
        cdf_would_reject=True,
        cdf_enforced=False,
        path=path,
    )

    record = json.loads(path.read_text().strip())
    assert record["schema_version"] == 2
    assert record["n_hard_mismatch"] == 1
    assert record["cdf_would_reject"] is True
    assert record["cdf_enforced"] is False
    assert record["per_rollout"] == per_rollout
