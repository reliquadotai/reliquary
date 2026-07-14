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
            "n_miss_gt_0_01": 1,
            "n_miss_gt_0_05": 1,
            "n_miss_gt_0_10": 1,
            "max_cdf_miss": 0.2,
            "first_hard_mismatch_offset": 17,
            "completion_length": 100,
            "claimed_forced": False,
            "forced": False,
            "validated_force_span_length": 0,
            "termination_path": "phase1_eos",
            "repeated_ngram_fraction": 0.1,
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
        n_miss_gt_0_01=1,
        n_miss_gt_0_05=1,
        n_miss_gt_0_10=1,
        max_cdf_miss=0.2,
        window_start=500,
        env_name="openmathinstruct",
        checkpoint_hash="sha256:test",
        cdf_boundary_epsilon=0.002,
        ratio_group_would_reject=False,
        ratio_rollout_would_reject=False,
        cdf_would_reject=True,
        cdf_enforced=False,
        runtime_profile={
            "profile_hash": "ab" * 32,
            "torch_version": "2.7.0+cu128",
            "fla_version": "0.5.0",
        },
        path=path,
    )

    record = json.loads(path.read_text().strip())
    assert record["schema_version"] == 4
    assert record["window_start"] == 500
    assert record["env_name"] == "openmathinstruct"
    assert record["checkpoint_hash"] == "sha256:test"
    assert record["n_hard_mismatch"] == 1
    assert record["n_miss_gt_0_10"] == 1
    assert record["cdf_would_reject"] is True
    assert record["cdf_enforced"] is False
    assert record["runtime_profile"]["fla_version"] == "0.5.0"
    assert record["per_rollout"] == per_rollout
