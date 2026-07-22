from scripts.report_utility_telemetry import summarize


def _group(
    role: str,
    candidate: str,
    operator: str,
    content: str,
    forensic_role: str | None = None,
):
    return {
        "candidate_id": candidate,
        "operator_pseudonym": operator,
        "role": role,
        "forensic_role": forensic_role,
        "checkpoint_revision": "checkpoint-a",
        "prompt_content_sha256": content,
        "rollouts": [
            {
                "reward": 1.0,
                "completion_length": 10,
                "natural_eos": True,
                "termination_path": "phase1_eos",
                "chosen_nll": {"mean": 0.2},
                "full_policy_entropy": {"mean": 1.2},
                "hidden_delta_f16_b64": "AAA=",
                "representation_shift_l2": 2.0,
                "token_degeneracy": {"repeated_ngram_fraction": 0.0},
            },
            {
                "reward": 0.0,
                "completion_length": 20,
                "natural_eos": False,
                "termination_path": "phase1_cap",
                "chosen_nll": {"mean": 0.4},
                "full_policy_entropy": {"mean": 1.4},
                "hidden_delta_f16_b64": "AAA=",
                "representation_shift_l2": 4.0,
                "token_degeneracy": {"repeated_ngram_fraction": 0.2},
            },
        ],
    }


def test_summary_is_explicitly_non_activating_and_reports_counterfactuals():
    bundles = [
        {
            "schema_version": 1,
            "window": 10,
            "environments": {
                "math": [
                    _group("winner", "a", "op-a", "content-a"),
                    _group(
                        "forensic", "b", "op-b", "content-b",
                        "counterfactual",
                    ),
                ]
            },
        }
    ]

    report = summarize(bundles)
    math = report["environments"]["math"]
    assert report["activation_allowed"] is False
    assert math["decision"] == "INSUFFICIENT_TELEMETRY"
    assert math["winner_groups"] == 1
    assert math["forensic_groups"] == 1
    assert math["counterfactual_groups"] == 1
    assert math["random_watch_groups"] == 0
    assert math["complete_field_rate"] == 1.0
    assert math["positive_reward_count_distribution"] == {1: 2}
    assert math["feature_means"]["completion_length_mean"] == 15.0
    assert math["feature_means"]["representation_shift_l2_mean"] == 3.0


def test_summary_detects_duplicate_canonical_content():
    bundles = [
        {
            "schema_version": 1,
            "window": 10,
            "environments": {
                "code": [
                    _group("winner", "a", "op-a", "same-content"),
                    _group("forensic", "b", "op-b", "same-content"),
                ]
            },
        }
    ]

    assert (
        summarize(bundles)["environments"]["code"][
            "duplicate_content_groups"
        ]
        == 1
    )
