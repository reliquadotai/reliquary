import json

from scripts.replay_checkpoint_epoch_market import (
    operator_round_counterfactual,
    summarize,
)


def _candidate(operator: str, prompt: int, value: float, *, selected=False):
    return {
        "operator_id": operator,
        "prompt_idx": prompt,
        "prompt_content_sha256": f"{prompt:064x}",
        "selection_digest": f"{prompt + 100:064x}",
        "value": value,
        "proof_passed": None,
        "selected": selected,
    }


def test_operator_round_counterfactual_keeps_utility_primary():
    candidates = [
        _candidate("a", 0, 2.0),
        _candidate("a", 1, 1.0),
        _candidate("b", 2, 1.0),
    ]

    selected = operator_round_counterfactual(
        candidates,
        window=7,
        environment="math",
        target=3,
    )

    assert selected[0]["prompt_idx"] == 0
    assert {row["operator_id"] for row in selected[1:]} == {"a", "b"}


def test_wrapped_public_archive_replay_reports_reward_concentration(tmp_path):
    candidates = [
        _candidate("a", 0, 1.0, selected=True),
        _candidate("a", 1, 1.0),
        _candidate("b", 2, 1.0, selected=True),
    ]
    archive = {
        "source": "public-r2",
        "data": {
            "window_start": 7,
            "environments": ["math"],
            "window_opened_wall_ts_by_environment": {"math": 100.0},
            "batch": [
                {
                    "env_name": "math",
                    "difficulty_auction_operator_id": "a",
                    "precommit_arrival_ts": 110.0,
                    "reward_vector": "1000",
                    "rollouts": [
                        {"completion_length": 100, "reward": 1.0},
                        {"completion_length": 100, "reward": 0.0},
                    ],
                },
                {
                    "env_name": "math",
                    "difficulty_auction_operator_id": "b",
                    "precommit_arrival_ts": 120.0,
                    "reward_vector": "1100",
                    "rollouts": [
                        {"completion_length": 300, "reward": 1.0},
                        {"completion_length": 300, "reward": 1.0},
                    ],
                },
            ],
            "difficulty_auction": {"math": {"candidates": candidates}},
        },
    }
    path = tmp_path / "7.json"
    path.write_text(json.dumps(archive), encoding="utf-8")

    report = summarize([path], target=2)
    math = report["environments"]["math"]

    assert report["archives"] == 1
    assert report["production_activation_allowed"] is False
    assert math["selected_groups"] == 2
    assert math["selected_completion_tokens"] == 800
    assert math["flat_selected_slot_operator_concentration"]["hhi"] == 0.5
    assert math["gross_completion_token_operator_concentration"]["hhi"] == 0.625
    assert math["selected_precommit_offset_p50_s"] == 15.0
    assert math["positive_reward_count_distribution"] == {1: 1, 2: 1}
