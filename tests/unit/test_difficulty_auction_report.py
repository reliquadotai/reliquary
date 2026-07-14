import importlib.util
import sys
from pathlib import Path

import pytest

from reliquary.validator.difficulty_auction import difficulty_score


SCRIPT = Path(__file__).parents[2] / "scripts" / "report_difficulty_auction.py"
SPEC = importlib.util.spec_from_file_location("report_difficulty_auction", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
report_module = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report_module
SPEC.loader.exec_module(report_module)


def _entry(hotkey, prompt_idx, rewards, *, selected=False, drand_round=1):
    return {
        "hotkey": hotkey,
        "prompt_idx": prompt_idx,
        "env_name": "openmathinstruct",
        "submitted_drand_round": drand_round,
        "selection_digest": (hotkey.encode().hex() + "00" * 32)[:64],
        "reward_vector": "".join(str(int(reward)) for reward in rewards),
        "selected_for_batch": selected,
        "rewarded": selected,
    }


def _with_arrival(entry, *, exact=None, proxy=None):
    entry = dict(entry)
    entry["arrival_age_seconds"] = exact
    entry["response_time"] = proxy
    return entry


def test_report_score_matches_validator_score():
    candidate = report_module._candidate_from_entry(
        _entry("hk", 1, [1, 1, 0, 0, 0, 0, 0, 0])
    )
    assert candidate is not None

    observed = report_module.difficulty_value(candidate, 1.0)
    expected = difficulty_score([1, 1, 0, 0, 0, 0, 0, 0], delta=1.0).value

    assert observed == pytest.approx(expected)


def test_replay_reports_coverage_and_difficulty_shift():
    archive = {
        "window_start": 10,
        "environment": "openmathinstruct",
        "batch": [
            _entry("easy", 1, [1, 1, 1, 1, 1, 1, 0, 0], selected=True),
        ],
        "runners_up": [
            _entry(
                "hard",
                2,
                [1, 1, 0, 0, 0, 0, 0, 0],
                selected=False,
                drand_round=9,
            )
        ],
        "reject_summary": {"batch_filled": 5},
    }

    report = report_module.replay_archives(
        [archive],
        environment="openmathinstruct",
        deltas=(1.0,),
        batch_size=1,
    )

    assert report["candidate_count"] == 2
    assert report["observed_batch_filled_reject_count"] == 5
    assert report["production"]["mean_reward"] == pytest.approx(0.75)
    shadow = report["counterfactual_by_delta"]["1"]
    assert shadow["mean_reward"] == pytest.approx(0.25)
    assert shadow["mean_selection_jaccard"] == 0.0


def test_historical_runner_without_reward_data_is_counted_not_invented():
    archive = {
        "window_start": 10,
        "environment": "openmathinstruct",
        "batch": [],
        "runners_up": [
            {
                "hotkey": "runner",
                "prompt_idx": 7,
                "env_name": "openmathinstruct",
            }
        ],
    }

    report = report_module.replay_archives(
        [archive],
        environment="openmathinstruct",
        deltas=(1.0,),
        batch_size=8,
    )

    assert report["candidate_count"] == 0
    assert report["candidate_entries_missing_reward_data"] == 1


def test_deadline_report_prefers_exact_http_arrival_over_acceptance_proxy():
    archive = {
        "window_start": 10,
        "environment": "openmathinstruct",
        "batch": [
            _with_arrival(
                _entry("exact", 1, [1, 1, 0, 0, 0, 0, 0, 0]),
                exact=100.0,
                proxy=400.0,
            ),
            _with_arrival(
                _entry("proxy", 2, [1, 1, 0, 0, 0, 0, 0, 0]),
                proxy=150.0,
            ),
        ],
    }

    report = report_module.replay_archives(
        [archive],
        environment="openmathinstruct",
        deltas=(1.0,),
        batch_size=2,
        deadlines=(120.0, 180.0),
    )

    assert report["arrival_timing_coverage"] == {
        "http_arrival": 1,
        "acceptance_proxy": 1,
        "missing": 0,
        "warning": (
            "historical response_time includes validator processing and "
            "is only an upper-bound proxy for HTTP arrival"
        ),
    }
    assert report["deadline_counterfactual"]["120"][
        "mean_distinct_prompts_by_deadline"
    ] == 1.0
    assert report["deadline_counterfactual"]["180"][
        "fraction_with_at_least_batch_size_distinct"
    ] == 1.0


def test_operator_cap_requires_complete_mapping_for_eligible_population():
    archive = {
        "window_start": 10,
        "environment": "openmathinstruct",
        "batch": [
            _entry(
                "hk1", 1, [1, 1, 0, 0, 0, 0, 0, 0], selected=True
            ),
            _entry(
                "hk2", 2, [1, 1, 0, 0, 0, 0, 0, 0], selected=True
            ),
        ],
    }

    incomplete = report_module.replay_archives(
        [archive],
        environment="openmathinstruct",
        deltas=(1.0,),
        batch_size=2,
        operator_of={"hk1": "owner"},
        max_slots_per_operator=1,
    )
    complete = report_module.replay_archives(
        [archive],
        environment="openmathinstruct",
        deltas=(1.0,),
        batch_size=2,
        operator_of={"hk1": "owner", "hk2": "owner"},
        max_slots_per_operator=1,
    )

    incomplete_shadow = incomplete["counterfactual_by_delta"]["1"]
    assert incomplete["operator_mapping"]["complete_for_cap"] is False
    assert incomplete["production"]["operator_concentration"]["distinct"] == 1
    assert incomplete_shadow["operator_cap_applied"] is False
    assert incomplete_shadow["operator_cap_applied_windows"] == 0
    assert incomplete_shadow["operator_cap_skipped_windows"] == 1
    assert incomplete_shadow["selected_count"] == 2

    complete_shadow = complete["counterfactual_by_delta"]["1"]
    assert complete["operator_mapping"]["complete_for_cap"] is True
    assert complete["production"]["operator_concentration"]["distinct"] == 1
    assert complete_shadow["operator_cap_applied"] is True
    assert complete_shadow["operator_cap_applied_windows"] == 1
    assert complete_shadow["operator_cap_skipped_windows"] == 0
    assert complete_shadow["selected_count"] == 1
