from dataclasses import FrozenInstanceError
import hashlib
import random
from types import SimpleNamespace

import pytest

import reliquary.validator.batcher as batcher_module
from reliquary.validator.batcher import GrpoWindowBatcher, ValidSubmission
from reliquary.validator.cooldown import CooldownMap
from reliquary.validator.dedup import RolloutHashSet
from reliquary.validator.difficulty_auction import (
    ShadowSubmission,
    select_shadow_auction as real_select_shadow_auction,
)


class MathEnv:
    name = "openmathinstruct"

    def __len__(self):
        return 100

    def get_problem(self, prompt_idx):
        return {"prompt": f"p{prompt_idx}", "ground_truth": "a"}


class CodeEnv(MathEnv):
    name = "opencodeinstruct"


def _batcher(
    env=None,
    *,
    cooldown_map=None,
    hash_set=None,
    operator_by_hotkey=None,
):
    return GrpoWindowBatcher(
        window_start=500,
        env=env or MathEnv(),
        model=SimpleNamespace(),
        completion_text_fn=lambda _rollout: "",
        cooldown_map=cooldown_map,
        hash_set=hash_set,
        verify_commitment_proofs_fn=lambda *_args, **_kwargs: None,
        verify_signature_fn=lambda *_args, **_kwargs: True,
        drand_round_check_enabled=False,
        operator_by_hotkey=operator_by_hotkey,
    )


def _submission(hotkey, prompt_idx, drand_round, k):
    digest = hotkey.encode().ljust(32, b"\x00")[:32]
    return ValidSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        merkle_root_bytes=digest,
        selection_digest_bytes=digest,
        drand_round=drand_round,
        rollouts=[SimpleNamespace(reward=1.0)] * k
        + [SimpleNamespace(reward=0.0)] * (8 - k),
    )


def test_shadow_does_not_change_production_fcfs_selection_or_rewards():
    batcher = _batcher()
    easy = [
        _submission(f"easy{i}", i, drand_round=1, k=6)
        for i in range(8)
    ]
    hard_late = _submission("hard", 99, drand_round=9, k=2)
    batcher._valid = easy + [hard_late]
    batcher.valid_count = len(batcher._valid)

    production_batch, rewards = batcher.seal_batch()

    assert {submission.hotkey for submission in production_batch} == {
        f"easy{i}" for i in range(8)
    }
    assert rewards == {f"easy{i}": pytest.approx(1 / 8) for i in range(8)}
    shadow = batcher.difficulty_auction_shadow
    assert shadow["mode"] == "observation_only"
    assert shadow["production_changed"] is False
    assert shadow["status"] == "computed"
    assert shadow["validated_candidates"] == 9
    assert shadow["computation_ms"] >= 0.0
    assert shadow["shadow_selected_count"] == 8
    assert shadow["selection_overlap_count"] == 7
    hard_row = next(
        row for row in shadow["candidates"] if row["hotkey"] == "hard"
    )
    assert hard_row["shadow_selected"] is True
    assert hard_row["production_selected"] is False
    assert hard_row["production_rewarded"] is False
    easy_rows = [
        row for row in shadow["candidates"] if row["hotkey"].startswith("easy")
    ]
    assert all(row["production_selected"] is True for row in easy_rows)
    assert all(row["production_rewarded"] is True for row in easy_rows)


def test_shadow_failure_cannot_fail_production_seal(monkeypatch):
    batcher = _batcher()
    batcher._valid = [_submission("miner", 1, drand_round=1, k=4)]
    batcher.valid_count = 1

    def _explode(*_args, **_kwargs):
        raise RuntimeError("shadow-only failure")

    monkeypatch.setattr(batcher_module, "select_shadow_auction", _explode)

    production_batch, rewards = batcher.seal_batch()

    assert [submission.hotkey for submission in production_batch] == ["miner"]
    assert rewards == {"miner": pytest.approx(1 / 8)}
    assert batcher.difficulty_auction_shadow["status"] == "error"
    assert batcher.difficulty_auction_shadow["error_type"] == "RuntimeError"


def test_shadow_receives_only_detached_immutable_values(monkeypatch):
    batcher = _batcher()
    live_submission = _submission("miner", 1, drand_round=1, k=4)
    batcher._valid = [live_submission]
    batcher.valid_count = 1

    def _inspect(pool, **kwargs):
        assert isinstance(pool, tuple)
        assert len(pool) == 1
        candidate = pool[0]
        assert isinstance(candidate, ShadowSubmission)
        assert candidate.source_id == id(live_submission)
        assert candidate is not live_submission
        assert candidate.rewards == (1.0,) * 4 + (0.0,) * 4
        with pytest.raises(FrozenInstanceError):
            candidate.prompt_idx = 99
        return real_select_shadow_auction(pool, **kwargs)

    monkeypatch.setattr(batcher_module, "select_shadow_auction", _inspect)

    production_batch, rewards = batcher.seal_batch()

    assert production_batch == [live_submission]
    assert rewards == {"miner": pytest.approx(1 / 8)}
    assert live_submission.prompt_idx == 1


def test_shadow_candidate_limit_skips_work_without_changing_production(
    monkeypatch,
):
    batcher = _batcher()
    batcher._valid = [
        _submission("miner-a", 1, drand_round=1, k=4),
        _submission("miner-b", 2, drand_round=1, k=4),
    ]
    batcher.valid_count = 2
    monkeypatch.setattr(
        batcher_module, "DIFFICULTY_AUCTION_SHADOW_MAX_CANDIDATES", 1
    )

    production_batch, rewards = batcher.seal_batch()

    assert {submission.hotkey for submission in production_batch} == {
        "miner-a",
        "miner-b",
    }
    assert rewards == {
        "miner-a": pytest.approx(1 / 8),
        "miner-b": pytest.approx(1 / 8),
    }
    assert batcher.difficulty_auction_metadata_by_id == {}
    assert batcher.difficulty_auction_shadow == {
        "schema_version": 1,
        "mode": "observation_only",
        "environment": "openmathinstruct",
        "population": "fully_validated_before_current_seal",
        "population_limitations": [
            "excludes_pre_validation_rejects",
            "excludes_batch_filled_rejects",
        ],
        "production_changed": False,
        "delta": 1.0,
        "batch_size": 8,
        "validated_candidates": 2,
        "max_candidates": 1,
        "operator_cap_requested": 2,
        "operator_mapping_snapshot_count": 0,
        "status": "skipped_candidate_limit",
    }


def test_shadow_operator_cap_uses_complete_metagraph_snapshot():
    mapping = {f"miner-{index}": "operator-a" for index in range(4)}
    batcher = _batcher(operator_by_hotkey=mapping)
    batcher._valid = [
        _submission(f"miner-{index}", index, drand_round=1, k=4)
        for index in range(4)
    ]
    batcher.valid_count = len(batcher._valid)

    production_batch, rewards = batcher.seal_batch()

    assert len(production_batch) == 4
    assert len(rewards) == 4
    shadow = batcher.difficulty_auction_shadow
    assert shadow["operator_cap_requested"] == 2
    assert shadow["operator_cap_applied"] is True
    assert shadow["operator_mapping_complete"] is True
    assert shadow["operator_mapping_snapshot_count"] == 4
    assert shadow["shadow_selected_count"] == 2
    assert {row["operator_id"] for row in shadow["candidates"]} == {
        "operator-a"
    }


def test_shadow_operator_cap_disables_itself_on_incomplete_mapping():
    batcher = _batcher(operator_by_hotkey={"miner-a": "operator-a"})
    batcher._valid = [
        _submission("miner-a", 1, drand_round=1, k=4),
        _submission("miner-b", 2, drand_round=1, k=4),
    ]
    batcher.valid_count = len(batcher._valid)

    production_batch, rewards = batcher.seal_batch()

    assert len(production_batch) == 2
    assert len(rewards) == 2
    shadow = batcher.difficulty_auction_shadow
    assert shadow["operator_cap_requested"] == 2
    assert shadow["operator_cap_applied"] is False
    assert shadow["operator_mapping_complete"] is False
    assert shadow["shadow_selected_count"] == 2
    operator_by_hotkey = {
        row["hotkey"]: row["operator_id"] for row in shadow["candidates"]
    }
    assert operator_by_hotkey == {
        "miner-a": "operator-a",
        "miner-b": None,
    }


def test_shadow_enabled_and_disabled_are_production_equivalent_randomized(
    monkeypatch,
):
    def _pool(seed):
        rng = random.Random(seed)
        submissions = []
        for index in range(rng.randint(1, 30)):
            digest = hashlib.sha256(f"{seed}:{index}".encode()).digest()
            successes = rng.randrange(9)
            rollouts = [SimpleNamespace(reward=1.0)] * successes + [
                SimpleNamespace(reward=0.0)
            ] * (8 - successes)
            rollout_hashes = [
                hashlib.sha256(
                    f"{seed}:{index}:rollout:{rollout}".encode()
                ).digest()
                for rollout in range(8)
            ]
            submissions.append(
                ValidSubmission(
                    hotkey=f"hk{rng.randrange(10)}",
                    prompt_idx=rng.randrange(12),
                    merkle_root_bytes=digest,
                    selection_digest_bytes=digest,
                    drand_round=rng.randrange(5),
                    rollouts=rollouts,
                    rollout_hashes=rollout_hashes,
                )
            )
        return submissions

    def _run(seed, *, enabled):
        cooldown = CooldownMap(cooldown_windows=50)
        cooldown.record_batched(0, 499)
        hash_set = RolloutHashSet(retention_windows=300)
        batcher = _batcher(cooldown_map=cooldown, hash_set=hash_set)
        batcher._valid = _pool(seed)
        batcher.valid_count = len(batcher._valid)
        monkeypatch.setattr(
            batcher_module, "DIFFICULTY_AUCTION_SHADOW_ENABLED", enabled
        )

        batch, rewards = batcher.seal_batch()
        metadata = {
            submission.selection_digest.hex(): dict(
                batcher.selection_metadata_by_id[id(submission)]
            )
            for submission in batcher._valid
        }
        return {
            "batch": [submission.selection_digest.hex() for submission in batch],
            "rewards": rewards,
            "selection_metadata": metadata,
            "rewarded_not_selected": dict(
                batcher.rewarded_but_not_selected_by_hotkey
            ),
            "cooldown": cooldown.export_state(),
            "hashes": dict(hash_set._entries),
            "shadow_status": batcher.difficulty_auction_shadow["status"],
        }

    for seed in range(50):
        enabled = _run(seed, enabled=True)
        disabled = _run(seed, enabled=False)

        assert enabled["shadow_status"] == "computed"
        assert disabled["shadow_status"] == "disabled"
        assert {
            key: value
            for key, value in enabled.items()
            if key != "shadow_status"
        } == {
            key: value
            for key, value in disabled.items()
            if key != "shadow_status"
        }


def test_disabled_shadow_never_calls_counterfactual(monkeypatch):
    batcher = _batcher()
    batcher._valid = [_submission("miner", 1, drand_round=1, k=4)]
    batcher.valid_count = 1
    monkeypatch.setattr(
        batcher_module, "DIFFICULTY_AUCTION_SHADOW_ENABLED", False
    )

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("disabled shadow must not call selector")

    monkeypatch.setattr(batcher_module, "select_shadow_auction", _unexpected)

    production_batch, rewards = batcher.seal_batch()

    assert [submission.hotkey for submission in production_batch] == ["miner"]
    assert rewards == {"miner": pytest.approx(1 / 8)}
    assert batcher.difficulty_auction_shadow["status"] == "disabled"


def test_code_environment_is_explicitly_out_of_scope():
    batcher = _batcher(CodeEnv())
    batcher._valid = [_submission("miner", 1, drand_round=1, k=4)]
    batcher.valid_count = 1

    production_batch, _rewards = batcher.seal_batch()

    assert [submission.hotkey for submission in production_batch] == ["miner"]
    assert (
        batcher.difficulty_auction_shadow["status"]
        == "out_of_scope_environment"
    )
