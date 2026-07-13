from reliquary.validator.training_accumulator import BalancedTrainingAccumulator


def test_accumulates_sparse_env_across_windows_without_overweighting_fast_env():
    acc = BalancedTrainingAccumulator({"math": 4, "code": 4})

    first = acc.add_window(
        {"math": ["m1"], "code": ["c1", "c2", "c3", "c4"]},
        window_n=10,
        checkpoint_revision="rev-a",
    )
    assert first["added"] == {"math": 1, "code": 4}
    assert acc.ready is False

    second = acc.add_window(
        {"math": ["m2", "m3"], "code": ["c5", "c6", "c7", "c8"]},
        window_n=11,
        checkpoint_revision="rev-a",
    )
    assert second["added"] == {"math": 2, "code": 0}
    assert second["not_accumulated"] == {"math": 0, "code": 4}
    assert acc.snapshot()["counts"] == {"math": 3, "code": 4}

    acc.add_window(
        {"math": ["m4"], "code": ["c9"]},
        window_n=12,
        checkpoint_revision="rev-a",
    )
    assert acc.ready is True
    assert acc.training_batches(["math", "code"]) == [
        ["m1", "m2", "m3", "m4"],
        ["c1", "c2", "c3", "c4"],
    ]
    assert acc.snapshot()["source_windows"] == {
        "math": [10, 11, 12],
        "code": [10],
    }


def test_checkpoint_change_discards_pending_groups_before_adding_new_data():
    acc = BalancedTrainingAccumulator({"math": 2, "code": 2})
    acc.add_window(
        {"math": ["old-m"], "code": ["old-c"]},
        window_n=20,
        checkpoint_revision="rev-a",
    )

    update = acc.add_window(
        {"math": ["new-m"], "code": ["new-c"]},
        window_n=21,
        checkpoint_revision="rev-b",
    )

    assert update["checkpoint_reset"]["counts"] == {"math": 1, "code": 1}
    assert update["snapshot"]["counts"] == {"math": 1, "code": 1}
    assert acc.ready is False
    assert acc.snapshot()["source_windows"] == {"math": [21], "code": [21]}


def test_reset_returns_consumed_snapshot_and_empties_accumulator():
    acc = BalancedTrainingAccumulator({"math": 1, "code": 1})
    acc.add_window(
        {"math": ["m"], "code": ["c"]},
        window_n=30,
        checkpoint_revision="rev-a",
    )

    consumed = acc.reset()

    assert consumed["ready"] is True
    assert consumed["counts"] == {"math": 1, "code": 1}
    assert acc.snapshot()["counts"] == {"math": 0, "code": 0}
    assert acc.checkpoint_revision is None


def test_zero_targets_are_ready_for_mocked_empty_batch_paths():
    acc = BalancedTrainingAccumulator({"fake": 0})
    acc.add_window({}, window_n=1, checkpoint_revision="rev-a")
    assert acc.ready is True
    assert acc.training_batches(["fake"]) == [[]]
