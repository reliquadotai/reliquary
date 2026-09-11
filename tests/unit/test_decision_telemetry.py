import json
import os
from types import SimpleNamespace

from reliquary.shared import decision_telemetry as dt


def test_observation_is_fail_open_and_receipts_reconcile(monkeypatch, tmp_path):
    monkeypatch.setenv("RELIQUARY_DECISION_TELEMETRY_ENABLED", "1")
    events = []
    monkeypatch.setattr(
        dt, "emit", lambda event, **fields: events.append(dict(event=event, **fields))
    )
    dt.capture("broken_metadata", lambda: 1 / 0)
    assert events[-1] == dict(event="measurement_missing", stage="broken_metadata")

    @dt.observe("call")
    def successful():
        return 17

    assert successful() == 17
    group = SimpleNamespace(prompt_idx=3, rollouts=[])
    decoded = SimpleNamespace(window_start=8, checkpoint_revision="abc")
    dt.journal_received(decoded, 81)
    dt.annotate_origins({"logic": [group]}, decoded)
    dt.optimizer_receipt(
        [(group, [], 1)],
        n_processed=1,
        window=8,
        step_index=2,
        item_builder=lambda _: [(None,) * 5 + ([True, False],)],
    )
    receipt = events[-1]
    assert receipt["reconciled"] and receipt["groups"][0]["trainable_tokens"] == 1
    assert receipt["groups"][0]["journal_key"] == 81
    dt.optimizer_receipt(
        [(group, [], 1)],
        n_processed=2,
        window=8,
        step_index=2,
        item_builder=lambda _: [(None,) * 5 + ([True],)],
    )
    assert events[-1]["reconciled"] is False

    writer = dt.EventWriter(tmp_path, max_bytes=500, keep=2)
    for i in range(8):
        writer.put({"i": i})
        writer.queue.join()
    files = list(tmp_path.glob("*.jsonl"))
    assert 1 <= len(files) <= 2
    assert all(os.stat(p).st_mode & 0o777 == 0o600 for p in files)
    assert all(
        json.loads(line)["sequence"] > 0
        for p in files
        for line in p.read_text().splitlines()
    )
    writer.put({"bad": float("nan")})
    writer.queue.join()
    assert writer.errors == 1
    blocked = tmp_path / "not_a_directory"
    blocked.write_text("x")
    bad_writer = dt.EventWriter(blocked)
    bad_writer.put({"a": 1})
    bad_writer.queue.join()
    assert bad_writer.errors == 1


def test_shadow_uses_only_ready_candidates_and_flags_missing(tmp_path):
    from scripts.decision_report import report

    events = [
        dict(
            event="candidate_eligible",
            window=1,
            environment="logic",
            receipt_id=x,
            ordinal=i,
        )
        for i, x in enumerate(["earliest_not_ready", "second", "third"])
    ]
    events.append(
        dict(
            event="pick_ready_set",
            window=1,
            environment="logic",
            candidates=[dict(receipt_id="second"), dict(receipt_id="third")],
            chosen_receipts=["third"],
        )
    )
    path = tmp_path / "events.jsonl"
    rows = [
        json.dumps(dict(e, run_id="test", sequence=i + 1)) for i, e in enumerate(events)
    ]
    path.write_text("\n".join(rows + [rows[-1], "broken"]))
    result = report([path])
    assert result["comparisons"][0]["eligible_order_receipts"] == ["second"]
    assert result["comparisons"][0]["replaced"] == 1
    assert result["event_counts"]["pick_ready_set"] == 1
    assert result["malformed_lines"] == 1 and not result["complete_observation"]


def test_no_optimizer_receipt_for_empty_training(monkeypatch):
    from reliquary.validator.training import train_step

    events = []
    monkeypatch.setenv("RELIQUARY_DECISION_TELEMETRY_ENABLED", "1")
    monkeypatch.setattr(dt, "emit", lambda event, **fields: events.append(event))
    model = object()
    assert train_step(model, [], ref_model=None) is model
    assert "training_call_returned" in events
    assert "optimizer_success" not in events
