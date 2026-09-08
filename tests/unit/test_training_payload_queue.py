"""Durable local queue feeding the R2 ``reliquary/training/`` prefix."""

import asyncio
import hashlib
import json
import threading

import pytest

from reliquary.infrastructure.training_payload_queue import (
    TrainingPayloadQueue,
    encoded_window_journal_key,
    payload_key,
    step_cursor_key,
    tombstone_key,
)


def _step_cursor_bytes(journal_key: int, *, written_at: float = 1.0) -> bytes:
    return json.dumps(
        {
            "schema_version": 1,
            "journal_key": journal_key,
            "written_at": written_at,
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def test_keys():
    assert payload_key(30100) == "reliquary/training/window-30100.npz"
    assert tombstone_key(30100) == (
        "reliquary/training/window-30100.tombstone.json"
    )


def test_encoded_journal_key_is_the_raw_window_with_the_gate_off(
    monkeypatch,
):
    """v4/v5 must stay byte-for-byte unchanged: one payload per window,
    keyed by the window number itself, regardless of ``batch_index``."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", False)

    assert encoded_window_journal_key(30100) == 30100
    assert encoded_window_journal_key(30100, batch_index=0) == 30100
    # Even a nonzero batch_index is ignored with the gate off -- v4/v5
    # code never passes one, but the encoding must not apply regardless.
    assert encoded_window_journal_key(30100, batch_index=3) == 30100


def test_encoded_journal_key_under_fill_closed(monkeypatch):
    """R11: window_start * FILL_CLOSED_EMISSIONS_PER_WINDOW + batch_index,
    gated on FILL_CLOSED_ENABLED. Pins ``batch_index=3`` at
    ``window_start=42`` per the brief; ``FILL_CLOSED_EMISSIONS_PER_WINDOW``
    itself is monkeypatched to a fixed 16 so this test doesn't depend on
    the runtime PROTOCOL_VERSION (it derives from
    CHECKPOINT_PUBLISH_INTERVAL_WINDOWS, which is profile-dependent --
    see the separate derivation test below)."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue_module, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)

    assert encoded_window_journal_key(42, batch_index=3) == 42 * 16 + 3
    assert encoded_window_journal_key(42, batch_index=0) == 42 * 16
    assert encoded_window_journal_key(43, batch_index=0) == 43 * 16
    # Consecutive integers within one window's range, exactly what the
    # trainer's cursor (stride=1) already walks -- no trainer change.
    keys = [
        encoded_window_journal_key(42, batch_index=i) for i in range(16)
    ]
    assert keys == list(range(42 * 16, 42 * 16 + 16))


def test_fill_closed_emissions_per_window_derives_from_publish_interval():
    """16 for the same reason FILL_CLOSED_TARGET_GROUPS_PER_ENV is 256:
    one B_BATCH-per-environment emission per optimizer step, and the
    target is CHECKPOINT_PUBLISH_INTERVAL_WINDOWS steps worth of groups."""
    from reliquary.constants import (
        CHECKPOINT_PUBLISH_INTERVAL_WINDOWS,
        FILL_CLOSED_EMISSIONS_PER_WINDOW,
    )

    assert (
        FILL_CLOSED_EMISSIONS_PER_WINDOW == CHECKPOINT_PUBLISH_INTERVAL_WINDOWS
    )


def test_encoded_journal_key_rejects_out_of_range_batch_index(monkeypatch):
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)

    with pytest.raises(ValueError):
        encoded_window_journal_key(42, batch_index=16)
    with pytest.raises(ValueError):
        encoded_window_journal_key(42, batch_index=-1)


def test_enqueue_writes_atomically(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")
    q.enqueue_tombstone(30101, b"{}")
    assert (tmp_path / "window-30100.npz").read_bytes() == b"abc"
    assert (tmp_path / "window-30101.tombstone.json").read_bytes() == b"{}"
    assert not list(tmp_path.glob("*.tmp"))


def test_worker_uploads_and_deletes(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")
    q.enqueue_tombstone(30101, b"{}")
    uploaded = {}

    def fake_upload(key, data):
        uploaded[key] = data

    asyncio.run(q.drain_once(upload_fn=fake_upload))
    assert uploaded == {
        "reliquary/training/window-30100.npz": b"abc",
        "reliquary/training/window-30101.tombstone.json": b"{}",
    }
    assert not list(tmp_path.glob("window-*"))
    assert q.snapshot()["uploads_succeeded_total"] == 2


def test_failed_upload_keeps_file_and_backs_off(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")

    def bad_upload(key, data):
        raise RuntimeError("r2 down")

    asyncio.run(q.drain_once(upload_fn=bad_upload))
    assert (tmp_path / "window-30100.npz").exists()
    snap = q.snapshot()
    assert snap["upload_failures_total"] == 1
    assert snap["depth"] == 1


def test_restart_rescan_picks_up_pending(tmp_path):
    q1 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q1.enqueue_payload(30100, b"abc")
    # Fresh instance over the same dir (process restart).
    q2 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    uploaded = {}
    asyncio.run(q2.drain_once(upload_fn=lambda k, d: uploaded.update({k: d})))
    assert "reliquary/training/window-30100.npz" in uploaded


def test_committed_entry_retry_is_idempotent_and_conflict_is_rejected(
    tmp_path,
):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    first = q.enqueue_committed_payload(30100, b"abc")
    second = q.enqueue_committed_payload(30100, b"abc")

    assert first == second
    assert first.read_bytes() == b"abc"
    receipt = json.loads(
        (tmp_path / "journal_commits" / "window-30100.json").read_text()
    )
    assert receipt["kind"] == "payload"
    assert receipt["size"] == 3

    with pytest.raises(RuntimeError, match="different commit"):
        q.enqueue_committed_payload(30100, b"different")
    with pytest.raises(RuntimeError, match="different commit"):
        q.enqueue_committed_tombstone(30100, b"{}")


def test_committed_receipt_survives_upload_and_restart(tmp_path):
    q1 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q1.enqueue_committed_payload(30100, b"abc")
    asyncio.run(q1.drain_once(upload_fn=lambda key, data: None))
    assert not (tmp_path / "window-30100.npz").exists()

    # A fresh process sees the retained digest receipt.  The identical call
    # is already committed (and therefore does not recreate an uploaded queue
    # artifact); an equivocation remains rejected.
    q2 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q2.enqueue_committed_payload(30100, b"abc")
    assert not (tmp_path / "window-30100.npz").exists()
    with pytest.raises(RuntimeError, match="different commit"):
        q2.enqueue_committed_payload(30100, b"changed")


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (
            b'{"schema_version":1,"schema_version":1,"journal_key":30100,'
            b'"kind":"payload","sha256":"'
            + hashlib.sha256(b"abc").hexdigest().encode()
            + b'","size":3}',
            "unreadable",
        ),
        (
            b'{"schema_version":NaN,"journal_key":30100,"kind":"payload",'
            b'"sha256":"'
            + hashlib.sha256(b"abc").hexdigest().encode()
            + b'","size":3}',
            "unreadable",
        ),
        (
            b'{"schema_version":true,"journal_key":30100,"kind":"payload",'
            b'"sha256":"'
            + hashlib.sha256(b"abc").hexdigest().encode()
            + b'","size":3}',
            "invalid schema",
        ),
    ],
)
def test_committed_receipt_rejects_ambiguous_durable_json(
    tmp_path,
    body,
    message,
):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    artifact = q.enqueue_committed_payload(30100, b"abc")
    receipt = tmp_path / "journal_commits" / "window-30100.json"
    receipt.write_bytes(body)

    with pytest.raises(RuntimeError, match=message):
        q.enqueue_committed_payload(30100, b"abc")

    assert artifact.read_bytes() == b"abc"


@pytest.mark.parametrize("journal_key", [True, 7.0, "7", -1])
def test_committed_journal_key_is_not_coerced(tmp_path, journal_key):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))

    with pytest.raises(ValueError, match="non-negative integer"):
        q.enqueue_committed_payload(journal_key, b"abc")

    assert not list(tmp_path.glob("window-*"))
    assert not list((tmp_path / "journal_commits").glob("window-*"))


def test_committed_entry_repairs_an_unreceipted_hidden_body(
    tmp_path, monkeypatch,
):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    original = q._enqueue_durable
    calls = 0

    def fail_receipt(filename, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("receipt fsync failed")
        return original(filename, data)

    monkeypatch.setattr(q, "_enqueue_durable", fail_receipt)
    with pytest.raises(OSError, match="receipt fsync failed"):
        q.enqueue_committed_payload(30100, b"abc")

    assert not (tmp_path / "window-30100.npz").exists()
    assert not (
        tmp_path / "journal_commits" / "window-30100.json"
    ).exists()
    staged = tmp_path / "journal_commits" / "window-30100.payload.body"
    assert staged.read_bytes() == b"abc"
    with pytest.raises(RuntimeError, match="different staged bytes"):
        q.enqueue_committed_payload(30100, b"changed")

    monkeypatch.setattr(q, "_enqueue_durable", original)
    q.enqueue_committed_payload(30100, b"abc")
    assert (tmp_path / "window-30100.npz").read_bytes() == b"abc"
    assert not staged.exists()


def test_restart_finishes_receipt_backed_hidden_commit(tmp_path, monkeypatch):
    import reliquary.infrastructure.training_payload_queue as queue_module

    q1 = TrainingPayloadQueue(queue_dir=str(tmp_path))

    def fail_visible_rename(staging_path, final_path):
        raise OSError("visible rename failed")

    monkeypatch.setattr(
        q1, "_publish_staged_journal_entry", fail_visible_rename,
    )
    with pytest.raises(OSError, match="visible rename failed"):
        q1.enqueue_committed_payload(30100, b"abc")

    assert (tmp_path / "journal_commits" / "window-30100.json").exists()
    assert (
        tmp_path / "journal_commits" / "window-30100.payload.body"
    ).exists()
    assert not (tmp_path / "window-30100.npz").exists()

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    TrainingPayloadQueue(queue_dir=str(tmp_path))
    assert (tmp_path / "window-30100.npz").read_bytes() == b"abc"
    assert not (
        tmp_path / "journal_commits" / "window-30100.payload.body"
    ).exists()


def test_restart_fails_closed_on_unreceipted_hidden_body(
    tmp_path, monkeypatch,
):
    import reliquary.infrastructure.training_payload_queue as queue_module

    q1 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    original = q1._enqueue_durable
    calls = 0

    def fail_receipt(filename, data):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("receipt fsync failed")
        return original(filename, data)

    monkeypatch.setattr(q1, "_enqueue_durable", fail_receipt)
    with pytest.raises(OSError, match="receipt fsync failed"):
        q1.enqueue_committed_payload(30100, b"abc")

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    with pytest.raises(RuntimeError, match="requires recovery"):
        TrainingPayloadQueue(queue_dir=str(tmp_path))


def test_restart_fails_closed_on_conflicting_artifact_kind(
    tmp_path, monkeypatch,
):
    import reliquary.infrastructure.training_payload_queue as queue_module

    q1 = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q1.enqueue_committed_payload(30100, b"abc")
    (tmp_path / "window-30100.tombstone.json").write_bytes(b"{}")

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    with pytest.raises(RuntimeError, match="conflicting tombstone"):
        TrainingPayloadQueue(queue_dir=str(tmp_path))


# ---------------- v6.1: trainer-paced picks step cursor ----------------


def test_step_cursor_absent_reads_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    assert q.read_step_cursor() is None


def test_step_cursor_round_trips(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(30142)
    assert q.read_step_cursor() == 30142


def test_step_cursor_is_overwritten_not_appended(tmp_path):
    """Never a growing log: each write replaces the single object."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(30142)
    q.write_step_cursor(30143)
    assert q.read_step_cursor() == 30143
    assert not list(tmp_path.glob("*.tmp"))
    assert len(list(tmp_path.glob("step-cursor.json*"))) == 1


def test_step_cursor_corrupt_json_reads_as_stale_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    (tmp_path / "step-cursor.json").write_bytes(b"{not valid json")
    assert q.read_step_cursor() is None


def test_step_cursor_missing_field_reads_as_stale_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    (tmp_path / "step-cursor.json").write_bytes(b"{}")
    assert q.read_step_cursor() is None


@pytest.mark.parametrize(
    "body",
    [
        b'{"schema_version":1,"journal_key":7,"journal_key":8,"written_at":1}',
        b'{"schema_version":1,"journal_key":7,"written_at":NaN}',
        b'{"schema_version":1,"journal_key":7,"written_at":1e999}',
        b'{"schema_version":true,"journal_key":7,"written_at":1}',
        b'{"schema_version":1,"journal_key":true,"written_at":1}',
        b'{"schema_version":1,"journal_key":7.0,"written_at":1}',
        b'{"schema_version":1,"journal_key":"7","written_at":1}',
        b'{"schema_version":1,"journal_key":-1,"written_at":1}',
    ],
)
def test_step_cursor_rejects_ambiguous_or_coerced_values(tmp_path, body):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    (tmp_path / "step-cursor.json").write_bytes(body)

    assert q.read_step_cursor() is None


@pytest.mark.parametrize("journal_key", [True, 7.0, "7", -1])
def test_step_cursor_invalid_write_is_atomic(tmp_path, journal_key):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(6)
    path = tmp_path / "step-cursor.json"
    original = path.read_bytes()

    with pytest.raises(ValueError, match="non-negative integer"):
        q.write_step_cursor(journal_key)

    assert path.read_bytes() == original
    assert q.read_step_cursor() == 6


def test_step_cursor_key_naming():
    assert step_cursor_key() == "reliquary/training/step-cursor.json"


def test_step_cursor_uploads_via_same_transport_and_stays_local(tmp_path):
    """The cursor rides the same drain/upload transport as payloads, but
    (unlike a payload) it is never deleted after upload -- it is a single
    overwritten object, not a consumed-once queue entry."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(30142)
    uploaded = {}
    asyncio.run(q.drain_once(upload_fn=lambda k, d: uploaded.update({k: d})))
    assert set(uploaded) == {"reliquary/training/step-cursor.json"}
    assert json.loads(uploaded[step_cursor_key()])["journal_key"] == 30142
    assert (tmp_path / "step-cursor.json").exists()


# ---------------- R38/R40: fetch_step_cursor (remote read) ------------
# R40 #1: fetch_step_cursor is fire-and-collect -- a call NEVER blocks on
# network I/O. It returns the last COMPLETED value immediately and, when
# that value is stale (>= the TTL) and no fetch is already in flight,
# kicks a background thread to refresh it. Tests that need to observe a
# freshly-fetched value join the kicked thread
# (``q._step_cursor_fetch_thread``) before reading the cache again.


def test_fetch_step_cursor_round_trips_through_a_shared_remote_store(
    tmp_path,
):
    """Simulates the real split: the trainer's drain uploads the cursor
    from ITS local queue instance, the validator reads it back through a
    DIFFERENT queue instance pointed at a different local directory --
    read_step_cursor (local file) cannot see this at all. No real R2 in
    unit tests, so both sides go through the same injectable fetch/upload
    seam the drain's own tests already use (an in-memory dict standing in
    for the bucket)."""
    remote_store: dict[str, bytes] = {}

    writer = TrainingPayloadQueue(queue_dir=str(tmp_path / "trainer"))
    writer.write_step_cursor(30142)
    asyncio.run(
        writer.drain_once(upload_fn=lambda k, d: remote_store.update({k: d}))
    )

    reader = TrainingPayloadQueue(queue_dir=str(tmp_path / "validator"))
    assert reader.read_step_cursor() is None  # nothing local to this side
    # First call: cache is cold, returns None immediately, kicks a fetch.
    assert reader.fetch_step_cursor(
        fetch_fn=lambda: remote_store.get(step_cursor_key())
    ) is None
    reader._step_cursor_fetch_thread.join(timeout=2)
    # Cache now warm -- a later call sees it without another GET.
    assert reader.fetch_step_cursor() == 30142


def test_fetch_step_cursor_absent_key_stays_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.fetch_step_cursor(fetch_fn=lambda: None)
    q._step_cursor_fetch_thread.join(timeout=2)
    assert q.fetch_step_cursor() is None


def test_fetch_step_cursor_corrupt_body_stays_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.fetch_step_cursor(fetch_fn=lambda: b"{not valid json")
    q._step_cursor_fetch_thread.join(timeout=2)
    assert q.fetch_step_cursor() is None


def test_fetch_step_cursor_missing_field_stays_none(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.fetch_step_cursor(fetch_fn=lambda: b"{}")
    q._step_cursor_fetch_thread.join(timeout=2)
    assert q.fetch_step_cursor() is None


def test_fetch_step_cursor_network_error_never_raises_and_stays_none(
    tmp_path,
):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))

    def timed_out():
        raise TimeoutError("connect timed out")

    # Must not raise even though the exception happens on the background
    # thread the very first call kicks.
    assert q.fetch_step_cursor(fetch_fn=timed_out) is None
    q._step_cursor_fetch_thread.join(timeout=2)
    assert q.fetch_step_cursor() is None


def test_fetch_step_cursor_keeps_the_last_good_value_on_a_later_failure(
    tmp_path,
):
    """A transient R2 hiccup after a prior successful fetch must not
    regress a known-good cursor back to None -- that would needlessly
    stall the pick gate through exactly the brief outage this cache
    exists to smooth over."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.fetch_step_cursor(fetch_fn=lambda: _step_cursor_bytes(9))
    q._step_cursor_fetch_thread.join(timeout=2)
    assert q.fetch_step_cursor() == 9

    def boom():
        raise RuntimeError("r2 down")

    q._step_cursor_cache_at = 0.0  # force staleness so the next call fetches
    q.fetch_step_cursor(fetch_fn=boom)
    q._step_cursor_fetch_thread.join(timeout=2)
    assert q.fetch_step_cursor() == 9


def test_fetch_step_cursor_default_fetch_fn_is_used_when_none_given(
    tmp_path, monkeypatch,
):
    """Wires to the production GetObject path by default -- monkeypatch
    the module-level default rather than inject, to prove the wiring
    itself (not just the seam) is correct."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(
        queue_module, "_default_fetch_step_cursor",
        lambda: _step_cursor_bytes(55),
    )
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.fetch_step_cursor()
    q._step_cursor_fetch_thread.join(timeout=2)
    assert q.fetch_step_cursor() == 55


def test_fetch_step_cursor_value_cache_limits_gets_to_one_per_window(
    tmp_path,
):
    """R40 #1c: at most one GET per TTL window regardless of poll rate --
    the throttle moved into the queue, off the caller."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    calls = []

    def counting_fetch():
        calls.append(1)
        return _step_cursor_bytes(42)

    q.fetch_step_cursor(fetch_fn=counting_fetch)
    q._step_cursor_fetch_thread.join(timeout=2)
    assert len(calls) == 1

    # Two more calls inside the TTL window: no new GET, even with a
    # fetch_fn passed each time -- the cache decides, not the caller.
    q.fetch_step_cursor(fetch_fn=counting_fetch)
    assert q.fetch_step_cursor(fetch_fn=counting_fetch) == 42
    assert len(calls) == 1


def test_fetch_step_cursor_never_runs_two_fetches_concurrently(tmp_path):
    """The in-flight guard, not just the TTL: two calls made before the
    first fetch has finished must still only ever run one GET."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    gate = threading.Event()
    calls = []

    def slow_fetch():
        calls.append(1)
        gate.wait(timeout=2)
        return _step_cursor_bytes(7)

    q.fetch_step_cursor(fetch_fn=slow_fetch)
    # Called again immediately, while the first fetch is still blocked on
    # the gate -- must not kick a second thread.
    q.fetch_step_cursor(fetch_fn=slow_fetch)
    gate.set()
    q._step_cursor_fetch_thread.join(timeout=2)
    assert len(calls) == 1


class _FakeStreamingBody:
    def __init__(self, body: bytes):
        self._body = body

    def read(self) -> bytes:
        return self._body


def test_default_fetch_step_cursor_uses_a_short_bounded_non_retrying_client(
    monkeypatch,
):
    """The drain's PUT/DELETE tolerate 15s connect / 30s read with up to
    3 retries because they run off the hot path in a background drain.
    This call sits on the validator's pick-gate poll cadence and must
    fail fast instead -- pins the actual production Config, not just the
    injectable seam."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "_STEP_CURSOR_CLIENT", None)
    captured = {}

    def fake_boto3_client(service, **kwargs):
        captured["config"] = kwargs["config"]
        captured["endpoint_url"] = kwargs["endpoint_url"]

        class _Client:
            def get_object(self, **_kwargs):
                captured["get_object_kwargs"] = _kwargs
                return {"Body": _FakeStreamingBody(b'{"journal_key": 7}')}

        return _Client()

    monkeypatch.setattr("boto3.client", fake_boto3_client)
    result = queue_module._default_fetch_step_cursor()
    assert result == b'{"journal_key": 7}'
    cfg = captured["config"]
    assert cfg.connect_timeout <= 3
    assert cfg.read_timeout <= 3
    assert cfg.retries["max_attempts"] <= 1
    assert (
        captured["get_object_kwargs"]["Key"]
        == "reliquary/training/step-cursor.json"
    )


def test_default_fetch_step_cursor_swallows_get_object_failure(monkeypatch):
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "_STEP_CURSOR_CLIENT", None)

    def fake_boto3_client(service, **kwargs):
        class _Client:
            def get_object(self, **_kwargs):
                raise RuntimeError("r2 down")

        return _Client()

    monkeypatch.setattr("boto3.client", fake_boto3_client)
    assert queue_module._default_fetch_step_cursor() is None


def test_default_fetch_step_cursor_reuses_one_memoised_client(monkeypatch):
    """R40 #1a: constructing a boto3 client costs ~50-200ms; a fresh
    client per call used to pay that on every poll tick. One
    construction, reused."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "_STEP_CURSOR_CLIENT", None)
    constructions = []

    def fake_boto3_client(service, **kwargs):
        constructions.append(1)

        class _Client:
            def get_object(self, **_kwargs):
                return {"Body": _FakeStreamingBody(b'{"journal_key": 1}')}

        return _Client()

    monkeypatch.setattr("boto3.client", fake_boto3_client)
    queue_module._default_fetch_step_cursor()
    queue_module._default_fetch_step_cursor()
    assert len(constructions) == 1


def test_default_delete_and_default_fetch_step_cursor_share_client_construction(
    monkeypatch,
):
    """R38: 'reuse its client construction, do not build a second config
    path' -- both go through the same env-var-resolving _r2_client
    helper rather than each re-deriving account/endpoint/credentials."""
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "_STEP_CURSOR_CLIENT", None)
    endpoints_seen = []

    def fake_boto3_client(service, **kwargs):
        endpoints_seen.append(kwargs["endpoint_url"])

        class _Client:
            def delete_object(self, **_kwargs):
                pass

            def get_object(self, **_kwargs):
                return {"Body": _FakeStreamingBody(b"{}")}

        return _Client()

    monkeypatch.setattr("boto3.client", fake_boto3_client)
    monkeypatch.setenv("R2_ACCOUNT_ID", "acct123")
    monkeypatch.delenv("R2_ENDPOINT_URL", raising=False)

    queue_module._default_delete("reliquary/training/window-1.tombstone.json")
    queue_module._default_fetch_step_cursor()

    assert len(endpoints_seen) == 2
    assert endpoints_seen[0] == endpoints_seen[1]
    assert "acct123" in endpoints_seen[0]


# ---------------- R40 #4: drain skips an unchanged cursor upload -------


def test_step_cursor_drain_skips_upload_when_body_is_unchanged(tmp_path):
    """An idle trainer (between real steps) must not PUT an identical
    step-cursor object every ~2s forever -- this repo has hit a real R2
    Class A billing incident from exactly this shape of waste before."""
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.write_step_cursor(30142)
    uploads = []
    asyncio.run(q.drain_once(upload_fn=lambda k, d: uploads.append(d)))
    assert len(uploads) == 1

    # Second drain cycle, nothing rewrote the local file -- same bytes,
    # must not upload again.
    asyncio.run(q.drain_once(upload_fn=lambda k, d: uploads.append(d)))
    assert len(uploads) == 1

    # A real step writes a NEW cursor -- must upload again promptly.
    q.write_step_cursor(30143)
    asyncio.run(q.drain_once(upload_fn=lambda k, d: uploads.append(d)))
    assert len(uploads) == 2
