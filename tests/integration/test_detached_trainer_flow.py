"""End-to-end detached-trainer flow over a dict standing in for R2.

Covers: ordered journal consumption with a tombstone, publish cadence
with the candidate manifest as commit point, crash-replay resume from
the manifest cursor, and the validator-side intake staging the published
checkpoint exactly once. No torch, no network.
"""

import json

from reliquary.infrastructure.training_payload_queue import (
    payload_key,
    tombstone_key,
)
from reliquary.shared.training_payload import (
    encode_tombstone,
    encode_training_payload,
)
from reliquary.trainer.journal import WindowJournal
from reliquary.trainer.worker import TrainerWorker
from reliquary.validator.checkpoint_intake import (
    CANDIDATE_MANIFEST_KEY,
    CheckpointIntake,
)

from tests.unit.test_training_payload_codec import _window_batches

ENV_ORDER = ["openmathinstruct", "opencodeinstruct"]
REV_0 = "0" * 40
REV_1 = f"{1:040x}"


def _seed_store():
    """Windows 101-103 payloads + tombstone at 104."""
    store = {}
    for n in (101, 102, 103):
        store[payload_key(n)] = encode_training_payload(
            _window_batches(), window_start=n, checkpoint_revision=REV_0,
            env_order=ENV_ORDER, window_quarantine={"quarantined": False},
        )
    store[tombstone_key(104)] = encode_tombstone(
        window_start=104, failure_stage="proof_capacity",
        failure_type="ProofCapacityAbort",
    )
    return store


class _StubPublisher:
    """Writes the candidate manifest into the store, like the real one."""

    def __init__(self, store, worker_ref):
        self.store = store
        self.worker_ref = worker_ref
        self.n = 0

    def publish(self, reason):
        self.n += 1
        revision = f"{self.n:040x}"
        self.store[
            f"reliquary/checkpoints/{revision}/model.safetensors"
        ] = b"weights-" + revision.encode()
        self.store[CANDIDATE_MANIFEST_KEY] = json.dumps({
            "checkpoint_n": self.n,
            "repo_id": "org/repo",
            "revision": revision,
            "trained_window_cursor": self.worker_ref()["cursor"],
            "reason": reason,
        }).encode()
        return revision


def _run_until_waited(worker, limit=50):
    outcomes = []
    for _ in range(limit):
        outcome = worker.run_once()
        if outcome == "waited":
            break
        outcomes.append(outcome)
    return outcomes


def _make_worker(store, cursor, *, trained_log, last_revision=None):
    def train_fn(decoded):
        trained_log.append(decoded.window_start)
        return True

    worker_holder = {}
    publisher = _StubPublisher(store, lambda: {
        "cursor": worker_holder["w"].cursor,
    })
    head = {"rev": last_revision}

    def publish_fn(reason):
        rev = publisher.publish(reason)
        head["rev"] = rev
        return rev

    worker = TrainerWorker(
        journal=WindowJournal(fetch_fn=store.get),
        train_fn=train_fn,
        publish_fn=publish_fn,
        head_revision_fn=lambda: head["rev"],
        cursor=cursor,
        stride=1,
        publish_every=2,
        last_published_revision=last_revision,
    )
    worker_holder["w"] = worker
    return worker


def test_full_flow_and_crash_replay(tmp_path):
    store = _seed_store()
    trained = []
    worker = _make_worker(store, 100, trained_log=trained)

    outcomes = _run_until_waited(worker)
    # 101, 102 trained -> publish; 103 trained; 104 tombstone; then wait.
    assert trained == [101, 102, 103]
    assert outcomes.count("published") == 1
    assert worker.tombstones_seen == 1
    assert worker.cursor == 104

    manifest = json.loads(store[CANDIDATE_MANIFEST_KEY])
    assert manifest["revision"] == REV_1
    # Published after window 102: the manifest cursor records it.
    assert manifest["trained_window_cursor"] == 102

    # ---- crash: a fresh worker resumes from the manifest cursor ----
    trained2 = []
    worker2 = _make_worker(
        store, manifest["trained_window_cursor"],
        trained_log=trained2, last_revision=manifest["revision"],
    )
    _run_until_waited(worker2)
    # Replays only the windows after the durable commit point.
    assert trained2 == [103]
    assert worker2.cursor == 104

    # ---- validator intake stages the published checkpoint once ----
    class _R2:
        def get_object(self, Bucket, Key):
            body = store[Key]
            return {"Body": type("B", (), {"read": lambda s: body})()}

        def list_objects_v2(self, Bucket, Prefix):
            return {"Contents": [
                {"Key": k} for k in store if k.startswith(Prefix)
            ]}

        def download_file(self, bucket, key, dest, Config=None):
            assert Config is not None
            with open(dest, "wb") as f:
                f.write(store[key])

    intake = CheckpointIntake(
        r2_client=_R2(), bucket="reliquary", staging_dir=str(tmp_path),
        installed_revision=REV_0,
        validate_fn=lambda p: {"ok": True},
    )
    candidate = intake.poll()
    assert candidate["revision"] == REV_1
    assert intake.stage(candidate) is True
    staged_manifest, staged_dir = intake.take_staged()
    assert staged_manifest["revision"] == REV_1
    assert (staged_dir / "model.safetensors").read_bytes() == (
        b"weights-" + REV_1.encode()
    )
    intake.mark_installed(REV_1, staged_dir)
    # Installed: the same manifest never re-polls.
    assert intake.poll() is None


def test_worker_never_skips_a_missing_window():
    store = _seed_store()
    del store[payload_key(102)]  # hole in the journal
    trained = []
    worker = _make_worker(store, 100, trained_log=trained)
    _run_until_waited(worker)
    # Stops at the hole: 103 exists but must NOT be consumed.
    assert trained == [101]
    assert worker.cursor == 101
