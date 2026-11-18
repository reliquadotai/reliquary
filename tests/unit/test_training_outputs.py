"""Measured training history, complete model files, scoped expiring downloads and final anchors."""

import hashlib
import io
import json

import pytest

from reliquary.shared.ckpt_index import index_key
from reliquary.shared.training_outputs import (
    MAX_BYTES,
    MAX_POINTS,
    SCHEMA,
    TrainingOutputsStore,
    append_metrics,
    checkpoint_entries,
    metrics_key,
)

RUN = "order-train-example"
SHA = "a" * 64


def entry(key, data=b"object"):
    return {"key": key, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}


def index():
    root = f"rl/{RUN}/anchors/00000005/"
    return {
        "format": "reliquary-ckpt-index/1",
        "run": RUN,
        "published_at": 100,
        "latest": {"version": 6, "sha256": SHA},
        "anchors": [
            {
                "version": 5,
                "sha256": SHA,
                "files": [entry(root + "model-00001-of-00001.safetensors")],
                "model_files": [
                    {"name": n, **entry(root + "model/" + n)}
                    for n in (
                        "config.json",
                        "tokenizer.json",
                        "model.safetensors.index.json",
                    )
                ],
            }
        ],
        "deltas": [
            {
                "from": 5,
                "to": 6,
                "target_sha256": SHA,
                **entry(f"rl/{RUN}/deltas/00000005-00000006.rqd"),
            }
        ],
    }


def test_metrics_are_finite_allowlisted_idempotent_and_bounded():
    values = {
        "train/entropy": 0.3,
        "prompt": "private",
        "train/kl": float("nan"),
        "train/env/swe_v1/reward_mean": 0.7,
    }
    previous = {
        "schema": SCHEMA,
        "task_id": RUN,
        "metrics": [
            {"checkpoint": k, "observed_at": 100 + k, "values": values}
            for k in range(MAX_POINTS + 2)
        ],
    }
    history = append_metrics(
        previous, RUN, MAX_POINTS + 2, 100 + MAX_POINTS + 2, values
    )
    assert len(history["metrics"]) == MAX_POINTS and history["metrics_truncated"]
    assert history["metrics"][0]["checkpoint"] == 3
    assert history["metrics"][-1]["values"] == {
        "train/entropy": 0.3,
        "train/env/swe_v1/reward_mean": 0.7,
    }
    assert len(json.dumps(history).encode()) <= MAX_BYTES // 2
    replay = append_metrics(history, RUN, MAX_POINTS + 2, 9000, {"train/entropy": 0.4})
    assert (
        len(replay["metrics"]) == MAX_POINTS
        and replay["metrics"][-1]["values"]["train/entropy"] == 0.4
    )
    with pytest.raises(ValueError):
        append_metrics(history, "another-task", 1, 3, {})


def test_kl_measurement_flag_preserves_uncomputed_zero_and_computed_values():
    for computed in (0.0, 1.0):
        history = append_metrics(None, RUN, 1, 100, {
            "train/kl": 0.0,
            "train/kl_term_computed_ratio": computed,
            "private_note": "not exported",
        })
        assert history["metrics"][0]["values"] == {
            "train/kl": 0.0,
            "train/kl_term_computed_ratio": computed,
        }
    old = append_metrics(None, RUN, 1, 100, {"train/kl": 0.0})
    assert "train/kl_term_computed_ratio" not in old["metrics"][0]["values"]


def test_full_requires_weights_model_index_config_and_tokenizer():
    document = index()
    checkpoints, truncated = checkpoint_entries(document, RUN)
    assert not truncated and [e["format"] for e in checkpoints] == ["full", "delta"]
    assert (
        checkpoints[0]["published_at"] is None
    )  # old indexes have no per-checkpoint clock
    document["anchors"][0]["model_files"].pop()
    assert checkpoint_entries(document, RUN)[0][0]["format"] == "delta"
    document["anchors"][0]["files"][0]["key"] = (
        "rl/another-task/anchors/00000005/weights.safetensors"
    )
    with pytest.raises(ValueError):
        checkpoint_entries(document, RUN)


class Bucket:
    class exceptions:
        class NoSuchKey(Exception):
            pass

    def __init__(self, objects):
        self.objects, self.signed = objects, []

    def get_object(self, *, Key, **kwargs):
        if Key not in self.objects:
            raise self.exceptions.NoSuchKey()
        return {"Body": io.BytesIO(json.dumps(self.objects[Key]).encode())}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn):
        self.signed.append((operation, Params, ExpiresIn))
        return "https://objects.example/signed"


def test_outputs_preserve_clocks_and_only_presign_enumerated_files():
    history = append_metrics(None, RUN, 6, 95, {"train/entropy": 0.3})
    bucket = Bucket({index_key(RUN): index(), metrics_key(RUN): history})
    store = TrainingOutputsStore(bucket, "private")
    outputs = store.outputs(RUN)
    assert outputs["schema"] == SCHEMA and outputs["latest_checkpoint"] == 6
    assert outputs["observed_at"] == 100 and outputs["metrics"][0]["observed_at"] == 95
    result = store.download(RUN, 5, "config.json", now=200)
    assert result["filename"] == "config.json" and result["expires_at"] == 1100
    assert bucket.signed == [
        (
            "get_object",
            {
                "Bucket": "private",
                "Key": f"rl/{RUN}/anchors/00000005/model/config.json",
                "ResponseContentDisposition": 'attachment; filename="config.json"',
            },
            900,
        )
    ]
    for filename in ("../config.json", "spend.json", "not-indexed.bin"):
        with pytest.raises(KeyError):
            store.download(RUN, 5, filename, now=200)
    assert len(bucket.signed) == 1
    with pytest.raises(ValueError):
        checkpoint_entries(index(), "another-task")


def test_no_publication_remains_explicitly_empty():
    output = TrainingOutputsStore(Bucket({}), "private").outputs(RUN)
    assert output["observed_at"] is output["latest_checkpoint"] is None
    assert output["checkpoints"] == output["metrics"] == []


def test_final_anchor_uses_published_snapshot_and_is_idempotent(tmp_path):
    from reliquary.trainer.ckpt_publisher import DeltaPublisher
    from tests.unit.test_ckpt_publication import MemoryStore, _model_dir, chain

    states, store = chain(2), MemoryStore()
    publisher = DeltaPublisher(
        run=RUN,
        store=store,
        staging_dir=tmp_path / "staging",
        model_dir=_model_dir(tmp_path),
        anchor_interval=50,
    )
    publisher.publish(0, states[0])
    publisher.publish(1, states[1])
    expected = json.loads(store.objects[index_key(RUN)])["latest"]
    publisher.finalize()
    document = json.loads(store.objects[index_key(RUN)])
    assert document["latest"] == expected
    latest = checkpoint_entries(document, RUN)[0][-1]
    assert latest["version"] == 1 and latest["format"] == "full"
    paths = {f["name"]: f for f in latest["files"]}
    weights_index = json.loads(
        store.objects[paths["model.safetensors.index.json"]["key"]]
    )
    assert set(weights_index["weight_map"]) == set(states[1])
    before = dict(store.objects)
    publisher.finalize()
    assert store.objects == before


def test_failed_final_index_never_claims_completion_and_can_retry(tmp_path):
    from reliquary.trainer.ckpt_publisher import DeltaPublisher
    from tests.unit.test_ckpt_publication import MemoryStore, _model_dir, chain

    class Store(MemoryStore):
        fail = False

        def put_bytes(self, key, data):
            if self.fail and key == index_key(RUN):
                raise RuntimeError("storage unavailable")
            super().put_bytes(key, data)

    store = Store()
    publisher = DeltaPublisher(
        run=RUN,
        store=store,
        staging_dir=tmp_path / "staging",
        model_dir=_model_dir(tmp_path),
        anchor_interval=50,
    )
    states = chain(2)
    publisher.publish(0, states[0])
    publisher.publish(1, states[1])
    store.fail = True
    with pytest.raises(RuntimeError):
        publisher.finalize()
    assert (
        checkpoint_entries(json.loads(store.objects[index_key(RUN)]), RUN)[0][-1][
            "format"
        ]
        == "delta"
    )
    store.fail = False
    publisher.finalize()
    assert (
        checkpoint_entries(json.loads(store.objects[index_key(RUN)]), RUN)[0][-1][
            "format"
        ]
        == "full"
    )


@pytest.mark.parametrize("version", [True, -1, 0.5, "6"])
def test_corrupt_latest_checkpoint_identity_is_refused(version):
    document = index()
    document["latest"]["version"] = version
    with pytest.raises(ValueError, match="latest checkpoint"):
        checkpoint_entries(document, RUN)


def test_remote_signer_accepts_checkpoint_publication_clocks_without_changing_old_indexes():
    from pydantic import ValidationError

    from reliquary.signer.protocol import RlCheckpointIndex

    document = index()
    RlCheckpointIndex.model_validate(document)
    document["anchors"][0]["published_at"] = 100.5
    document["deltas"][0]["published_at"] = 101
    RlCheckpointIndex.model_validate(document)
    document["anchors"][0]["published_at"] = float("nan")
    with pytest.raises(ValidationError):
        RlCheckpointIndex.model_validate(document)
