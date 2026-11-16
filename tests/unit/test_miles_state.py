"""Checkpoint snapshots from the Miles gateway's sampler export (reliquary/trainer/miles_engine.MilesState)."""
import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from reliquary.trainer.miles_engine import MilesState, expected_weight_keys

KEYS = {"model.embed_tokens.weight": (8, 4), "model.norm.weight": (4,), "lm_head.weight": (8, 4)}


class Future:
    def __init__(self, value):
        self.value = value

    def result(self):
        return self.value


class Exporter:
    """``save_weights_for_sampler`` writing a committed export dir under the checkpoint root, as the gateway does."""

    def __init__(self, root, keys=KEYS, model_id="run-1"):
        self.root, self.keys, self.model_id, self.names = root, keys, model_id, []

    def save_weights_for_sampler(self, name):
        self.names.append(name)
        committed = self.root / ".committed" / name
        hf = committed / "hf"
        hf.mkdir(parents=True)
        tensors = {k: torch.full(shape, float(len(self.names)), dtype=torch.bfloat16) for k, shape in self.keys.items()}
        items = sorted(tensors)
        save_file({k: tensors[k] for k in items[:1]}, hf / "model-00001-of-00002.safetensors")
        save_file({k: tensors[k] for k in items[1:]}, hf / "model-00002-of-00002.safetensors")
        index = {"weight_map": {k: f"model-0000{1 if i == 0 else 2}-of-00002.safetensors" for i, k in enumerate(items)}}
        (hf / "model.safetensors.index.json").write_text(json.dumps(index))
        link = self.root / self.model_id / "sampler_weights" / name
        link.parent.mkdir(parents=True, exist_ok=True)
        link.symlink_to(committed)
        return Future(SimpleNamespace(path=f"tinker://{self.model_id}/sampler_weights/{name}"))


def test_loads_exported_safetensors(tmp_path):
    exporter = Exporter(tmp_path)
    state = MilesState(exporter, checkpoint_root=tmp_path, expected_keys=set(KEYS), delete=lambda path: None)
    snapshot = state()
    assert set(snapshot) == set(KEYS)
    assert snapshot["lm_head.weight"].dtype == torch.bfloat16 and snapshot["lm_head.weight"].device.type == "cpu"
    assert float(snapshot["model.norm.weight"][0]) == 1.0


def test_export_refuses_key_mismatch(tmp_path):
    exporter = Exporter(tmp_path, keys={k: v for k, v in KEYS.items() if k != "lm_head.weight"})
    state = MilesState(exporter, checkpoint_root=tmp_path, expected_keys=set(KEYS), delete=lambda path: None)
    with pytest.raises(RuntimeError, match="lm_head.weight"):
        state()


def test_old_exports_pruned(tmp_path):
    deleted = []
    state = MilesState(Exporter(tmp_path), checkpoint_root=tmp_path, expected_keys=set(KEYS), delete=deleted.append)
    for _ in range(4):
        state()
    assert deleted == ["tinker://run-1/sampler_weights/snapshot-0", "tinker://run-1/sampler_weights/snapshot-1"]


def test_expected_keys_from_index_or_single_file(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "f", "y": "f"}}))
    assert expected_weight_keys(tmp_path / "a") == {"x", "y"}
    (tmp_path / "b").mkdir()
    save_file({"z": torch.zeros(2)}, tmp_path / "b" / "model.safetensors")
    assert expected_weight_keys(tmp_path / "b") == {"z"}


def test_export_found_under_the_launchers_run_dir(tmp_path):
    run_root = tmp_path / "checkpoints" / "run-20261116"         # the launcher's --tinker-checkpoint-root
    run_root.mkdir(parents=True)
    state = MilesState(Exporter(run_root), checkpoint_root=tmp_path, expected_keys=set(KEYS), delete=lambda p: None)
    assert set(state()) == set(KEYS)
