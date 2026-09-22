"""Where a streamed replica gets one decoder layer's weights.

A source hands out layer *i* in the layout the module expects, and nothing else. Keeping it apart
from the forward is what lets the same traversal run off host memory on a small model and off the
checkpoint on one too large to hold: only the source changes.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

import torch

LAYER_PREFIX = "model.layers."


class LayerSource:
    """Yields the state dict of one decoder layer, in module layout."""

    n_layers: int

    def state(self, index: int) -> Mapping[str, torch.Tensor]:  # pragma: no cover - interface
        raise NotImplementedError

    def close(self) -> None:
        """Release anything the source holds. Safe to call twice."""


class HostLayers(LayerSource):
    """The layers of a model already loaded on the host.

    Used when the model fits in host memory: the traversal still keeps one layer on the device,
    which is what bounds device memory, but pays no disk read.
    """

    def __init__(self, model: Any) -> None:
        self._layers = model.model.layers
        self.n_layers = len(self._layers)

    def state(self, index: int) -> Mapping[str, torch.Tensor]:
        return self._layers[index].state_dict()


class CheckpointLayers(LayerSource):
    """The layers of a checkpoint on disk, read one at a time.

    Experts are stored one per tensor and the module wants them stacked, so a layer is fused on
    the way out. Fusing dominates a traversal — measured at four fifths of it — which is why a
    pre-fused copy written once at intake is worth its disk space; this reads that copy when it is
    there and falls back to fusing when it is not.
    """

    def __init__(self, path: str | Path, fused_dir: str | Path | None = None) -> None:
        self._path = Path(path)
        self._fused = Path(fused_dir) if fused_dir is not None else None
        self._reader = _ShardedTensors(self._path)
        config = json.loads((self._path / "config.json").read_text())
        text = config.get("text_config", config)
        self._config = text
        self.n_layers = int(text["num_hidden_layers"])
        self._names = {
            index: [n for n in self._reader.names() if n.startswith(f"{LAYER_PREFIX}{index}.")]
            for index in range(self.n_layers)
        }

    def fused_file(self, index: int) -> Path | None:
        if self._fused is None:
            return None
        candidate = self._fused / f"layer{index:03d}.safetensors"
        return candidate if candidate.exists() else None

    def state(self, index: int) -> Mapping[str, torch.Tensor]:
        fused = self.fused_file(index)
        if fused is not None:
            from safetensors.torch import load_file

            return load_file(str(fused))
        prefix = f"{LAYER_PREFIX}{index}."
        raw = self._reader.read(self._names[index])
        return fuse_layer_state({k[len(prefix):]: v for k, v in raw.items()})

    def close(self) -> None:
        self._reader = None


class _ShardedTensors:
    """Named tensors out of a safetensors checkpoint, single file or sharded."""

    def __init__(self, path: Path) -> None:
        from safetensors import safe_open

        index = path / "model.safetensors.index.json"
        single = path / "model.safetensors"
        if index.exists():
            weight_map = json.loads(index.read_text())["weight_map"]
            self._files = {name: path / shard for name, shard in weight_map.items()}
        elif single.exists():
            with safe_open(str(single), "pt") as handle:
                self._files = {name: single for name in handle.keys()}
        else:
            raise FileNotFoundError(f"no safetensors checkpoint in {path}")

    def names(self) -> list[str]:
        return list(self._files)

    def read(self, names: list[str]) -> dict[str, torch.Tensor]:
        from collections import defaultdict

        from safetensors import safe_open

        by_file: dict[Path, list[str]] = defaultdict(list)
        for name in names:
            by_file[self._files[name]].append(name)
        tensors: dict[str, torch.Tensor] = {}
        for file, wanted in by_file.items():
            with safe_open(str(file), "pt") as handle:
                for name in wanted:
                    tensors[name] = handle.get_tensor(name)
        return tensors


def fuse_layer_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Stack per-expert tensors the way the module holds them; leave everything else alone.

    How many experts there are is read off the tensors, not off a config key: checkpoints and
    transformers have renamed that field more than once, and a miscount here would surface as a
    miner who cannot be verified.
    """
    indices = {
        int(name.split("mlp.experts.")[1].split(".")[0])
        for name in state
        if "mlp.experts." in name and name.split("mlp.experts.")[1].split(".")[0].isdigit()
    }
    if not indices:
        return dict(state)
    if indices != set(range(len(indices))):
        raise ValueError(f"expert tensors are not numbered 0..n: {sorted(indices)[:5]}")
    fused = {k: v for k, v in state.items() if "mlp.experts." not in k}
    gate_up, down = [], []
    for expert in range(len(indices)):
        prefix = f"mlp.experts.{expert}."
        gate_up.append(torch.cat([state[prefix + "gate_proj.weight"], state[prefix + "up_proj.weight"]], dim=0))
        down.append(state[prefix + "down_proj.weight"])
    fused["mlp.experts.gate_up_proj"] = torch.stack(gate_up)
    fused["mlp.experts.down_proj"] = torch.stack(down)
    return fused


class Prefetching(LayerSource):
    """Reads layer i+1 while the caller is still using layer i.

    One thread, one layer ahead: enough to hide a read behind the device's work, and few enough
    buffers in flight that the host does not grow with the depth of the model.
    """

    def __init__(self, source: LayerSource) -> None:
        self._source = source
        self.n_layers = source.n_layers
        self._pool = ThreadPoolExecutor(1)
        self._pending: dict[int, Any] = {}

    def state(self, index: int) -> Mapping[str, torch.Tensor]:
        pending = self._pending.pop(index, None)
        state = pending.result() if pending is not None else self._source.state(index)
        if index + 1 < self.n_layers and index + 1 not in self._pending:
            self._pending[index + 1] = self._pool.submit(self._source.state, index + 1)
        return state

    def close(self) -> None:
        for pending in self._pending.values():
            pending.cancel()
        self._pending.clear()
        self._pool.shutdown(cancel_futures=True)
        self._source.close()
