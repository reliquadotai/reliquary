"""A model replica that keeps one decoder layer on the device at a time.

Verification runs the model over a rollout and reads what it gives each token. On a
Mixture-of-Experts model at frontier scale, holding the model to do that is a rack of GPUs; almost
all of its weight is in experts that any given token barely touches. This replica keeps the fixed
parts — embeddings, final norm, output head — and walks the layers one at a time, applying each to
every rollout in the batch before moving on. One traversal of the model serves a whole batch, so
the cost of moving the weights is paid once and divided by however many rollouts are in flight.

It answers the handful of questions the verification path asks of a model, so nothing downstream
changes: `forward_single_layer` drives it exactly as it drives a resident model, and the hidden
states it returns are identical to the bit.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from reliquary.shared.fused_layers import is_fused_store
from reliquary.shared.layer_source import (
    CheckpointLayers,
    HostLayers,
    LayerSource,
    PinnedLayers,
    Prefetching,
)


@dataclass(frozen=True)
class _Output:
    """What the base model returns; `forward_single_layer` reads it by name or by index."""

    last_hidden_state: torch.Tensor

    def __getitem__(self, index: int) -> torch.Tensor:
        if index != 0:
            raise IndexError(index)
        return self.last_hidden_state


class _Base:
    """The callable `forward_single_layer` reaches through `model.<base_model_prefix>`."""

    def __init__(self, replica: "StreamedReplica") -> None:
        self._replica = replica

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        **_: Any,
    ) -> _Output:
        return _Output(self._replica.traverse(input_ids, attention_mask))


class StreamedReplica:
    """A stand-in for a loaded model that never holds more than one decoder layer."""

    base_model_prefix = "model"

    def __init__(self, skeleton: Any, source: LayerSource, device: str | torch.device = "cpu") -> None:
        self.config = skeleton.config
        self.generation_config = getattr(skeleton, "generation_config", None)
        self.device = torch.device(device)
        self._source = source
        self._skeleton = skeleton
        base = skeleton.model
        self.embed_tokens = base.embed_tokens.to(self.device).eval()
        self.norm = base.norm.to(self.device).eval()
        self.rotary_emb = base.rotary_emb.to(self.device)
        self.lm_head = skeleton.lm_head.to(self.device).eval()
        # One buffer, reused for every layer: this is what bounds device memory.
        self._buffer = copy.deepcopy(base.layers[0]).to(self.device).eval()
        self.model = _Base(self)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype | None = None,
        prefetch: bool = False,
        fused_dir: str | Path | None = None,
        attn_implementation: str | None = None,
    ) -> "StreamedReplica":
        """Build a replica whose layers stay on disk until the traversal reaches them.

        ``attn_implementation`` must match what a resident replica would use: the hidden states a
        verifier compares are the ones the kernels produce, so a different kernel here shows up
        downstream as a miner whose sketch does not match.
        """
        from transformers import AutoConfig, AutoModelForCausalLM

        config = AutoConfig.from_pretrained(path)
        extra = {} if attn_implementation is None else {"attn_implementation": attn_implementation}
        # Built in the target dtype rather than cast into it: casting the whole model afterwards
        # would take the rotary tables down with it, which a resident replica keeps in float32,
        # and the hidden states would part ways in the last bits.
        if dtype is not None:
            extra["dtype"] = dtype
        skeleton = AutoModelForCausalLM.from_config(config, **extra).eval()
        _load_fixed_parts(skeleton, Path(path))
        if fused_dir is None and is_fused_store(path):
            # A store is its own fused copy: it was written to be read this way.
            fused_dir = path
        source = _staged(
            CheckpointLayers(path, fused_dir=fused_dir),
            prefetch,
            pin=torch.device(device).type == "cuda",
        )
        return cls(skeleton, source, device=device)

    @classmethod
    def from_model(
        cls, model: Any, *, device: str | torch.device = "cpu", prefetch: bool = False
    ) -> "StreamedReplica":
        """Build a replica over a model already in host memory."""
        held = next(model.parameters()).device.type
        source = _staged(
            HostLayers(model), prefetch, pin=held == "cpu" and torch.device(device).type == "cuda",
        )
        return cls(model, source, device=device)

    @property
    def resident_bytes(self) -> int:
        """What this replica keeps: the fixed parts, plus the one layer buffer."""
        parts = (self.embed_tokens, self.norm, self.lm_head, self._buffer)
        return sum(p.numel() * p.element_size() for part in parts for p in part.parameters())

    def parameters(self):
        yield from self.embed_tokens.parameters()

    def eval(self) -> "StreamedReplica":
        return self

    @torch.no_grad()
    def traverse(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        """One pass over the model, one layer resident at a time."""
        if attention_mask is not None and not bool(attention_mask.all()):
            raise NotImplementedError(
                "streamed replica: a padded attention mask is not supported yet; "
                "batch rollouts of equal length or pass no mask"
            )
        from transformers.masking_utils import create_causal_mask

        input_ids = input_ids.to(self.device)
        hidden = self.embed_tokens(input_ids)
        positions = torch.arange(input_ids.shape[1], device=self.device).unsqueeze(0)
        cos, sin = self.rotary_emb(hidden, positions)
        # The model builds this mask itself before it reaches a layer, and an eager attention
        # given None would read the future. Built here the same way, from the same helper.
        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=hidden,
            attention_mask=None,
            past_key_values=None,
            position_ids=positions,
        )
        for index in range(self._source.n_layers):
            self._buffer.load_state_dict(self._source.state(index))
            out = self._buffer(
                hidden_states=hidden,
                attention_mask=causal_mask,
                position_ids=positions,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            hidden = out[0] if isinstance(out, tuple) else out
        return self.norm(hidden)

    def close(self) -> None:
        self._source.close()


def _staged(source: LayerSource, prefetch: bool, *, pin: bool) -> LayerSource:
    """Wrap a source in what the traversal it feeds is worth paying for.

    Page-locking costs host memory and buys nothing unless the layer is actually crossing a bus,
    which it is not when the weights already sit on the device the traversal runs on.
    """
    if pin:
        source = PinnedLayers(source)
    return Prefetching(source) if prefetch else source


def _load_fixed_parts(skeleton: Any, path: Path) -> None:
    """Materialise only what a traversal needs outside the layers."""
    from reliquary.shared.layer_source import LAYER_PREFIX, _ShardedTensors

    reader = _ShardedTensors(path)
    names = [name for name in reader.names() if not name.startswith(LAYER_PREFIX)]
    state = reader.read(names)
    missing = skeleton.load_state_dict(state, strict=False)
    if missing.unexpected_keys:
        raise ValueError(f"checkpoint carries tensors this model has no place for: {missing.unexpected_keys[:3]}")
    if getattr(skeleton.config, "tie_word_embeddings", False) and hasattr(skeleton, "tie_weights"):
        skeleton.tie_weights()
