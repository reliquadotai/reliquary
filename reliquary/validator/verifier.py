"""GRAIL proof verification — primitives used by GrpoWindowBatcher.

The orchestration lives in `reliquary.validator.batcher`. This module only
exposes the per-commit checks that touch the model or the signature scheme.
"""

import logging
from typing import Any, NamedTuple

import torch

from reliquary.constants import (
    LAYER_INDEX,
    T_PROTO,
)
from reliquary.shared.modeling import LOGIT_SOFTCAP_ATTR

logger = logging.getLogger(__name__)

class DrawSettings(NamedTuple):
    """The forced draw the active profile prescribes, as the verifier checks it."""

    draw: str
    t: float
    top_k: int
    top_p: float
    stochastic_threshold: float
    # In the draw's own miss units: CDF distance, or Gumbel relative score shortfall.
    boundary_epsilon: float


def draw_settings() -> DrawSettings:
    """Read at call time, so a process checks the draw its profile declares."""
    from reliquary import constants
    from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX

    gumbel = constants.FORCED_DRAW == DRAW_GUMBEL_MAX
    return DrawSettings(
        draw=constants.FORCED_DRAW,
        t=T_PROTO,
        top_k=constants.TOP_K_PROTO,
        top_p=constants.TOP_P_PROTO,
        stochastic_threshold=constants.FORCED_SEED_STOCHASTIC_MAXPROB,
        boundary_epsilon=(
            constants.FORCED_SEED_GUMBEL_BOUNDARY_EPSILON
            if gumbel else constants.FORCED_SEED_CDF_BOUNDARY_EPSILON
        ),
    )


def lm_head_fp32() -> bool:
    """Whether the profile's logits come from an fp32 LM head (``sampling.lm_head_fp32``)."""
    from reliquary import constants

    return bool(constants.LM_HEAD_FP32)


def verify_signature(commit: dict, hotkey: str) -> bool:
    """Hard check: verify Ed25519 signature on commit binding."""
    from reliquary.protocol.signatures import verify_commit_signature

    return verify_commit_signature(commit, hotkey)


class _LazyLogitRows:
    """Read-only row view of ``lm_head(hidden)`` that never builds the block.

    The full [seq_len, vocab] tensor is 8.2 GB at production scale (16.5k
    tokens x 248k vocab), allocated once per rollout and 8 times per proof on
    a GPU already past its allocator cliff. Every consumer below only ever
    reads rows out of it, so this computes lm_head on the rows asked for and
    nothing else — including skipping the prompt rows, which nothing reads.

    Duck-types the read-only tensor surface those consumers use (integer,
    slice and tensor indexing, ``index_select``, ``shape``, ``size``,
    ``device``) so the shared ``forced_sampling`` chunk loop works unchanged.

    Rows are NOT bit-identical to indexing a materialised ``lm_head(hidden)``:
    projecting [n, hidden] and [seq_len, hidden] selects different kernels and
    accumulation orders, which measures at 1-2 ulps. That matters because the
    forced-seed gate enforces *exact* inverse-CDF picks, so the drift is what
    could flip an honest miner's verdict. It moves the CDF by ~6e-8 against a
    FORCED_SEED_CDF_BOUNDARY_EPSILON of 2e-3; the bound is asserted in
    tests/unit/test_verifier_fused_lm_head.py.
    """

    def __init__(self, hidden: torch.Tensor, lm_head: Any, vocab_size: int, *,
                 fp32_head: bool = False) -> None:
        self._hidden = hidden
        self._lm_head = lm_head
        self._fp32_head = bool(fp32_head)
        self._weight_fp32 = cached_fp32_head_weight(lm_head) if self._fp32_head else None
        self.shape = torch.Size((int(hidden.shape[0]), int(vocab_size)))

    @property
    def device(self) -> Any:
        return self._hidden.device

    def size(self, dim: int | None = None) -> Any:
        return self.shape if dim is None else self.shape[dim]

    def _project(self, rows: torch.Tensor) -> torch.Tensor:
        # The rows are projected after the caller's no_grad forward has
        # returned, so guard here too: a graph held on lm_head weights would
        # retain exactly the memory this class exists to free.
        with torch.no_grad():
            if self._fp32_head:
                logits = _fp32_head_logits(rows, self._lm_head, weight_fp32=self._weight_fp32)
            else:
                logits = self._lm_head(rows)
            return apply_logit_softcap(logits, self._lm_head)


    def index_select(self, dim: int, index: torch.Tensor) -> torch.Tensor:
        if dim != 0:
            raise ValueError("_LazyLogitRows only selects along rows")
        return self._project(self._hidden.index_select(0, index))

    def __getitem__(self, key: Any) -> torch.Tensor:
        return self._project(self._hidden[key])


_MM_OUT_DTYPE: bool | None = None


def _mm_has_out_dtype() -> bool:
    """Whether this torch's ``torch.mm`` takes ``out_dtype`` on CUDA (probed once on a 1x1 product)."""
    global _MM_OUT_DTYPE
    if _MM_OUT_DTYPE is None:
        try:
            one = torch.ones(1, 1, dtype=torch.bfloat16, device="cuda")
            torch.mm(one, one, out_dtype=torch.float32)
            _MM_OUT_DTYPE = True
        except TypeError:
            _MM_OUT_DTYPE = False
    return _MM_OUT_DTYPE


def fp32_head_weight(weight: torch.Tensor) -> torch.Tensor | None:
    """The fp32 operand ``fp32_head_matmul`` would cast ``weight`` to on every call, cast once, for a caller that
    projects many row chunks through the same weight (the forced draw of a micro-batch: 4 GB re-cast per 512-row
    chunk on the 9B costs as much as the TF32 product). None when the call needs no cast (``out_dtype``, or not a
    bf16 CUDA weight) or ``RELIQUARY_FP32_HEAD_WEIGHT_CACHE=0``."""
    import os

    if os.environ.get("RELIQUARY_FP32_HEAD_WEIGHT_CACHE", "1") == "0":
        return None
    weight = weight.detach()
    if not (weight.is_cuda and weight.dtype == torch.bfloat16) or _mm_has_out_dtype():
        return None
    return weight.float()


def fp32_head_matmul(rows: torch.Tensor, weight: torch.Tensor, *,
                     weight_fp32: torch.Tensor | None = None) -> torch.Tensor:
    """``sampling.lm_head_fp32``: rows @ W.T with fp32 products and accumulation, fp32 out. With bf16 operands on
    CUDA (production) it is the cuBLAS call vLLM's ``head_dtype=float32`` makes (``torch.mm(..., out_dtype=
    float32)``, tensor cores; 13.7x faster than an fp32 SGEMM at 2048 rows of the 9B on H100, 1 different Gumbel
    pick in 381k, 2026-10-16). On a torch without ``out_dtype`` (2.7, the pinned one) the same tensor-core product:
    the operands upcast to fp32 with TF32 on. A bf16 value (8-bit significand) is exact in TF32 (11-bit), so TF32
    rounds nothing and the products and fp32 accumulation are those of ``out_dtype``; only the summation order may
    differ. With TF32 off this fallback ran the head 10x slower: ~65 % of every training step on
    teutonic-rl-async-v1 (2026-10-23). Any other operands: an fp32 matmul with TF32 off. ``weight_fp32``:
    ``fp32_head_weight(weight)``, the same upcast done once by the caller (the products are unchanged).
    bf16 x bf16 products are exact in fp32: only the summation order differs (~4e-4 at most on the 9B)."""
    weight = weight.detach()
    bf16_cuda = rows.is_cuda and rows.dtype == weight.dtype == torch.bfloat16
    if bf16_cuda and _mm_has_out_dtype():
        with torch.autocast(device_type="cuda", enabled=False):
            return torch.mm(rows, weight.T, out_dtype=torch.float32)
    # no out_dtype in this torch (or not bf16 CUDA): the same products through TF32 below
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bf16_cuda
    try:
        with torch.autocast(device_type=rows.device.type, enabled=False):
            rhs = weight_fp32 if weight_fp32 is not None and bf16_cuda else weight.float()
            return rows.float() @ rhs.T
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def apply_logit_softcap(logits: torch.Tensor, lm_head: Any) -> torch.Tensor:
    """``cap * tanh(logits / cap)`` when the model caps its final logits (Gemma: ``final_logit_softcapping``,
    marked on the head by ``shared.modeling``), as its own forward and vLLM do before any sampling; else as is."""
    cap = getattr(lm_head, LOGIT_SOFTCAP_ATTR, None)
    if not cap:
        return logits
    return torch.tanh(logits / cap) * cap


def _fp32_head_logits(rows: torch.Tensor, lm_head: Any, *, weight_fp32: torch.Tensor | None = None) -> torch.Tensor:
    if getattr(lm_head, "bias", None) is not None:
        raise RuntimeError("fp32 LM head: a biased head is not supported")
    return fp32_head_matmul(rows.reshape(-1, rows.shape[-1]), lm_head.weight,
                            weight_fp32=weight_fp32).reshape(*rows.shape[:-1], -1)


_HEAD_WEIGHT_CACHE_ATTR = "_reliquary_fp32_head_weight"


def cached_fp32_head_weight(lm_head: Any) -> torch.Tensor | None:
    """``fp32_head_weight`` of ``lm_head``, cast once and kept on the module until its weight changes (a swap or a
    delta writes it in place and bumps its version). The audit projects a sequence's rows in hundreds of chunks:
    re-cast per chunk, the head took ~60 % of the warped verify on the 9B (H200, torch 2.7, 2026-11-07)."""
    weight = getattr(lm_head, "weight", None)
    if not isinstance(weight, torch.Tensor):
        return None
    try:
        stamp = (weight.data_ptr(), weight._version, tuple(weight.shape))
    except RuntimeError:  # an inference-mode tensor has no version counter: never cached
        return fp32_head_weight(weight)
    held = getattr(lm_head, _HEAD_WEIGHT_CACHE_ATTR, None)
    if held is not None and held[0] == stamp:
        return held[1]
    cast = fp32_head_weight(weight)
    try:
        object.__setattr__(lm_head, _HEAD_WEIGHT_CACHE_ATTR, (stamp, cast) if cast is not None else None)
    except (AttributeError, TypeError):
        pass
    return cast


def _lm_head_vocab_size(lm_head: Any) -> int | None:
    """Vocabulary width of an lm_head without projecting anything through it."""
    out_features = getattr(lm_head, "out_features", None)
    if isinstance(out_features, int) and out_features > 0:
        return out_features
    weight = getattr(lm_head, "weight", None)
    if weight is not None and getattr(weight, "ndim", 0) == 2:
        return int(weight.shape[0])
    return None


def signed_episode_spans(rollout_meta: dict[str, Any], seq_len: int) -> list[tuple[int, int]] | None:
    """The model spans of a signed episode; None for any other rollout, ``[]`` when malformed
    (no proof passes an empty span list). Read strictly: integer pairs, ordered, non-empty,
    non-overlapping, inside the sequence and after the declared prompt."""
    from reliquary.protocol.submission import is_signed_episode

    if not is_signed_episode(rollout_meta):
        return None
    episode = rollout_meta["episode"]
    raw_spans = episode.get("assistant_spans")
    if not isinstance(raw_spans, (list, tuple)):
        return []
    try:
        prompt_length = int(rollout_meta.get("prompt_length", 0))
    except (TypeError, ValueError, OverflowError):
        return []
    spans: list[tuple[int, int]] = []
    previous = max(1, prompt_length)
    for raw in raw_spans:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            return []
        if not all(type(value) is int for value in raw):
            return []
        start, end = raw
        if start < previous or end <= start or end > seq_len:
            return []
        spans.append((start, end))
        previous = end
    return spans


def proof_forward_rows(model: Any, tokens: list[int], *, forward: tuple[Any, Any] | None = None,
                       device: Any = None) -> tuple[torch.Tensor, Any]:
    """The proof path's forward of one sequence: (hidden [seq_len, hidden], logit rows [seq_len, vocab]).

    Batch 1 on the model's device unless ``forward`` hands this sequence's rows out of a batched pass.
    The logit rows are projected lazily through ``lm_head`` (in fp32 when the profile's
    ``sampling.lm_head_fp32`` says so); a model without a separable head returns its materialised logits."""
    from reliquary.shared.forward import forward_single_layer

    lm_head = getattr(model, "lm_head", None)
    # Row-wise projection needs a separable head whose width we can read
    # without projecting; anything else keeps the old materialised block.
    vocab_size = _lm_head_vocab_size(lm_head)
    can_project_rows = vocab_size is not None
    if forward is not None:
        # This rollout's rows out of a pass that ran for several of them: one traversal of the
        # model serves the whole batch, which is what makes a streamed replica affordable.
        hidden_states_gpu, logits_batch = forward
    else:
        if device is None:
            device = next(model.parameters()).device
        input_ids = torch.tensor([list(tokens)], device=device)
        with torch.no_grad():
            hidden_rows, logits_rows = forward_single_layer(
                model, input_ids, None, LAYER_INDEX,
                materialize_logits=not can_project_rows,
            )
        hidden_states_gpu = hidden_rows[0]
        logits_batch = None if logits_rows is None else logits_rows[0]

    fp32_head = lm_head_fp32()
    if logits_batch is None:
        return hidden_states_gpu, _LazyLogitRows(hidden_states_gpu, lm_head, vocab_size,
                                                 fp32_head=fp32_head)
    if fp32_head:
        raise RuntimeError("fp32 LM head: this model's logits come materialised from its "
                           "forward, not from a separable lm_head")
    return hidden_states_gpu, logits_batch  # [seq_len, vocab_size], kept on GPU


def _gpu_seed_consistency_diagnostics(
    logits_slice: torch.Tensor,
    token_ids: list[int],
    u_values: list[float],
    *,
    position_offsets: list[int] | None = None,
    logit_positions: list[int] | None = None,
):
    from reliquary.environment.forced_sampling import (
        seed_consistency_diagnostics,
    )

    settings = draw_settings()
    return seed_consistency_diagnostics(
        logits_slice,
        list(token_ids),
        list(u_values),
        t=settings.t,
        top_k=settings.top_k,
        top_p=settings.top_p,
        stochastic_threshold=settings.stochastic_threshold,
        boundary_epsilon=settings.boundary_epsilon,
        position_offsets=position_offsets,
        logit_positions=logit_positions,
        draw=settings.draw,
    )


