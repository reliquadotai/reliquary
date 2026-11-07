"""The forced-draw check as data, so every audit path computes it alike.

A draw job's sample is checked by the model forward that already scores it: the in-process auditor, the
validator's GPU process, a remote executor. Each gets, per row, the sample's public draw context (``DrawSpec``)
and returns the row's diagnostics (``diagnostics_to_wire``); the auditor applies the job's judge
(``eval_draw.judge``). Nothing about a verdict crosses, as with TOPLOC's chunk scores.

A row of a draw job carries no proof: the draw is its only check (``proofs`` is None).
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.environment.forced_sampling import (
    IDENTITY_WARP,
    SeedConsistencyDiagnostics,
    Warp,
    check_warp,
)

MAX_EDGE_MARGIN = 0.5
_SPEC_FIELDS = ("randomness", "prompt_idx", "checkpoint_hash", "rollout_index", "domain", "warp", "edge_margin")
_DIAG_FIELDS = ("positions", "stochastic", "exact", "max_miss", "last_miss")
# Sent by an executor that counts its far misses (``n_far_miss``); an older one omits it.
_FAR_FIELD = "far_misses"


@dataclass(frozen=True, slots=True)
class DrawSpec:
    """Where a sample sits in its job's public draw, and what the draw samples."""

    randomness: str
    prompt_idx: int
    checkpoint_hash: str
    rollout_index: int
    domain: str
    warp: Warp = IDENTITY_WARP
    edge_margin: float = 0.0

    def to_wire(self) -> dict[str, Any]:
        return {"randomness": self.randomness, "prompt_idx": self.prompt_idx,
                "checkpoint_hash": self.checkpoint_hash, "rollout_index": self.rollout_index,
                "domain": self.domain, "warp": self.warp.to_dict(), "edge_margin": self.edge_margin}

    @classmethod
    def from_wire(cls, raw: Mapping) -> "DrawSpec":
        if not isinstance(raw, Mapping) or set(raw) != set(_SPEC_FIELDS):
            raise ValueError(f"a draw spec has exactly {list(_SPEC_FIELDS)}")
        for name in ("prompt_idx", "rollout_index"):
            if isinstance(raw[name], bool) or not isinstance(raw[name], int) or raw[name] < 0:
                raise ValueError(f"draw spec {name} must be a non-negative whole number")
        for name in ("randomness", "checkpoint_hash", "domain"):
            if not isinstance(raw[name], str) or not 0 < len(raw[name]) <= 256:
                raise ValueError(f"draw spec {name} must be a short string")
        margin = raw["edge_margin"]
        if isinstance(margin, bool) or not isinstance(margin, (int, float)) or not 0 <= margin <= MAX_EDGE_MARGIN:
            raise ValueError(f"draw spec edge_margin must be in [0, {MAX_EDGE_MARGIN}]")
        warp = Warp.from_dict(raw["warp"])
        check_warp(warp)
        return cls(raw["randomness"], raw["prompt_idx"], raw["checkpoint_hash"], raw["rollout_index"],
                   raw["domain"], warp, float(margin))

    def uniforms(self, positions: int) -> list[float]:
        from reliquary.environment.forced_sampling import u_range

        return u_range(self.randomness, self.prompt_idx, self.checkpoint_hash, self.rollout_index,
                       int(positions), domain=self.domain)


def spec_for(job: Any, prompt_index: int, draw_slot: int, completion: int = 0) -> DrawSpec:
    """A draw job's sample (``eval_draw.draw_context``) with its warp and its judge's edge margin."""
    from reliquary.corpus import eval_draw

    context = eval_draw.draw_context(job, prompt_index, draw_slot, completion)
    judge = getattr(job, "judge", None)
    return DrawSpec(context["randomness"], context["prompt_idx"], context["checkpoint_hash"],
                    context["rollout_index"], context["domain"], eval_draw.job_warp(job),
                    0.0 if judge is None else float(judge.edge_margin))


def row_diagnostics(lm_head, rows, tokens, prompt_len: int, spans, spec: DrawSpec) -> dict[str, Any]:
    """One row's diagnostics, on the hidden states the forward produced: a completion's rows, or a trajectory's
    list of span rows (draw position counted over the model tokens across turns). The LM head runs in fp32."""
    import torch

    from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, seed_consistency_diagnostics
    from reliquary.constants import FORCED_SEED_GUMBEL_BOUNDARY_EPSILON, FORCED_SEED_STOCHASTIC_MAXPROB
    from reliquary.validator.verifier import _LazyLogitRows, _lm_head_vocab_size

    if spans is None:
        hidden, drawn = rows, list(tokens[prompt_len:])
    else:
        hidden = torch.cat(list(rows), dim=0)
        drawn = [tokens[p] for start, end in spans for p in range(start, end)]
    logits = _LazyLogitRows(hidden, lm_head, _lm_head_vocab_size(lm_head), fp32_head=True)
    w = spec.warp
    d = seed_consistency_diagnostics(
        logits, drawn, spec.uniforms(len(drawn)), t=w.t, top_k=w.top_k, top_p=w.top_p, min_p=w.min_p,
        edge_margin=spec.edge_margin, stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB,
        boundary_epsilon=FORCED_SEED_GUMBEL_BOUNDARY_EPSILON, draw=DRAW_GUMBEL_MAX)
    return diagnostics_to_wire(d)


def diagnostics_to_wire(d: SeedConsistencyDiagnostics) -> dict[str, Any]:
    """What the judge reads (``eval_draw.judge``), as plain numbers."""
    return {"positions": int(d.n_positions), "stochastic": int(d.n_stochastic),
            "exact": int(d.n_drawn_exact_match), "max_miss": float(d.max_cdf_miss),
            "last_miss": float(d.last_miss), _FAR_FIELD: int(d.n_far_miss)}


def diagnostics_from_wire(raw: Mapping) -> SeedConsistencyDiagnostics:
    if not isinstance(raw, Mapping) or set(raw) - {_FAR_FIELD} != set(_DIAG_FIELDS):
        raise ValueError(f"draw diagnostics have exactly {list(_DIAG_FIELDS)} (and {_FAR_FIELD!r})")
    far = raw.get(_FAR_FIELD, 0)
    if isinstance(far, bool) or not isinstance(far, int) or not 0 <= far <= raw.get("positions", 0):
        raise ValueError(f"draw diagnostics {_FAR_FIELD} must be a whole number within its positions")
    counts = [raw[name] for name in ("positions", "stochastic", "exact")]
    if any(isinstance(c, bool) or not isinstance(c, int) or c < 0 for c in counts):
        raise ValueError("draw diagnostics counts must be non-negative whole numbers")
    positions, stochastic, exact = counts
    if stochastic > positions or exact > positions:
        raise ValueError("draw diagnostics count more than their positions")
    misses = [raw["max_miss"], raw["last_miss"]]
    if any(isinstance(m, bool) or not isinstance(m, (int, float)) or not math.isfinite(m) or not 0 <= m <= 1
           for m in misses):
        raise ValueError("draw diagnostics misses must be in [0, 1]")
    return SeedConsistencyDiagnostics(n_positions=positions, n_stochastic=stochastic,
                                      n_drawn_exact_match=exact, max_cdf_miss=float(misses[0]),
                                      last_miss=float(misses[1]), n_far_miss=far)


# Two honest scorers of one row, on different hardware: exact picks within one, the gaps the judge reads (the worst
# miss's and the last token's, ``eval_draw.judge``) within a near-tie's drift. A scorer that hides an edit (a gap of 4
# or more reported as 0) or invents one disagrees.
GAP_AGREEMENT = 1.0


def _gap(miss: float) -> float:
    return -math.log1p(-miss) if miss < 1.0 else math.inf


def diagnostics_agree(a: Mapping, b: Mapping, *, slack: int = 1, gap_slack: float = GAP_AGREEMENT) -> bool:
    """Two scorers' diagnostics of one row agree: the same positions, exact picks within ``slack``, and the worst
    and last gaps within ``gap_slack`` (both infinite, or both finite and close)."""
    if a["positions"] != b["positions"] or abs(a["exact"] - b["exact"]) > slack:
        return False
    if _FAR_FIELD in a and _FAR_FIELD in b and abs(a[_FAR_FIELD] - b[_FAR_FIELD]) > slack:
        return False
    for name in ("max_miss", "last_miss"):
        ga, gb = _gap(float(a[name])), _gap(float(b[name]))
        if math.isinf(ga) or math.isinf(gb):
            if ga != gb:
                return False
        elif abs(ga - gb) > gap_slack:
            return False
    return True


__all__ = ["DrawSpec", "diagnostics_agree", "diagnostics_from_wire", "diagnostics_to_wire", "row_diagnostics",
           "spec_for"]
