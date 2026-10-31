"""The forced draw of an eval job: one legal generation per (job, prompt, sample).

An eval job whose ``sampling.draw`` is ``gumbel_max`` fixes every sampled token with a public Gumbel-max draw
(``environment.forced_sampling``), at T=1 over the full vocabulary with the LM head in fp32, the RL path's draw.
Its stream is keyed on the job alone, never on the miner:

* ``randomness``: ``sha256(EVAL_DRAW_DOMAIN, job_id, checkpoint_sha256, seed)`` in hex;
* ``prompt_idx``: the source row;
* ``checkpoint_hash``: the job's checkpoint revision;
* ``rollout_index``: the sample, ``draw_slot * n + completion`` (``draw_slot`` is the submission's ``cursor``,
  which a free-order job does not walk);
* ``t``: the model-token position (an episode counts its model tokens across turns, as ``rl_draw`` does).

So whoever generates sample k of a prompt gets the same tokens: a miner cannot re-roll a sample until it likes
the outcome, and the score does not depend on who mined it. The validator recomputes the draw over its own
forward and refuses a sample that does not follow it (``judge``): a 0.98 match floor and the RL audit's edit check.
A token that differs is never substituted (the tokens after it would not fit it); honest bf16 drift stays far under
the floor.

A draw slot is taken once per attempt (``slot_key`` joins the accepted digests). A sample that failed its audit
reopens its own slot, with the same draw, up to ``ATTEMPTS_PER_SLOT`` attempts: the samples a job scores never
depend on which ones failed or who mined them.
"""
from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any

from reliquary.constants import (
    FORCED_SEED_GUMBEL_BOUNDARY_EPSILON,
    FORCED_SEED_STOCHASTIC_MAXPROB,
)
from reliquary.environment.forced_sampling import DRAW_GUMBEL_MAX, u_range

EVAL_DRAW = DRAW_GUMBEL_MAX
EVAL_DRAW_DOMAIN = "reliquary-eval-draw-v1"
EVAL_DRAWS = (EVAL_DRAW,)
REASON_DRAW_MISMATCH = "draw_mismatch"
# The draw a sample must follow: at least this share of its stochastic positions exact (honest bf16 drift
# measured 0.69 % at worst on H200, 2026-10-28), with up to EVAL_MATCH_GRACE misses allowed whatever its length
# (never more than a quarter of it): honest short samples miss 2-2.5 % (live, 2026-10-28), a re-roll 30-50 %.
# Nothing is substituted: a token changed would not fit the tokens after it.
EVAL_MATCH_FLOOR = 0.98
# Gumbel score gaps (best score minus the submitted token's): honest misses on Qwen3.8-27B at T 1, 2026-10-31:
# q99.9 0.64, max 2.9; half of all runner-up edits are over 4. The last token's is tighter (just over the honest
# maximum): a sample cut where the draw went on has its end token far from the pick.
EVAL_GAP_TOLERANCE = 4.0
EVAL_TERMINAL_GAP = 3.0
EVAL_MATCH_GRACE = 8
REASON_DRAW_SLOT_TAKEN = "draw_slot_taken"
REASON_BAD_CURSOR = "bad_cursor"


def has_draw(job: Any) -> bool:
    return getattr(job.sampling, "draw", None) is not None


def draw_randomness(job: Any) -> str:
    """The job's public stream key. The job id binds the set and the checkpoint step; the seed is the job's."""
    h = hashlib.sha256()
    for part in (EVAL_DRAW_DOMAIN, job.job_id, job.checkpoint_sha256, str(job.seed)):
        data = part.encode("utf-8")
        h.update(len(data).to_bytes(4, "big"))
        h.update(data)
    return h.hexdigest()


def slot_count(job: Any) -> int:
    """Draw slots a prompt has: one per sample. A failed one reopens with the same draw (``slot_key``'s attempt)."""
    return int(job.slots_per_prompt)


def rollout_index(job: Any, draw_slot: int, completion: int = 0) -> int:
    return int(draw_slot) * int(job.sampling.n) + int(completion)


def slot_key(prompt_index: int, draw_slot: int, attempt: int = 0) -> str:
    """The seen key of a taken draw slot at one attempt in the job's ledger: a digest like the others it sits with,
    domain-separated so it never equals a completion's or a session's. Attempt 0 keeps the first key's form."""
    base = f"{EVAL_DRAW_DOMAIN}/slot/{int(prompt_index)}/{int(draw_slot)}"
    return hashlib.sha256((base if not attempt else f"{base}/{int(attempt)}").encode()).hexdigest()


def attempt_digests(prompt_index: int, draw_slot: int, attempt: int, digests) -> list[str]:
    """A draw job's seen digests: the completion digests salted with their slot, attempt and position, so the
    honest redo of a failed slot (the same tokens) is not a duplicate, and two equal completions of one slot
    are not either (the draw fixes them; the slot key already makes each sample unique)."""
    prefix = f"{EVAL_DRAW_DOMAIN}/sample/{int(prompt_index)}/{int(draw_slot)}/{int(attempt)}"
    return [hashlib.sha256(f"{prefix}/{c}/{d}".encode()).hexdigest() for c, d in enumerate(digests)]


def draw_context(job: Any, prompt_index: int, draw_slot: int, completion: int = 0) -> dict[str, Any]:
    """The forced-seed request fields (``vllm_generation.forced_seed_extra_args``) of one sample."""
    return {"randomness": draw_randomness(job), "prompt_idx": int(prompt_index),
            "checkpoint_hash": str(job.checkpoint_revision),
            "rollout_index": rollout_index(job, draw_slot, completion),
            "domain": EVAL_DRAW_DOMAIN, "draw": EVAL_DRAW}


def uniforms(job: Any, prompt_index: int, draw_slot: int, completion: int, positions: int) -> list[float]:
    """The public uniforms of one sample at model-token positions ``0 .. positions - 1``."""
    context = draw_context(job, prompt_index, draw_slot, completion)
    return u_range(context["randomness"], context["prompt_idx"], context["checkpoint_hash"],
                   context["rollout_index"], int(positions), domain=EVAL_DRAW_DOMAIN)


def diagnostics(logits, token_ids, u_values):
    """``seed_consistency_diagnostics`` under the eval draw (T=1, full vocabulary)."""
    from reliquary.environment.forced_sampling import seed_consistency_diagnostics

    return seed_consistency_diagnostics(
        logits, list(token_ids), list(u_values), t=1.0, top_k=0, top_p=1.0,
        stochastic_threshold=FORCED_SEED_STOCHASTIC_MAXPROB,
        boundary_epsilon=FORCED_SEED_GUMBEL_BOUNDARY_EPSILON, draw=EVAL_DRAW)


def _counted(d) -> tuple[int, int]:
    """(positions, exact) the floor reads: every drawn position. Honest misses are near-ties, which stay near-ties
    at any temperature, so a count over stochastic positions only would refuse more honest samples as the
    temperature drops (measured on Qwen3.8-27B, 2026-10-31: 3.4 % at T 1, 21 % at T 0.7 / top_p 0.8, against
    0.1-0.3 % over every position)."""
    return d.n_positions, d.n_drawn_exact_match


def match_rate(d) -> float:
    counted, exact = _counted(d)
    return exact / counted if counted else 0.0


def allowed_misses(counted: int) -> int:
    return max(min(EVAL_MATCH_GRACE, max(1, counted // 4)), math.floor((1 - EVAL_MATCH_FLOOR) * counted))


def _gap(miss: float) -> float:
    """A Gumbel miss (``1 - exp(-gap)``) as its score gap; an impossible token is an infinite one."""
    return -math.log1p(-miss) if miss < 1.0 else math.inf


def judge(diags) -> tuple[bool, dict[str, Any]]:
    """One sample's verdict over its diagnostics (one per completion or trajectory). Every one must:

    * follow the draw: at most ``1 - EVAL_MATCH_FLOOR`` of its drawn positions missed, up to ``EVAL_MATCH_GRACE``
      misses allowed in any case (a quarter of a shorter sample);
    * carry no edit: every miss within ``EVAL_GAP_TOLERANCE`` of the pick's score (an honest miss is a near-tie;
      a token written in place of the draw is far from it);
    * end where the draw ended it: the last token within ``EVAL_TERMINAL_GAP`` (a sample cut short is not a
      near-tie)."""
    worst: dict[str, Any] = {}
    for i, d in enumerate(diags):
        counted, exact = _counted(d)
        misses = counted - exact
        detail = {"completion": i, "match": round(exact / counted, 4) if counted else 0.0,
                  "positions": d.n_positions, "stochastic": d.n_stochastic, "misses": misses,
                  "max_gap": round(_gap(d.max_cdf_miss), 4), "last_gap": round(_gap(d.last_miss), 4)}
        if not worst or detail["match"] < worst["match"]:
            worst = detail
        followed = counted > 0 and misses <= allowed_misses(counted)
        if (not followed or _gap(d.max_cdf_miss) > EVAL_GAP_TOLERANCE
                or _gap(d.last_miss) > EVAL_TERMINAL_GAP):
            return False, detail
    return True, worst


def parse_draw(raw: Mapping[str, Any]) -> str | None:
    """``sampling.draw`` of a job contract: absent (free sampling) or the eval draw."""
    value = raw.get("draw")
    if value is None:
        return None
    if value not in EVAL_DRAWS:
        raise ValueError(f"sampling.draw must be one of {EVAL_DRAWS}, got {value!r}")
    return value


__all__ = ["EVAL_DRAW", "EVAL_DRAW_DOMAIN", "EVAL_GAP_TOLERANCE", "EVAL_TERMINAL_GAP", "REASON_BAD_CURSOR", "REASON_DRAW_MISMATCH", "REASON_DRAW_SLOT_TAKEN", "allowed_misses", "attempt_digests", "diagnostics", "draw_context",
           "draw_randomness", "has_draw", "judge", "match_rate", "parse_draw", "rollout_index", "slot_count",
           "slot_key", "uniforms"]
