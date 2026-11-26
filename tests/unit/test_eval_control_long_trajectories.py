"""Long agentic trajectories on the order control's production audit path.

The order control (``eval_control.PairedAuditDispatcher.claim``) leases a drawn agentic sample in v3 (v4 for a
greedy warp) with its spans; the executor reads the lease with ``AuditLease.model_validate`` on the response's
JSON and scores it. While ``MAX_ITEM_SPANS`` was 64, samples of 70 and 95 turns failed that parse on every
executor, which lost the leases, was quarantined, and left its model's queue frozen behind them.
"""

import asyncio
import json

import pytest

from reliquary.corpus import eval_draw
from reliquary.corpus.draw_audit import DrawSpec
from reliquary.corpus.job import DEFAULT_DRAW_JUDGE
from reliquary.environment.forced_sampling import Warp
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.submission import MAX_SIGNED_EPISODE_TURNS
from reliquary.validator.corpus_audit_protocol import (
    AUDIT_PROTOCOL,
    AUDIT_PROTOCOL_V3,
    AUDIT_PROTOCOL_V4,
    AuditLease,
    AuditResult,
)
from tests.unit.test_eval_control import MODEL_A, _executor, _Harness

SAMPLED, GREEDY = Warp(t=0.7, top_k=20), Warp(top_k=1)


def _drawn_item(turns, warp, prompt_len=8):
    """A drawn sample of ``turns`` one-token assistant turns, each followed by a one-token tool reply."""
    spec = DrawSpec("ab" * 32, 3, "rev1", 0, eval_draw.EVAL_DRAW_DOMAIN, warp, 0.15)
    return {"tokens": [10 + i % 400 for i in range(prompt_len + 2 * turns)], "prompt_len": prompt_len,
            "proofs": None, "spans": [(prompt_len + 2 * k, prompt_len + 2 * k + 1) for k in range(turns)],
            "draw": spec.to_wire()}


def _order_control_lease(item):
    """The lease the order control hands an executor that claims every protocol, for this one sample."""
    async def go():
        harness = _Harness(scorers=1)
        scoring = asyncio.ensure_future(
            harness.dispatcher.view(*MODEL_A, PROOF, "j", DEFAULT_DRAW_JUDGE).score([item]))
        await asyncio.sleep(0)
        lease = harness.dispatcher.claim(_executor("a"), (AUDIT_PROTOCOL, AUDIT_PROTOCOL_V3, AUDIT_PROTOCOL_V4))
        scoring.cancel()
        return lease
    return asyncio.run(go())


def _as_the_executor_reads_it(lease):
    return AuditLease.model_validate(json.loads(json.dumps(lease)))


@pytest.mark.parametrize("turns,warp,protocol", [
    (70, SAMPLED, AUDIT_PROTOCOL_V3), (95, SAMPLED, AUDIT_PROTOCOL_V3), (95, GREEDY, AUDIT_PROTOCOL_V4),
    (MAX_SIGNED_EPISODE_TURNS, SAMPLED, AUDIT_PROTOCOL_V3), (MAX_SIGNED_EPISODE_TURNS, GREEDY, AUDIT_PROTOCOL_V4)])
def test_the_order_controls_lease_of_a_long_trajectory_parses_on_the_executor(turns, warp, protocol):
    lease = _order_control_lease(_drawn_item(turns, warp))
    assert lease is not None and lease["protocol"] == protocol
    assert len(_as_the_executor_reads_it(lease).items[0].spans) == turns


@pytest.mark.parametrize("turns,warp", [(95, SAMPLED), (MAX_SIGNED_EPISODE_TURNS, GREEDY)])
def test_the_executor_scores_a_long_drawn_trajectory(turns, warp):
    from reliquary.validator.corpus_audit_executor import AuditExecutor
    from tests.unit.test_draw_only_audit import tiny

    lease = _as_the_executor_reads_it(_order_control_lease(_drawn_item(turns, warp)))
    executor = AuditExecutor(http=None, executor_id="pod", token="x", model_id="m", model_revision="r",
                             load_model=lambda model, revision: None)
    executor._model = tiny()
    (score,) = AuditResult.model_validate({"scores": executor._score(lease)}).scores
    assert score.status == "ok", score.detail
    assert score.chunks == [] and score.draw["positions"] == turns
