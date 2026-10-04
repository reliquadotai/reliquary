"""Replay one recorded episode from a real 27B trace on a real box (Docker host only)."""

import json
import os

import pytest

pytest.importorskip("reliquary_swe")
TRACES = os.environ.get("RELIQUARY_REPLAY_TRACES")
pytestmark = pytest.mark.skipif(not TRACES, reason="set RELIQUARY_REPLAY_TRACES to a traces.jsonl")


async def test_an_honest_episode_replays_to_its_own_diff():
    from reliquary.corpus.replay_compare import actions_from_trace, compare
    from reliquary.validator.agentic_replay import replay_swe, swesmith_task

    with open(TRACES) as f:
        row = next(json.loads(l) for l in f if json.loads(l)["traces"][0]["info"].get("patch"))
    trace = row["traces"][0]
    actions = actions_from_trace(trace)
    task = swesmith_task(row["task"]["data"]["instance_id"])
    observations, diff = await replay_swe(task, actions)
    report = compare(actions, observations, trace["info"]["patch"], diff)
    assert report.diff_equal, report
