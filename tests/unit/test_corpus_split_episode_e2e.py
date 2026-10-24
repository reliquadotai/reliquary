"""Real processes, real time: the split validator serving an episode job in its
front beside a single-turn job judged in its own process. One honest
trajectory is admitted through the front's route, audited through the GPU
process (its spans cross the socket), graded and replayed by two fake grade
executors over ``/corpus/internal/grade/*``, and paid by the settler once
graded. No GPU, no Docker: the harness's on-disk bucket and fakes."""

from __future__ import annotations

import json

import httpx
import pytest

from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from reliquary.protocol.signatures import corpus_submission_id

from tests.unit import corpus_split_episode_fakes as ep
from tests.unit import corpus_split_harness as h

MATH = "math-v1"
HOTKEY = h.HOTKEYS[1]
AUDITOR = {"accept_slack_seconds": 2.0, "rescan_every_seconds": 1.0}


@pytest.fixture
def bucket(tmp_path):
    root = tmp_path / "bucket"
    root.mkdir()
    log = tmp_path / "children.log"
    with h.environment(h.harness_env(root, CORPUS_HARNESS_LOG=log, CORPUS_HARNESS_BEACON="const",
                                     CORPUS_HARNESS_GPU_TPS=20000)):
        h.seed_bucket(root, [MATH], hotkeys=h.HOTKEYS[:4])
        ep.seed_episode(root, hotkeys=h.HOTKEYS[:4])
        yield root, log


def test_an_honest_trajectory_is_admitted_audited_graded_and_paid(bucket):
    root, log = bucket
    served = [(h.entry("corpus-math", MATH, 0.1), 0.1),
              # Every trajectory audited (q=1), so the GPU path is the one taken.
              (h.entry(ep.EPISODE_TASK, ep.EPISODE_JOB, 0.05, audit_q=1.0,
                       audit_hold_seconds=5.0), 0.05)]
    port = h.free_port()
    base = f"http://127.0.0.1:{port}"
    spec = h.split_spec(root, served, [[MATH]], port=port, settle_every_seconds=3.0,
                        auditor_kwargs=AUDITOR)
    spec.child_init = "tests.unit.corpus_split_episode_fakes:install"
    try:
        with h.SupervisorThread(spec) as sup:
            import asyncio

            asyncio.run(h.wait_http(base, timeout=120))
            with ep.FakeExecutors(base) as executors:
                answer = httpx.post(f"{base}/corpus/jobs/{ep.EPISODE_JOB}/submit",
                                    json=ep.submission(HOTKEY), timeout=60.0)
                assert answer.status_code == 200, answer.text
                assert answer.json().get("accepted"), answer.json()
                sid = corpus_submission_id(
                    CorpusSubmissionRequest.model_validate(ep.submission(HOTKEY)))

                # Audited, through the GPU process, with its spans.
                ep.wait(lambda: sid in h.listed(root, ep.EPISODE_JOB, "verdicts"), 120,
                        "the trajectory's verdict")
                verdict = h.listed(root, ep.EPISODE_JOB, "verdicts")[sid]
                assert verdict["passed"] and verdict["audited"], verdict
                spans = [json.loads(line)
                         for line in (root / ep.SPANS_LOG).read_text().splitlines()]
                assert spans and all(len(s["spans"]) == 3 for s in spans), spans

                # Graded and replayed by the executors, through the front's grade routes.
                ep.wait(lambda: sid in h.listed(root, ep.EPISODE_JOB, "grades"), 120,
                        "the trajectory's grade")
                grade = h.listed(root, ep.EPISODE_JOB, "grades")[sid]
                # A native decision may be written before its result receipt
                # finishes persisting and the executor sees the HTTP answer.
                ep.wait(lambda: {"grade", "replay"} <= {
                    mode for _, mode, outcome in executors.answered if outcome == "accepted"},
                    120, "the grading and replay result acknowledgements")
                modes = {mode for _, mode, outcome in executors.answered if outcome == "accepted"}
                assert {"grade", "replay"} <= modes, executors.answered

                # Paid once graded: settled, and the window archive rewards the miner.
                ep.wait(lambda: sid in (h.settlement(root, ep.EPISODE_JOB).get("settled") or []),
                        120, "the trajectory's settlement")
                rewards = {}
                for path in (root / "archives").glob(f"{ep.EPISODE_TASK}-*.json"):
                    rewards.update(json.loads(path.read_text())["rewards_by_hotkey"])
                assert rewards.get(HOTKEY, 0) > 0, rewards
                assert not h.listed(root, ep.EPISODE_JOB, "voided")
            # No child restarted: nothing crashed on the way.
            assert all(child.starts == 1 for child in sup.children.values()), \
                {name: c.starts for name, c in sup.children.items()}
    except AssertionError:
        print(log.read_text()[-20000:] if log.exists() else "(no child log)")
        raise
    assert grade, grade
