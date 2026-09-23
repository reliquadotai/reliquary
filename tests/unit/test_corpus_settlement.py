"""The corpus task pays its cap by verified tokens and never moves another task."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from reliquary.validator.corpus_settlement import CorpusSettler, R2Archives, choose_window, rewards_for
from reliquary.validator.weight_only import WeightOnlyValidator

STALL = 100.0


def test_rewards_split_the_cap_by_passed_tokens():
    verdicts = [
        {"hotkey": "A", "token_count": 300, "passed": True},
        {"hotkey": "B", "token_count": 100, "passed": True},
        {"hotkey": "C", "token_count": 900, "passed": False},
    ]
    assert rewards_for(verdicts, 0.1) == pytest.approx({"A": 0.075, "B": 0.025})


def test_no_passed_token_pays_nobody():
    assert rewards_for([{"hotkey": "C", "token_count": 9, "passed": False}], 0.1) == {}


def test_a_live_other_task_bounds_the_index():
    # RL at 46000, corpus already settled there: wait for RL to seal again.
    assert choose_window(last_window=46000, other_max=46000, other_max_seen_at=0.0, now=10.0, stall_seconds=STALL) is None
    assert choose_window(last_window=46000, other_max=46001, other_max_seen_at=10.0, now=10.0, stall_seconds=STALL) == 46001


def test_the_first_settlement_joins_the_other_tasks_horizon():
    assert choose_window(last_window=None, other_max=46000, other_max_seen_at=0.0, now=0.0, stall_seconds=STALL) == 46000


def test_an_idle_other_task_lets_the_corpus_advance_alone():
    assert choose_window(last_window=46000, other_max=46000, other_max_seen_at=0.0, now=STALL + 1, stall_seconds=STALL) == 46001


def test_alone_the_corpus_counts_from_zero():
    assert choose_window(last_window=None, other_max=None, other_max_seen_at=None, now=0.0, stall_seconds=STALL) == 0
    assert choose_window(last_window=4, other_max=None, other_max_seen_at=None, now=0.0, stall_seconds=STALL) == 5


class _Records:
    def __init__(self, verdicts):
        self.verdicts = verdicts
        self.state, self.etag = {}, None
        self.fail_state_writes_after = None

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts[sid]

    async def read_settlement(self, job_id):
        return dict(self.state), self.etag

    async def write_settlement(self, job_id, state, etag):
        if self.fail_state_writes_after == 0:
            raise OSError("crash")
        if self.fail_state_writes_after is not None:
            self.fail_state_writes_after -= 1
        assert etag == self.etag
        self.state, self.etag = dict(state), f"e{len(str(state))}"
        return self.etag


class _Archives:
    def __init__(self, other_max):
        self.other = other_max
        self.written = {}

    async def other_max(self, task_id):
        return self.other

    async def write(self, task_id, window, data):
        self.written[window] = data


def _settler(records, archives, now=0.0):
    return CorpusSettler(task_id="corpus-math", job_id="math-v1", cap=0.1,
                         records=records, archives=archives, stall_seconds=STALL,
                         clock=lambda: now)


def _v(hk, n, ok=True):
    return {"hotkey": hk, "token_count": n, "passed": ok}


def test_a_settlement_writes_one_archive_and_marks_its_verdicts():
    records = _Records({"1" * 64: _v("A", 10), "2" * 64: _v("B", 30)})
    archives = _Archives(46000)
    assert asyncio.run(_settler(records, archives).settle_once()) == 46000
    archive = archives.written[46000]
    assert archive["window_start"] == 46000 and archive["window_status"] == "completed"
    assert archive["rewards_by_hotkey"] == pytest.approx({"A": 0.025, "B": 0.075})
    assert sorted(records.state["settled"]) == ["1" * 64, "2" * 64]
    assert records.state["pending"] is None


def test_settled_verdicts_are_never_paid_again():
    records = _Records({"1" * 64: _v("A", 10)})
    archives = _Archives(46000)
    asyncio.run(_settler(records, archives).settle_once())
    archives.other = 46001
    assert asyncio.run(_settler(records, archives).settle_once()) is None
    assert list(archives.written) == [46000]


def test_a_crash_after_the_archive_rewrites_the_same_archive_not_a_second_one():
    records = _Records({"1" * 64: _v("A", 10)})
    archives = _Archives(46000)
    records.fail_state_writes_after = 1  # the pending write lands, the final one crashes
    with pytest.raises(OSError):
        asyncio.run(_settler(records, archives).settle_once())
    records.fail_state_writes_after = None
    archives.other = 46005
    assert asyncio.run(_settler(records, archives).settle_once()) == 46000
    assert list(archives.written) == [46000]
    assert records.state["settled"] == ["1" * 64]


def test_rl_weights_are_identical_with_and_without_the_corpus_task():
    rl = [{"task_id": "default", "window_start": w, "window_status": "completed",
           "rewards_by_hotkey": {"R1": 0.6, "R2": 0.3}} for w in range(45800, 46001)]
    corpus = [{"task_id": "corpus-math", "window_start": w, "window_status": "completed",
               "rewards_by_hotkey": {"C1": 0.1}} for w in range(45990, 46001)]
    caps = {"default": 0.9, "corpus-math": 0.1}
    alone = WeightOnlyValidator._replay_ema(rl, caps=caps)
    both = WeightOnlyValidator._replay_ema(sorted(rl + corpus, key=lambda r: (r["window_start"], r["task_id"])), caps=caps)
    assert {k: both[k] for k in alone} == pytest.approx(alone)
    assert both["C1"] > 0


def test_r2archives_other_max_excludes_its_own_task_and_is_none_when_no_other_task_has_windows():
    async def fake_list_task_ids(*, strict=False, **kw):
        return ["corpus-math", "default"]

    async def fake_list_all_window_keys(*, strict=False, task_id=None, **kw):
        assert task_id != "corpus-math"
        return {"default": [46000, 46001]}.get(task_id, [])

    with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=fake_list_task_ids)), \
         patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=fake_list_all_window_keys)):
        assert asyncio.run(R2Archives().other_max("corpus-math")) == 46001

    async def fake_list_task_ids_alone(*, strict=False, **kw):
        return ["corpus-math"]

    async def fake_list_all_window_keys_alone(*, strict=False, task_id=None, **kw):
        return []

    with patch("reliquary.infrastructure.storage.list_task_ids", AsyncMock(side_effect=fake_list_task_ids_alone)), \
         patch("reliquary.infrastructure.storage.list_all_window_keys", AsyncMock(side_effect=fake_list_all_window_keys_alone)):
        assert asyncio.run(R2Archives().other_max("corpus-math")) is None
