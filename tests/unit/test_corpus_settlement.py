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


def test_the_stall_clock_persists_so_the_corpus_advances_after_one_more_rl_seal():
    # Reproduces the wedge: other_max_seen/_at must be CAS-written even on a
    # call that settles nothing, or every later call sees "other_max changed
    # just now" forever and the corpus never advances again.
    records = _Records({"1" * 64: _v("A", 10)})
    archives = _Archives(46000)
    assert asyncio.run(_settler(records, archives, now=0).settle_once()) == 46000
    records.verdicts["2" * 64] = _v("A", 10)
    # RL is already stalled 200s past t=0 with stall_seconds=100: advance alone.
    assert asyncio.run(_settler(records, archives, now=200).settle_once()) == 46001
    records.verdicts["3" * 64] = _v("A", 10)
    archives.other = 46001  # RL seals once more, then stops forever
    # other_max just changed (from the settler's point of view): too soon to
    # call it a stall.
    assert asyncio.run(_settler(records, archives, now=300).settle_once()) is None
    # STALL has now elapsed since the settler first observed other_max=46001
    # (at t=300): it must advance, not stay wedged on None forever.
    assert asyncio.run(_settler(records, archives, now=1000).settle_once()) == 46002
    assert archives.written[46002]["rewards_by_hotkey"] == pytest.approx({"A": 0.1})


def test_an_all_failed_period_writes_no_archive_but_marks_ids_settled():
    records = _Records({"1" * 64: _v("C", 9, ok=False)})
    archives = _Archives(46000)
    assert asyncio.run(_settler(records, archives).settle_once()) is None
    assert archives.written == {}
    assert records.state["settled"] == ["1" * 64]
    assert records.state["last_window"] is None
    assert records.state["pending"] is None

    records.verdicts["2" * 64] = _v("A", 10)
    assert asyncio.run(_settler(records, archives).settle_once()) == 46000
    assert archives.written[46000]["rewards_by_hotkey"] == pytest.approx({"A": 0.1})
    assert sorted(records.state["settled"]) == ["1" * 64, "2" * 64]
    assert records.state["last_window"] == 46000


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


# --- final review, finding 1: stall mode advances at the RL cadence, not the settle cadence ---


class _FailingArchives(_Archives):
    """Crashes on the archive write itself, so a pending entry is left with no archive."""

    def __init__(self, other_max):
        super().__init__(other_max)
        self.fail = False

    async def write(self, task_id, window, data):
        if self.fail:
            raise OSError("crash before the archive landed")
        await super().write(task_id, window, data)


def test_stall_mode_advances_at_most_once_per_rl_window_not_once_per_settle_call():
    # RL sealed 46000 long ago and the corpus already joined it; then ten
    # settle calls 60 s apart, each with a fresh verdict, as the loop makes.
    records = _Records({"0" * 64: _v("A", 10)})
    archives = _Archives(46000)
    assert asyncio.run(_settler(records, archives, now=0).settle_once()) == 46000
    start = 10 * STALL
    for i in range(10):
        records.verdicts[f"{i + 1:064d}"] = _v("A", 10)
        asyncio.run(_settler(records, archives, now=start + 60 * i).settle_once())
    assert max(archives.written) - 46000 <= 1


def test_stall_mode_advances_again_once_an_rl_window_of_wall_time_has_passed():
    from reliquary.validator.corpus_settlement import RL_WINDOW_SECONDS

    records = _Records({"0" * 64: _v("A", 10)})
    archives = _Archives(46000)
    asyncio.run(_settler(records, archives, now=0).settle_once())
    start = 10 * STALL
    records.verdicts["1" * 64] = _v("A", 10)
    assert asyncio.run(_settler(records, archives, now=start).settle_once()) == 46001
    records.verdicts["2" * 64] = _v("A", 10)
    assert asyncio.run(_settler(records, archives, now=start + RL_WINDOW_SECONDS - 1).settle_once()) is None
    assert asyncio.run(_settler(records, archives, now=start + RL_WINDOW_SECONDS).settle_once()) == 46002


def test_a_pending_stall_entry_finished_after_rl_revived_does_not_lead_the_horizon():
    from reliquary.validator.corpus_settlement import RL_WINDOW_SECONDS

    records = _Records({"0" * 64: _v("A", 10)})
    archives = _FailingArchives(46000)
    asyncio.run(_settler(records, archives, now=0).settle_once())
    # Stall: the corpus advances alone to 46002, one RL window apart.
    t = 10 * STALL
    for sid in ("1", "2"):
        records.verdicts[sid * 64] = _v("A", 10)
        asyncio.run(_settler(records, archives, now=t).settle_once())
        t += RL_WINDOW_SECONDS
    assert records.state["last_window"] == 46002
    # The next stall settlement crashes after its pending write, before any archive.
    records.verdicts["3" * 64] = _v("B", 10)
    archives.fail = True
    with pytest.raises(OSError):
        asyncio.run(_settler(records, archives, now=t).settle_once())
    assert records.state["pending"]["window"] == 46003
    archives.fail = False
    # RL revives and seals 46001: finishing the pending entry now would write
    # 46003 above a live task's horizon.
    archives.other = 46001
    assert asyncio.run(_settler(records, archives, now=t + 10).settle_once()) is None
    assert 46003 not in archives.written
    assert records.state["pending"]["window"] == 46003
    # Once RL reaches the pending window, it is finished there, once.
    archives.other = 46003
    assert asyncio.run(_settler(records, archives, now=t + 20).settle_once()) == 46003
    assert archives.written[46003]["rewards_by_hotkey"] == pytest.approx({"B": 0.1})
    assert "3" * 64 in records.state["settled"] and records.state["pending"] is None


def test_a_delayed_finish_spaces_the_next_lone_advance_from_the_finish_time():
    from reliquary.validator.corpus_settlement import RL_WINDOW_SECONDS

    records = _Records({"0" * 64: _v("A", 10)})
    archives = _FailingArchives(46000)
    asyncio.run(_settler(records, archives, now=0).settle_once())
    t = 10 * STALL
    records.verdicts["1" * 64] = _v("A", 10)
    archives.fail = True  # the lone advance to 46001 crashes before its archive
    with pytest.raises(OSError):
        asyncio.run(_settler(records, archives, now=t).settle_once())
    archives.fail = False
    # Finished two RL windows later, still in the same stall.
    late = t + 2 * RL_WINDOW_SECONDS
    assert asyncio.run(_settler(records, archives, now=late).settle_once()) == 46001
    records.verdicts["2" * 64] = _v("A", 10)
    assert asyncio.run(_settler(records, archives, now=late + 60).settle_once()) is None
    assert max(archives.written) == 46001
