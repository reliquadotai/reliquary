"""The weight setter no longer replays window archives of corpus tasks (a corpus
task is paid by period only), and RL's replay is unchanged to the bit."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.shared.task_registry import (
    MECHANISM_CORPUS_GENERATION,
    MECHANISM_RL_DISCOVERED_PRICE,
    TaskEntry,
)

GENESIS = 1_692_803_367.0


def _entry(task_id, mechanism, cap, **extra):
    params = {"start": cap, "decay": 0.99, "rounds_per_step": 1000, "deadband": 0.8,
              "snap": 1.2, "floor": cap if mechanism == MECHANISM_CORPUS_GENERATION else 0.0,
              "cap": cap, "median_rounds": 4800, "last_good_fills": 50, **extra}
    return TaskEntry(task_id=task_id, profile_id="p", profile_sha256="a" * 64,
                     mechanism=mechanism, params=params, status="active", retired_at=None,
                     job_id=task_id if mechanism == MECHANISM_CORPUS_GENERATION else None)


# RL windows 100..139, a window-settled corpus task 120..151 (past RL: it moves
# the shared horizon, as it did in prod), a period task beside them.
RL_WINDOWS = list(range(100, 140))
CORPUS_WINDOWS = list(range(120, 152))


def _rl_rewards(w):
    hotkeys = ["rl-a", "rl-b", "rl-c", "rl-d", "rl-tiny"]
    weights = [(w * 7 + i * 13) % 17 + 1 for i in range(len(hotkeys))]
    weights[-1] = 1 if w % 3 else 0
    total = sum(weights)
    return {hk: 0.75 * x / total for hk, x in zip(hotkeys, weights) if x}


def _corpus_rewards(w):
    return {"cx-a": 0.05 * (w % 4 + 1) / 10, "cx-b": 0.05 * (10 - w % 4 - 1) / 10}


DECLARED = {
    "default": _entry("default", MECHANISM_RL_DISCOVERED_PRICE, 0.75),
    "corpus-x": _entry("corpus-x", MECHANISM_CORPUS_GENERATION, 0.05),
    "corpus-p": _entry("corpus-p", MECHANISM_CORPUS_GENERATION, 0.1,
                       settlement="period-ema-v1"),
}
PERIOD_DOCS = {("corpus-p", 30, 31): {"rewards_by_hotkey": {"p-a": 0.06, "p-b": 0.04},
                                      "cap": 0.1}}


class _PeriodArchives:
    async def list(self, task_id):
        return sorted((w, e) for (t, w, e) in PERIOD_DOCS if t == task_id)

    async def read(self, task_id, work, entry):
        return PERIOD_DOCS.get((task_id, work, entry))


def _run(monkeypatch, declared):
    import reliquary.validator.weight_only as wov_mod
    from reliquary.validator.weight_only import WeightOnlyValidator

    windows = {"default": RL_WINDOWS, "corpus-x": CORPUS_WINDOWS}
    asked = {}

    async def list_task_ids(strict=False, **kw):
        return sorted(windows)

    async def list_all_window_keys(*, task_id=None, strict=False, **kw):
        return list(windows.get(task_id, []))

    async def list_recent(current_window, n, *, task_id=None, **kw):
        asked[task_id] = (current_window, n)
        rewards = _rl_rewards if task_id == "default" else _corpus_rewards
        return [{"window_start": w, "rewards_by_hotkey": rewards(w)}
                for w in windows[task_id] if current_window - n <= w < current_window]

    async def registry():
        return dict(declared), "etag"

    period_weights = WeightOnlyValidator._period_weights

    async def periods(declared):
        return await period_weights(declared, archives=_PeriodArchives(),
                                    now=GENESIS + 31 * 4320 + 5, genesis=GENESIS)

    async def subtensor():
        return object()

    async def close(_):
        return None

    monkeypatch.setattr(wov_mod.storage, "list_task_ids", list_task_ids)
    monkeypatch.setattr(wov_mod.storage, "list_all_window_keys", list_all_window_keys)
    monkeypatch.setattr(wov_mod.storage, "list_recent_datasets", list_recent)
    monkeypatch.setattr(wov_mod, "read_registry", registry)
    monkeypatch.setattr(wov_mod.chain, "get_subtensor", subtensor)
    monkeypatch.setattr(wov_mod.chain, "close_subtensor", close)
    wov = WeightOnlyValidator.__new__(WeightOnlyValidator)
    wov._active_submit_epoch = None
    monkeypatch.setattr(wov, "_period_weights", periods, raising=False)
    submitted = {}

    async def submit(sub, weights):
        submitted.update(weights)
        return True

    wov._submit_weights = submit
    result = asyncio.run(wov.submit_once())
    return result, submitted, asked


# Computed by origin/main (d30b9fd7) on this exact fixture, before corpus window
# archives stopped being replayed: RL's weights must not move by one bit.
RL_GOLDEN = {
    "rl-a": 0.12171095510953138,
    "rl-b": 0.1215339831296693,
    "rl-c": 0.12791327852303339,
    "rl-d": 0.12362788089543264,
    "rl-tiny": 0.008336802268515369,
}


def test_rl_weights_are_unchanged_to_the_bit(monkeypatch):
    result, submitted, asked = _run(monkeypatch, DECLARED)
    assert result is True
    rl = {hk: v for hk, v in submitted.items() if hk.startswith("rl-")}
    assert rl == RL_GOLDEN
    # The shared horizon still spans every task's windows, as before.
    assert asked["default"] == (152, 216)


def test_corpus_window_archives_are_not_replayed(monkeypatch):
    result, submitted, asked = _run(monkeypatch, DECLARED)
    assert result is True
    assert not any(hk.startswith("cx-") for hk in submitted)
    assert "corpus-x" not in asked  # not even read
    # The period task is paid as before.
    assert submitted["p-a"] == pytest.approx(2 / 7 * 0.06)
    assert submitted["p-b"] == pytest.approx(2 / 7 * 0.04)


def test_a_retired_window_corpus_task_does_not_make_the_setter_abstain(monkeypatch):
    from dataclasses import replace

    declared = {**DECLARED, "corpus-x": replace(DECLARED["corpus-x"], status="retired",
                                                params={**DECLARED["corpus-x"].params,
                                                        "cap": 0.0, "floor": 0.0})}
    result, submitted, _ = _run(monkeypatch, declared)
    assert result is True
    assert {hk for hk in submitted if hk.startswith("rl-")} == set(RL_GOLDEN)


def test_an_undeclared_task_with_window_archives_still_abstains(monkeypatch):
    result, submitted, _ = _run(monkeypatch, {"default": DECLARED["default"]})
    assert result is False and submitted == {}
