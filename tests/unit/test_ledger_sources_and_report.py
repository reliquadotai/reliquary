"""The payout ledger's inputs are read, never written, and its report and CLI
put one pass together: what is due that day, what is held, what awaits audit."""

from __future__ import annotations

import asyncio
import gzip
import json
import os
from contextlib import asynccontextmanager
from datetime import date, datetime, timezone
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from reliquary.ledger import obligations as ob
from reliquary.ledger import sources
from reliquary.ledger.cli import ledger_app
from reliquary.ledger.report import daily_report, obligation_rows, render_text
from reliquary.validator import corpus_periods as cp

E = 147.60082398718242
TASK = "order-gen-src"
WORK = 23_817


def _utc(text: str) -> float:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc).timestamp()


def _doc(rewards, **extra):
    return {"task_id": TASK, "job_id": TASK, "work_period": WORK, "rewards_by_hotkey": rewards, "cap": 0.05,
            "budget": 0.05, "max_price_per_token": 1e-8, "tokens": 10**6, **extra}


# -- the chain -------------------------------------------------------------------------------------------------------


def _info(**overrides):
    info = {"hotkeys": ["5Owner", "5Miner", "5Validator", "5OwnerSecond", "5Idle"],
            "coldkeys": ["ck-owner", "ck-m", "ck-v", "ck-owner", "ck-i"],
            "emission": [150.0, 1.5, 20.0, 0.3, 0.0], "dividends": [0.1, 0.0, 0.9, 0.0, 0.0],
            "incentives": [0.95, 0.01, 0.02, 0.002, 0.0], "owner_hotkey": "5Owner", "owner_coldkey": "ck-owner",
            "last_step": 1000, "tempo": 360}
    info.update(overrides)
    return SimpleNamespace(**info)


def test_an_epochs_miner_pay_leaves_the_owners_hotkeys_out_as_burned():
    epoch = sources.epoch_pay_from_info(_info(), block=1000, at=123.0)
    # A miner gets its emission; a validator its incentive at the miners' rate (1.5 / 0.01).
    assert epoch.pay == {"5Miner": 1.5, "5Validator": pytest.approx(0.02 * 150.0)}
    # Every hotkey of the owner's coldkey is the owner: the chain keeps what they are given.
    assert epoch.burned == pytest.approx((0.95 + 0.002) * 150.0)
    assert epoch.pot == pytest.approx(epoch.burned + 1.5 + 3.0)
    assert (epoch.block, epoch.at) == (1000, 123.0)


def test_an_epoch_with_no_miner_incentive_pays_validators_nothing_rather_than_guess():
    epoch = sources.epoch_pay_from_info(_info(emission=[150.0, 0.0, 20.0, 0.0, 0.0],
                                              incentives=[1.0, 0.0, 0.0, 0.0, 0.0]), block=1, at=1.0)
    assert epoch.pay == {} and epoch.burned == 0.0 and epoch.pot is None


class _FakeSubtensor:
    """Epochs every 360 blocks from 640 to 3520, 72 minutes apart; one tempo change."""

    def __init__(self):
        self.steps = [640 + 360 * k for k in range(9)]
        self.calls = []

    def _last_step(self, block):
        return max((s for s in self.steps if s <= block), default=0)

    async def get_current_block(self):
        return 3600

    async def get_metagraph_info(self, netuid, selected_indices=None, block=None, **_):
        self.calls.append(("get_metagraph_info", block))
        step = self._last_step(block)
        return SimpleNamespace(hotkeys=["5Owner", "5M"], coldkeys=["ck-o", "ck-m"], emission=[10.0, float(step)],
                               dividends=[0.0, 0.0], incentives=[0.5, 0.5], owner_hotkey="5Owner",
                               owner_coldkey="ck-o", last_step=step, tempo=360 if step > 1000 else 999,
                               commit_reveal_weights_enabled=True, commit_reveal_period=1)

    async def get_timestamp(self, block=None, **_):
        self.calls.append(("get_timestamp", block))
        return datetime.fromtimestamp(1_700_000_000 + block * 12, tz=timezone.utc)


def test_epochs_are_walked_back_from_the_head_over_the_window_only():
    chain = _FakeSubtensor()
    start, end = 1_700_000_000 + 1000 * 12, 1_700_000_000 + 3000 * 12
    epochs, meta = asyncio.run(sources.read_epoch_pays(chain, 81, start=start, end=end))
    assert [e.block for e in epochs] == [1000, 1360, 1720, 2080, 2440, 2800]
    assert all(start <= e.at < end for e in epochs)
    assert epochs[0].pay == {"5M": 1000.0}
    assert meta["commit_reveal"] is True and meta["commit_reveal_period"] == 1 and meta["tempo"] == 360
    # Only reads were made.
    assert {name for name, _ in chain.calls} == {"get_metagraph_info", "get_timestamp"}


# -- the bucket, read-only -----------------------------------------------------------------------------------------------


class _ReadOnlyClient:
    """An S3 client that lists and gets, and fails the test on anything else."""

    def __init__(self, objects):
        self.objects = objects
        self.calls = []

    def __getattr__(self, name):
        raise AssertionError(f"the ledger called {name} on the bucket")

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        self.calls.append(name)
        objects = self.objects

        class _Paginator:
            def paginate(self, Bucket, Prefix):
                async def pages():
                    yield {"Contents": [{"Key": key, "LastModified": modified}
                                        for key, (modified, _) in sorted(objects.items()) if key.startswith(Prefix)]}

                return pages()

        return _Paginator()

    async def get_object(self, Bucket, Key):
        from botocore.exceptions import ClientError

        self.calls.append(("get_object", Key))
        if Key not in self.objects:
            raise ClientError({"Error": {"Code": "NoSuchKey"}}, "GetObject")
        body = self.objects[Key][1]

        class _Body:
            async def read(self, *_):
                return body

        return {"Body": _Body(), "ETag": "e"}


def _bucket(monkeypatch, objects):
    from reliquary.infrastructure import storage, task_registry_store

    client = _ReadOnlyClient(objects)

    @asynccontextmanager
    async def get_s3_client(**_):
        yield client

    monkeypatch.setattr(storage, "get_s3_client", get_s3_client)
    # The registry store binds the client factory at import.
    monkeypatch.setattr(task_registry_store, "get_s3_client", get_s3_client)

    async def download_json(key, strict=False, **_):
        client.calls.append(("download_json", key))
        return json.loads(gzip.decompress(objects[key][1]))

    monkeypatch.setattr(storage, "download_json", download_json)
    return client


def test_archives_are_listed_with_their_settlement_time_and_read_only(monkeypatch):
    when = datetime(2026, 11, 26, 15, 0, tzinfo=timezone.utc)
    objects = {
        f"reliquary/corpus-periods/{TASK}/{WORK:010d}-{WORK + 1:010d}.json.gz":
            (when, gzip.compress(json.dumps(_doc({"5A": 0.01})).encode())),
        f"reliquary/corpus-periods/{TASK}/{WORK - 900:010d}-{WORK - 899:010d}.json.gz":
            (when, gzip.compress(json.dumps(_doc({"5A": 0.01})).encode())),
        f"reliquary/corpus-periods/{TASK}/notes/readme.txt": (when, b""),
        "reliquary/corpus-periods/stray.json.gz": (when, b""),
    }
    client = _bucket(monkeypatch, objects)
    refs = asyncio.run(sources.BucketArchives().refs(work_from=WORK - 10))
    assert [(r.task_id, r.work_period, r.entry_period) for r in refs] == [(TASK, WORK, WORK + 1)]
    assert refs[0].settled_at == when.timestamp() and refs[0].document["rewards_by_hotkey"] == {"5A": 0.01}
    # One listing and one read: the archive older than the history is never fetched.
    assert [c if isinstance(c, str) else c[0] for c in client.calls] == ["list_objects_v2", "download_json"]


def test_the_registry_is_read_never_written(monkeypatch):
    from reliquary.shared.task_registry import TaskEntry, render_registry

    params = {"start": 0.05, "decay": 0.98, "rounds_per_step": 1, "deadband": 0.0, "snap": 0.0, "floor": 0.05,
              "cap": 0.05, "median_rounds": 1, "last_good_fills": 1, "settlement": cp.SETTLEMENT_PERIOD_EMA,
              "pricing": "auction", **cp.price_params(2.0, E, 1_795_685_015.0, input_alpha_per_mtoken=0.5)}
    entry = TaskEntry(task_id=TASK, profile_id="p", profile_sha256="0" * 64, mechanism="corpus-generation",
                      params=params, status="active", retired_at=None, job_id=TASK)
    body = render_registry({TASK: entry})
    client = _bucket(monkeypatch, {"reliquary/tasks/registry.json": (None, body)})
    ceilings, pay_ceilings, floored = asyncio.run(sources.read_registry_ceilings())
    (ceiling,) = ceilings[TASK]
    assert ceiling.miner_alpha_per_period == E and ceiling.pricing == ob.PRICING_AUCTION
    assert ceiling.max_price_pool_share_per_token == pytest.approx(2.0 / 1e6 / E)
    assert pay_ceilings == {TASK: 0.05} and floored == []
    assert all(c[0] == "get_object" for c in client.calls if not isinstance(c, str))


def test_selection_keeps_recent_work_late_parts_of_older_work_and_whole_histories():
    window = WORK - 320                       # 16 days back
    assert sources.selected(TASK, WORK, WORK + 1, work_from=window)
    # Work older than the window, settled recently: a late part, read.
    assert sources.selected(TASK, window - 400, WORK, work_from=window)
    # Work and settlement both older than the window: not read, unless the task is read in full.
    assert not sources.selected(TASK, window - 400, window - 399, work_from=window)
    assert sources.selected(TASK, window - 400, window - 399, work_from=window, full_tasks=[TASK])
    # A bounded query reads only its own work periods.
    assert not sources.selected(TASK, window - 400, WORK, work_from=WORK, work_to=WORK)


def test_a_late_part_of_work_older_than_the_window_is_read_and_falls_due(monkeypatch):
    when = datetime(2026, 11, 30, 9, 0, tzinfo=timezone.utc)
    old_work = WORK - 400                     # about 20 days before
    late_entry = cp.period_of(when.timestamp()) + 1
    objects = {f"reliquary/corpus-periods/{TASK}/{old_work:010d}-{late_entry:010d}.json.gz":
               (when, gzip.compress(json.dumps(_doc({"5A": 0.01}, work_period=old_work)).encode()))}
    _bucket(monkeypatch, objects)
    refs = asyncio.run(sources.BucketArchives().refs(work_from=WORK - 320))
    assert [(r.work_period, r.entry_period) for r in refs] == [(old_work, late_entry)]
    ceilings = {TASK: [ob.Ceiling(miner_alpha_per_period=E, max_price_pool_share_per_token=1e-8)]}
    (obligation,) = ob.build_ledger(refs, ceilings=ceilings).obligations
    assert obligation.due_at == _utc("2026-11-30T12:00:00")


def test_a_directory_laid_out_like_the_bucket_reads_the_same(tmp_path):
    folder = tmp_path / TASK
    folder.mkdir()
    (folder / f"{WORK:010d}-{WORK + 1:010d}.json").write_text(json.dumps(_doc({"5A": 0.01}, settled_at=5.0)))
    (folder / f"{WORK + 1:010d}-{WORK + 2:010d}.json.gz").write_bytes(
        gzip.compress(json.dumps(_doc({"5B": 0.02}, work_period=WORK + 1)).encode()))
    os.utime(folder / f"{WORK + 1:010d}-{WORK + 2:010d}.json.gz", (7.0, 7.0))
    refs = asyncio.run(sources.DirectoryArchives(tmp_path).refs(work_from=WORK))
    assert [(r.work_period, r.entry_period, r.settled_at) for r in refs] == [(WORK, WORK + 1, 5.0),
                                                                            (WORK + 1, WORK + 2, 7.0)]
    assert asyncio.run(sources.DirectoryArchives(tmp_path).refs(work_from=WORK + 1, work_to=WORK + 1))[0] \
        .document["rewards_by_hotkey"] == {"5B": 0.02}


# -- the public routes -------------------------------------------------------------------------------------------------


def _routes(now):
    return {
        "https://api.test/corpus/pay": {
            "miner_alpha_per_period": E,
            "jobs": [{"job_id": TASK, "task_id": TASK, "cap": 0.05, "max_price_alpha_per_mtoken": 2.0,
                      "unsettled_periods": [{"work_period": WORK - 1, "verified_tokens": 5_000_000},
                                            {"work_period": WORK, "verified_tokens": 10}],
                      "settlement_lag_seconds": 3 * 86_400.0}]},
        "https://api.test/corpus/jobs": {"jobs": [TASK], "orders": [
            {"job_id": TASK, "pricing": "auction"}, {"job_id": "order-eval-9b", "pricing": "auction"}]},
        f"https://api.test/corpus/jobs/{TASK}/status": {"submissions_accepted": 10, "audited": 10, "pay": {
            "price_ceiling": {"max_price_alpha_per_mtoken": 2.0, "max_price_pool_share_per_token": 2.0 / 1e6 / E,
                              "miner_alpha_per_period": E, "set_at": 1.0}}},
        "https://api.test/corpus/jobs/order-eval-9b/status": {"submissions_accepted": 44, "audited": 0,
                                                              "awaiting_audit": True, "held": "no_gateway"},
    }


def test_work_awaiting_audit_comes_from_the_public_routes():
    now = cp.period_end(WORK) + 3600
    routes = _routes(now)
    pending, pricing = sources.api_pending("https://api.test/", now=now, settled={(TASK, WORK)},
                                           fetch=routes.get, first_seen={"order-eval-9b": now - 9 * 86_400})
    reasons = sorted((p.reason, p.work_period) for p in pending)
    assert reasons == [("awaiting_audit", None), ("settled_in_part", WORK), ("undecided", cp.period_of(now - 3 * 86_400)),
                       ("unsettled", WORK - 1)]
    unsettled = next(p for p in pending if p.reason == "unsettled")
    # An upper bound at the ceiling, never past the cap in alpha.
    assert unsettled.ceiling_alpha == pytest.approx(min(5.0 * 2.0, 0.05 * E))
    nine_b = next(p for p in pending if p.reason == "awaiting_audit")
    assert nine_b.submissions_awaiting == 44 and nine_b.held == "no_gateway"
    assert nine_b.escalate_at(now, ob.PayTerms()) is True
    assert pricing == {TASK: "auction", "order-eval-9b": "auction"}


def test_price_ceilings_come_from_the_public_job_status():
    routes = _routes(0.0)
    found = sources.api_ceilings("https://api.test", [TASK, "order-gone"], pricing={TASK: "auction"},
                                 fetch=routes.get)
    (ceiling,) = found[TASK]
    assert ceiling.miner_alpha_per_period == E and ceiling.pricing == "auction" and "order-gone" not in found


def test_only_http_urls_are_fetched():
    with pytest.raises(ValueError):
        sources.http_json("file:///etc/passwd")


# -- the report and the CLI ----------------------------------------------------------------------------------------------


def _ledger():
    refs = []
    for i, (hotkeys, settled) in enumerate([({"5A": 0.01, "5B": 0.02}, "2026-11-26T15:00:00"),
                                             ({"5A": 0.03}, "2026-11-25T10:00:00"),
                                             ({"5C": 0.004}, "2026-11-28T13:00:00")]):
        work = WORK - 20 * i
        refs.append(ob.ArchiveRef(task_id=TASK, work_period=work, entry_period=work + 1,
                                  document=_doc(hotkeys, work_period=work), settled_at=_utc(settled)))
    ceilings = {TASK: [ob.Ceiling(miner_alpha_per_period=E, max_price_pool_share_per_token=1e-8)]}
    return ob.build_ledger(refs, ceilings=ceilings), refs


def test_the_report_of_a_pass_lists_what_falls_due_that_day():
    ledger, refs = _ledger()
    terms = ob.PayTerms()
    now = _utc("2026-11-28T12:30:00")
    report = daily_report(ledger, now=now, pay_day=date(2026, 11, 28), terms=terms, rates=ob.RatePolicy(),
                          pending=ob.pending_from_archives(refs, now))
    pay = report["pass"]
    # Work of 11-26 is due on 11-28; work of 11-25 was due on 11-27; the archive settled at 13:00 is not yet seen.
    assert pay["work_days"] == ["2026-11-26"] and pay["preview"] is False
    assert {row["hotkey"] for row in pay["by_hotkey"]} == {"5A", "5B"}
    assert pay["rao"] == ob.amount_rao(0.01, E) + ob.amount_rao(0.02, E)
    assert report["states"]["due"]["obligations"] == 3 and report["states"]["held"]["obligations"] == 0
    assert report["checks"] == {"obligation_ids_unique": True, "states_beyond_due": 0}
    assert report["mode"] == "ledger"
    # The period that ended on 11-24 and settled on 11-28 at 13:00 was still awaiting audit at 12:30.
    assert [item["reason"] for item in report["pending_audit"]["items"]] == ["settled_later"]
    text = render_text(report)
    assert "nothing is paid here" in text and "5B" in text
    rows = obligation_rows(ledger.obligations, now)
    assert len(rows) == 3 and {row["state"] for row in rows} == {"due"}


def test_a_report_before_the_pass_is_a_preview_and_a_late_settlement_joins_the_next_pass():
    ledger, _ = _ledger()
    terms = ob.PayTerms()
    preview = daily_report(ledger, now=_utc("2026-11-28T09:00:00"), pay_day=date(2026, 11, 28), terms=terms,
                           rates=ob.RatePolicy())
    assert preview["pass"]["preview"] is True and preview["pass"]["obligations"] == 2
    later = daily_report(ledger, now=_utc("2026-11-29T12:00:00"), pay_day=date(2026, 11, 29), terms=terms,
                         rates=ob.RatePolicy())
    # Work of 11-24 settled on 11-28 at 13:00, after that day's pass: it is due on 11-29, a late one.
    assert later["pass"]["work_days"] == ["2026-11-24"] and later["pass"]["late_obligations"] == 1


def test_the_cli_builds_a_report_from_a_directory_without_network(tmp_path):
    folder = tmp_path / "archives" / TASK
    folder.mkdir(parents=True)
    for i in range(3):
        work = WORK - i
        doc = _doc({"5A": 0.01 * (i + 1), "5B": 0.002}, work_period=work, settled_at=cp.period_end(work) + 300)
        (folder / f"{work:010d}-{work + 1:010d}.json").write_text(json.dumps(doc))
    out = tmp_path / "out"
    result = CliRunner().invoke(ledger_app, [
        "report", "--archives", str(tmp_path / "archives"), "--prices", "none", "--rate", str(E),
        "--no-pending", "--no-chain", "--as-of", "2026-11-28T12:05:00", "--out", str(out)])
    assert result.exit_code == 0, result.output
    assert "Pass of 2026-11-28" in result.output
    report = json.loads((out / "report-2026-11-28.json").read_text())
    assert report["pass"]["obligations"] == 6
    assert report["rate"] == {"mode": "fixed", "alpha_per_period": E, "fallbacks": []}
    assert report["pass"]["by_rail"]["native"]["paid_by"] == "the weights"
    lines = (out / "obligations-2026-11-28.jsonl").read_text().splitlines()
    assert len(lines) == 6 and {json.loads(line)["state"] for line in lines} == {"due"}
    again = CliRunner().invoke(ledger_app, [
        "report", "--archives", str(tmp_path / "archives"), "--prices", "none", "--rate", str(E),
        "--no-pending", "--no-chain", "--as-of", "2026-11-28T12:05:00", "--json"])
    assert again.exit_code == 0
    assert json.loads(again.output)["pass"] == report["pass"]


def test_the_cli_refuses_bad_settings(tmp_path):
    for args in (["--rate", "lots"], ["--prices", "chain"], ["--pay-at", "25:00"], ["--order-budget", "x"]):
        result = CliRunner().invoke(ledger_app, ["report", "--archives", str(tmp_path), "--no-pending",
                                                 "--no-chain", *args])
        assert result.exit_code != 0, args
