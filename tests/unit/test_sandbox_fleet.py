"""Which machines may get sessions: fresh signed heartbeats, the directory's address,
images, env package, tools, caps and free slots; stale machines are drained."""

import asyncio
import json

import pytest

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox import fleet as fleet_module  # noqa: E402
from reliquary.sandbox.fleet import Fleet, Placement  # noqa: E402
from tests.unit.sandbox_fixtures import (  # noqa: E402
    ADDRESS, BUDGETS, ENV, ENV_PACKAGE, IMAGE, MACHINE, NOW, capacity_report, machine_document,
    signer,
)

PICK = dict(image=IMAGE, env=ENV, env_package=ENV_PACKAGE, budgets=BUDGETS, validity_s=4500)


class Clock:
    def __init__(self, now=NOW):
        self.now = now

    def __call__(self):
        return self.now


def make(tmp_path, documents=None, reports=None, **kw):
    machine = signer(tmp_path, "m", "k1")
    documents = documents if documents is not None else [machine_document(machine)]
    served = {} if reports is None else reports
    clock = kw.pop("clock", Clock())

    async def read_documents():
        return documents

    async def fetch(address):
        return served.get(address)

    fleet = Fleet(read_documents=read_documents, fetch_report=fetch, clock=clock, **kw)
    asyncio.run(fleet.refresh_directory())
    return fleet, machine, served, clock


def test_a_fresh_signed_heartbeat_makes_a_machine_eligible(tmp_path):
    fleet, machine, served, clock = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) == Placement(MACHINE, ADDRESS)


def test_heartbeat_valid_always_gets_now_and_a_max_age(tmp_path, monkeypatch):
    seen = []
    real = fleet_module.heartbeat_valid

    def spy(report, directory, **kwargs):
        seen.append(kwargs)
        return real(report, directory, **kwargs)

    monkeypatch.setattr(fleet_module, "heartbeat_valid", spy)
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    assert seen == [{"now": NOW, "max_age_s": fleet_module.HEARTBEAT_MAX_AGE_S}]


@pytest.mark.parametrize("change", [
    dict(address="http://10.9.9.9:8080"),          # a report naming another address
    dict(machine_id="machine-2"),
    dict(at=NOW - 31),                             # stale
    dict(at=NOW + 31),                             # from the future
])
def test_a_report_the_validator_cannot_trust_is_refused(tmp_path, change):
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, **{"at": NOW, **change})
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) is None


def test_a_report_signed_by_another_key_is_refused(tmp_path):
    fleet, _, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(signer(tmp_path, "x", "k1"), at=NOW)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) is None


@pytest.mark.parametrize("report,pick", [
    (dict(images=("other@sha256:" + "b" * 64,)), {}),
    (dict(env_packages={ENV: "reliquary-swe==9"}), {}),
    (dict(tools_version="reliquary-tools/2"), {}),
    (dict(free=0), {}),
    ({}, dict(budgets={**BUDGETS, "max_calls": 600})),
    ({}, dict(validity_s=8 * 86400)),
])
def test_a_machine_that_cannot_serve_the_session_is_skipped(tmp_path, report, pick):
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW, **report)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **{**PICK, **pick}) is None


def test_issued_tokens_count_against_free_slots_until_a_newer_report(tmp_path):
    fleet, machine, served, clock = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW, free=1)
    asyncio.run(fleet.poll_once())
    fleet.note_issued(MACHINE, NOW)
    assert fleet.pick(now=NOW, **PICK) is None
    clock.now = NOW + 5
    served[ADDRESS] = capacity_report(machine, at=NOW + 5, free=1)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW + 5, **PICK) is not None


def test_the_least_loaded_machine_is_picked(tmp_path):
    one, two = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    documents = [machine_document(one, machine_id="m-a", address="http://a:1"),
                 machine_document(two, machine_id="m-b", address="http://b:1")]
    fleet, _, served, _ = make(tmp_path, documents=documents)
    served["http://a:1"] = capacity_report(one, at=NOW, machine_id="m-a", address="http://a:1", free=1)
    served["http://b:1"] = capacity_report(two, at=NOW, machine_id="m-b", address="http://b:1", free=3)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) == Placement("m-b", "http://b:1")


def test_draining_and_revoked_machines_get_no_session(tmp_path):
    for status in ("draining", "revoked"):
        machine = signer(tmp_path, status, "k1")
        fleet, _, served, _ = make(tmp_path, documents=[machine_document(machine, status=status)])
        served[ADDRESS] = capacity_report(machine, at=NOW)
        asyncio.run(fleet.poll_once())
        assert fleet.pick(now=NOW, **PICK) is None


def test_a_silent_machine_is_drained_once_and_comes_back(tmp_path):
    drained = []
    fleet, machine, served, clock = make(tmp_path, on_drained=drained.append)
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    del served[ADDRESS]
    clock.now = NOW + 61
    asyncio.run(fleet.poll_once())
    asyncio.run(fleet.poll_once())
    assert drained == [MACHINE] and MACHINE in fleet.drained
    served[ADDRESS] = capacity_report(machine, at=NOW + 61)
    asyncio.run(fleet.poll_once())
    assert MACHINE not in fleet.drained


def test_heartbeat_summaries_are_written_at_most_once_a_minute(tmp_path):
    written = []

    async def record(machine_id, *, at, summary):
        written.append((machine_id, at))

    fleet, machine, served, clock = make(tmp_path, record_heartbeat=record)
    for step in range(0, 70, 5):
        clock.now = NOW + step
        served[ADDRESS] = capacity_report(machine, at=NOW + step)
        asyncio.run(fleet.poll_once())
    assert [at for _, at in written] == [NOW, NOW + 60]


# --- invariants added on top of the brief (controller adjustments, review focus) ---


def test_capacity_is_fetched_only_at_the_directory_address(tmp_path):
    asked = []
    machine = signer(tmp_path, "m", "k1")

    async def read_documents():
        return [machine_document(machine)]

    async def fetch(address):
        asked.append(address)
        return capacity_report(machine, at=NOW, address="http://evil:1")

    fleet = Fleet(read_documents=read_documents, fetch_report=fetch, clock=Clock())
    asyncio.run(fleet.refresh_directory())
    asyncio.run(fleet.poll_once())
    assert asked == [ADDRESS]
    assert fleet.pick(now=NOW, **PICK) is None


def test_a_trailing_slash_on_the_report_address_is_the_same_address(tmp_path):
    fleet, machine, served, _ = make(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW, address=ADDRESS + "/")
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) == Placement(MACHINE, ADDRESS)


@pytest.mark.parametrize("cap", [8 * 1024**2 + 1, 0])
def test_a_machine_whose_transcript_cap_is_refused_is_never_placed(tmp_path, cap):
    fleet, machine, served, _ = make(tmp_path)
    caps = capacity_report(machine, at=NOW)["document"]["caps"]
    served[ADDRESS] = capacity_report(machine, at=NOW, caps={**caps, "max_transcript_bytes": cap})
    assert fleet.accept_report(MACHINE, served[ADDRESS], NOW) is False
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) is None


def test_a_slow_machine_does_not_stall_the_others(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_module, "FETCH_TIMEOUT_S", 0.2)
    one, two = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    documents = [machine_document(one, machine_id="m-a", address="http://a:1"),
                 machine_document(two, machine_id="m-b", address="http://b:1")]
    reports = {"http://b:1": capacity_report(two, at=NOW, machine_id="m-b", address="http://b:1")}

    async def read_documents():
        return documents

    async def fetch(address):
        if address == "http://a:1":
            await asyncio.sleep(3600)      # hangs forever
        return reports[address]

    async def scenario():
        fleet = Fleet(read_documents=read_documents, fetch_report=fetch, clock=Clock())
        await fleet.refresh_directory()
        loop = asyncio.get_running_loop()
        started = loop.time()
        await fleet.poll_once()
        return fleet, loop.time() - started

    async def bounded():
        return await asyncio.wait_for(scenario(), timeout=5.0)   # a regression fails, never hangs

    fleet, elapsed = asyncio.run(bounded())
    assert elapsed < 2.0
    assert fleet.pick(now=NOW, **PICK) == Placement("m-b", "http://b:1")


def test_a_fetch_that_raises_is_only_that_machines_loss(tmp_path):
    one, two = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    documents = [machine_document(one, machine_id="m-a", address="http://a:1"),
                 machine_document(two, machine_id="m-b", address="http://b:1")]
    fleet, _, served, _ = make(tmp_path, documents=documents)
    served["http://b:1"] = capacity_report(two, at=NOW, machine_id="m-b", address="http://b:1")

    async def fetch(address):
        if address == "http://a:1":
            raise RuntimeError("boom")
        return served[address]

    fleet._fetch = fetch
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) == Placement("m-b", "http://b:1")


def _mock_transport(handler):
    import httpx

    return httpx.MockTransport(handler)


def _streamed(body: bytes):
    """A response body that arrives as a stream, as from a real socket."""
    async def chunks():
        for start in range(0, len(body), 65536):
            yield body[start:start + 65536]
    return chunks()


def test_http_fetch_report_reads_the_capacity_path(tmp_path):
    import httpx

    machine = signer(tmp_path, "m", "k1")
    report = capacity_report(machine, at=NOW)
    seen = []

    def handler(request):
        seen.append(str(request.url))
        return httpx.Response(200, content=_streamed(json.dumps(report).encode()))

    got = asyncio.run(fleet_module.http_fetch_report(ADDRESS, transport=_mock_transport(handler)))
    assert got == report and seen == [ADDRESS + "/capacity"]


def test_http_fetch_report_caps_the_body_it_reads():
    import httpx

    produced = []

    async def endless():
        # Effectively endless, but bounded so a regression fails instead of exhausting RAM.
        for _ in range(4 * fleet_module.MAX_REPORT_BYTES // 65536):
            produced.append(1)
            yield b"[" * 65536

    def handler(request):
        return httpx.Response(200, content=endless())

    got = asyncio.run(fleet_module.http_fetch_report(ADDRESS, transport=_mock_transport(handler)))
    assert got is None
    assert len(produced) * 65536 <= fleet_module.MAX_REPORT_BYTES + 2 * 65536


def test_http_fetch_report_refuses_an_announced_oversized_body():
    import httpx

    def handler(request):
        return httpx.Response(200, headers={"content-length": str(fleet_module.MAX_REPORT_BYTES + 1)},
                              content=_streamed(b"{}"))

    assert asyncio.run(fleet_module.http_fetch_report(
        ADDRESS, transport=_mock_transport(handler))) is None


@pytest.mark.parametrize("response", [
    dict(status_code=500, body=b"{}"),
    dict(status_code=302, headers={"location": "http://evil:1/capacity"}, body=b""),
    dict(status_code=200, body=b"not json"),
    dict(status_code=200, body=b"[" * 100_000),     # nesting deeper than json can parse
    dict(status_code=200, headers={"content-encoding": "gzip"}, body=b"{}"),
])
def test_http_fetch_report_returns_none_for_an_unusable_answer(response):
    import httpx

    seen = []

    def handler(request):
        seen.append(str(request.url))
        body = response.get("body")
        return httpx.Response(response["status_code"], headers=response.get("headers"),
                              content=_streamed(body))

    assert asyncio.run(fleet_module.http_fetch_report(
        ADDRESS, transport=_mock_transport(handler))) is None
    assert seen == [ADDRESS + "/capacity"]                 # no redirect followed


# --- the directory snapshot: bounded refresh, bounded age, fail closed ---


class Documents:
    """An R2 directory read that can change or fail between refreshes."""

    def __init__(self, documents):
        self.documents = documents
        self.failing = False
        self.reads = 0

    async def __call__(self):
        self.reads += 1
        if self.failing:
            raise OSError("r2 unreachable")
        return self.documents


def fleet_over(tmp_path, **kw):
    machine = signer(tmp_path, "m", "k1")
    documents = Documents([machine_document(machine)])
    served = {}
    clock = Clock()

    async def fetch(address):
        return served.get(address)

    fleet = Fleet(read_documents=documents, fetch_report=fetch, clock=clock, **kw)
    return fleet, machine, documents, served, clock


def test_the_directory_is_rebuilt_on_its_period_so_an_ended_key_stops_counting(tmp_path):
    fleet, machine, documents, served, clock = fleet_over(tmp_path, directory_refresh_s=30)
    asyncio.run(fleet.step())
    assert documents.reads == 1
    assert fleet.directory().public_key(MACHINE, "k1", NOW) is not None
    # the operator backdates the key's end (compromise)
    documents.documents = [machine_document(machine, valid_until=NOW - 100)]
    clock.now = NOW + 29
    asyncio.run(fleet.step())
    assert documents.reads == 1                                  # not due yet
    assert fleet.directory().public_key(MACHINE, "k1", NOW) is not None
    clock.now = NOW + 30
    asyncio.run(fleet.step())
    assert documents.reads == 2
    assert fleet.directory().public_key(MACHINE, "k1", NOW) is None
    served[ADDRESS] = capacity_report(machine, at=NOW + 30)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW + 30, **PICK) is None


def test_a_never_read_directory_fails_closed(tmp_path):
    fleet, machine, _, served, _ = fleet_over(tmp_path)
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    assert not fleet.directory_ready()
    assert fleet.directory().entries() == ()
    assert fleet.pick(now=NOW, **PICK) is None


def test_a_directory_older_than_its_max_age_fails_closed_and_alerts(tmp_path, caplog):
    fleet, machine, documents, served, clock = fleet_over(
        tmp_path, directory_refresh_s=30, directory_max_age_s=120)
    asyncio.run(fleet.step())
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.poll_once())
    assert fleet.pick(now=NOW, **PICK) is not None
    documents.failing = True
    for step in range(5, 125, 5):
        clock.now = NOW + step
        served[ADDRESS] = capacity_report(machine, at=NOW + step)
        asyncio.run(fleet.step())
    assert fleet.directory_ready()                               # age 120: still inside
    assert fleet.pick(now=NOW + 120, **PICK) is not None
    clock.now = NOW + 121
    served[ADDRESS] = capacity_report(machine, at=NOW + 121)
    with caplog.at_level("ERROR", logger="reliquary.sandbox.fleet"):
        asyncio.run(fleet.step())
    assert not fleet.directory_ready()
    assert fleet.directory_age() == 121
    # no key, so neither a heartbeat nor a transcript verifies against it
    assert fleet.directory().public_key(MACHINE, "k1", NOW) is None
    assert fleet.accept_report(MACHINE, served[ADDRESS], NOW + 121) is False
    assert fleet.pick(now=NOW + 121, **PICK) is None
    assert any("directory" in r.getMessage() and "stale" in r.getMessage()
               for r in caplog.records if r.levelname == "ERROR")
    # R2 is back: the next step refreshes and the machine is placed again, not drained
    documents.failing = False
    clock.now = NOW + 200
    served[ADDRESS] = capacity_report(machine, at=NOW + 200)
    asyncio.run(fleet.step())             # due after backoff: polls (still stale), then reads
    asyncio.run(fleet.poll_once())        # the next tick's poll
    assert fleet.directory_ready() and MACHINE not in fleet.drained
    assert fleet.pick(now=NOW + 200, **PICK) == Placement(MACHINE, ADDRESS)


def test_a_stale_directory_drains_no_machine(tmp_path):
    drained = []
    fleet, machine, documents, served, clock = fleet_over(tmp_path, on_drained=drained.append)
    asyncio.run(fleet.step())
    documents.failing = True
    for step in range(5, 400, 5):
        clock.now = NOW + step
        # heartbeats while the snapshot is usable, then silence once it is stale
        if step <= 120:
            served[ADDRESS] = capacity_report(machine, at=NOW + step)
        else:
            served.pop(ADDRESS, None)
        asyncio.run(fleet.step())
    assert not fleet.directory_ready()
    assert drained == [] and fleet.drained == set()
    # R2 back: silence is counted from the recovery, not from the last heartbeat
    documents.failing = False
    clock.now = NOW + 400
    asyncio.run(fleet.refresh_directory())   # the recovery read (backoff timing aside)
    asyncio.run(fleet.poll_once())
    assert drained == []
    clock.now = NOW + 461
    asyncio.run(fleet.step())
    assert drained == [MACHINE]


def test_failed_reads_are_logged_and_the_last_snapshot_kept_inside_its_age(tmp_path, caplog):
    fleet, machine, documents, served, clock = fleet_over(tmp_path)
    asyncio.run(fleet.step())
    documents.failing = True
    clock.now = NOW + 30
    with caplog.at_level("ERROR", logger="reliquary.sandbox.fleet"):
        asyncio.run(fleet.step())
    assert fleet.directory().entry(MACHINE) is not None
    assert any(r.levelname == "ERROR" for r in caplog.records)


@pytest.mark.parametrize("kw", [
    dict(directory_refresh_s=0),
    dict(directory_refresh_s=60, directory_max_age_s=60),
    dict(directory_max_age_s=-1),
])
def test_directory_settings_must_leave_room_for_a_refresh(tmp_path, kw):
    with pytest.raises(ValueError):
        fleet_over(tmp_path, **kw)


def test_directory_settings_default_to_30_and_120():
    assert fleet_module.DIRECTORY_REFRESH_SECONDS == 30.0
    assert fleet_module.DIRECTORY_MAX_AGE_SECONDS == 120.0


def test_run_stops_on_its_event(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_module, "POLL_SECONDS", 0.01)
    fleet, machine, documents, served, clock = fleet_over(tmp_path)

    async def scenario():
        stop = asyncio.Event()
        task = asyncio.create_task(fleet.run(stop))
        await asyncio.sleep(0.05)
        stop.set()
        await asyncio.wait_for(task, timeout=2.0)

    asyncio.run(scenario())
    assert documents.reads >= 1 and fleet.directory_ready()


# --- fix round 1: bounded, non-blocking, backed-off directory reads; drain baseline ---


def test_a_hanging_directory_read_times_out(tmp_path):
    fleet, machine, documents, served, clock = fleet_over(tmp_path, directory_read_timeout_s=0.1)

    async def hang():
        await asyncio.sleep(3600)

    fleet._read_documents = hang

    async def scenario():
        with pytest.raises(TimeoutError):
            await fleet.refresh_directory()

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
    assert fleet.directory_read_timeout_s == 0.1


def test_the_read_timeout_defaults_to_15_and_must_be_positive(tmp_path):
    assert fleet_module.DIRECTORY_READ_TIMEOUT_SECONDS == 15.0
    with pytest.raises(ValueError):
        fleet_over(tmp_path, directory_read_timeout_s=0)


def test_a_hanging_read_does_not_stop_polls_or_drain_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_module, "POLL_SECONDS", 0.01)
    drained = []
    fleet, machine, documents, served, clock = fleet_over(
        tmp_path, on_drained=drained.append, directory_refresh_s=1,
        directory_read_timeout_s=3600)
    fetches = []
    real_read = documents.__call__

    async def read():
        if documents.reads >= 1:          # the first read works, every later one hangs
            documents.reads += 1
            await asyncio.sleep(3600)
        return await real_read()

    async def fetch(address):             # the machine is silent; each poll is 10 s of clock
        fetches.append(address)
        clock.now += 10
        return None

    fleet._read_documents, fleet._fetch = read, fetch

    async def scenario():
        stop = asyncio.Event()
        fleet.on_drained = lambda machine_id: (drained.append(machine_id), stop.set())
        await fleet.run(stop)

    asyncio.run(asyncio.wait_for(scenario(), timeout=5.0))
    assert drained == [MACHINE]
    assert documents.reads >= 2           # a read was hanging while the polls went on
    assert len(fetches) >= 6


def test_failed_refreshes_back_off_exponentially_up_to_30_s(tmp_path, caplog):
    fleet, machine, documents, served, clock = fleet_over(
        tmp_path, directory_refresh_s=30, directory_max_age_s=1000)
    asyncio.run(fleet.step())
    assert documents.reads == 1
    documents.failing = True
    attempts = []
    with caplog.at_level("ERROR", logger="reliquary.sandbox.fleet"):
        for second in range(1, 200):
            clock.now = NOW + second
            before = documents.reads
            asyncio.run(fleet.step())
            if documents.reads > before:
                attempts.append(second)
    assert attempts == [30, 35, 45, 65, 95, 125, 155, 185]
    failures = [r for r in caplog.records
                if r.levelname == "ERROR" and "read failed" in r.getMessage()]
    assert len(failures) == len(attempts)              # one line per backoff step
    # a success resets the backoff: the next read is a full period later
    documents.failing = False
    clock.now = NOW + 215
    asyncio.run(fleet.step())
    assert documents.reads == len(attempts) + 2
    clock.now = NOW + 244
    asyncio.run(fleet.step())
    assert documents.reads == len(attempts) + 2
    clock.now = NOW + 245
    asyncio.run(fleet.step())
    assert documents.reads == len(attempts) + 3
    # and the backoff starts over at 5 s after a success
    documents.failing = True
    retried = []
    for second in range(246, 300):
        clock.now = NOW + second
        before = documents.reads
        asyncio.run(fleet.step())
        if documents.reads > before:
            retried.append(second)
    assert retried == [275, 280, 290]


def test_silence_counts_from_when_a_machine_entered_the_directory(tmp_path):
    drained = []
    fleet, machine, documents, served, clock = fleet_over(tmp_path, on_drained=drained.append)
    newcomer = signer(tmp_path, "n", "kn")
    served[ADDRESS] = capacity_report(machine, at=NOW)
    asyncio.run(fleet.step())
    asyncio.run(fleet.poll_once())
    clock.now = NOW + 100
    documents.documents = [machine_document(machine),
                           machine_document(newcomer, machine_id="m-new", address="http://n:1")]
    served[ADDRESS] = capacity_report(machine, at=NOW + 100)
    asyncio.run(fleet.refresh_directory())
    clock.now = NOW + 130
    served[ADDRESS] = capacity_report(machine, at=NOW + 130)
    asyncio.run(fleet.poll_once())
    assert drained == []                                  # 30 s in the directory, not 130
    clock.now = NOW + 161
    served[ADDRESS] = capacity_report(machine, at=NOW + 161)
    asyncio.run(fleet.refresh_directory())
    asyncio.run(fleet.poll_once())
    assert drained == ["m-new"]


def test_no_alert_before_the_first_read_has_finished(tmp_path, caplog):
    fleet, machine, documents, served, clock = fleet_over(tmp_path)
    with caplog.at_level("INFO", logger="reliquary.sandbox.fleet"):
        assert fleet.pick(now=NOW, **PICK) is None
        assert fleet.directory().entries() == ()
        asyncio.run(fleet.poll_once())
    assert not [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("not loaded yet" in r.getMessage() for r in caplog.records
               if r.levelname == "INFO")
    caplog.clear()
    documents.failing = True
    with caplog.at_level("ERROR", logger="reliquary.sandbox.fleet"):
        asyncio.run(fleet.step())
        fleet.directory()
    assert any("ALERT" in r.getMessage() for r in caplog.records if r.levelname == "ERROR")


def test_http_fetch_report_ignores_proxy_environment(monkeypatch):
    import httpx

    made = []
    real = httpx.AsyncClient

    def client(*args, **kwargs):
        made.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(httpx, "AsyncClient", client)
    transport = _mock_transport(lambda request: httpx.Response(500, content=_streamed(b"")))
    asyncio.run(fleet_module.http_fetch_report(ADDRESS, transport=transport))
    assert made and made[0].get("trust_env") is False


def _two_machines(tmp_path, a, b):
    one, two = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    # listed b first: the order of the documents must not matter
    documents = [machine_document(two, machine_id="m-b", address="http://b:1", capacity=b[1]),
                 machine_document(one, machine_id="m-a", address="http://a:1", capacity=a[1])]
    fleet, _, served, _ = make(tmp_path, documents=documents)
    served["http://a:1"] = capacity_report(one, at=NOW, machine_id="m-a", address="http://a:1",
                                           free=a[0], capacity=a[1])
    served["http://b:1"] = capacity_report(two, at=NOW, machine_id="m-b", address="http://b:1",
                                           free=b[0], capacity=b[1])
    asyncio.run(fleet.poll_once())
    return fleet


@pytest.mark.parametrize("a,b,winner", [
    ((2, 4), (2, 4), "m-a"),          # same free fraction: the smaller machine id
    ((1, 2), (2, 4), "m-a"),          # same fraction, different sizes: still the smaller id
    ((2, 2), (3, 4), "m-a"),          # the free FRACTION decides, not the free count
    ((1, 4), (2, 4), "m-b"),
])
def test_the_tie_break_order_is_pinned(tmp_path, a, b, winner):
    fleet = _two_machines(tmp_path, a, b)
    assert fleet.pick(now=NOW, **PICK).machine_id == winner
