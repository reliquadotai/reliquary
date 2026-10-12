"""The RL wiring runs the yield ledger from the window's settlement and the refusal stages."""
import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.sandbox.rl_capacity import YieldStats  # noqa: E402
from reliquary.sandbox.rl_yield import LedgerEvents  # noqa: E402
from reliquary.sandbox.sessions import SUBMITTED, Refusal  # noqa: E402
from reliquary.services.runtime import WindowUses  # noqa: E402
from tests.unit.episode_v2_fixtures import EPISODE, SANDBOX_ENV, SPLIT  # noqa: E402
from tests.unit.service_v2_fixtures import MATH  # noqa: E402
from tests.unit.test_rl_episode_wiring import _record, _services  # noqa: E402
from tests.unit.test_service_runtime_v2 import archive, runtime, train  # noqa: E402

SHA = "a" * 64


def test_the_runtime_says_which_groups_its_settlement_used(tmp_path):
    rt = runtime(tmp_path)
    assert rt.window_uses(1) is None                                  # not settled yet
    train(rt, prompt=7, hotkey="t")
    rt.reconcile_archive(archive(batch=[("t", MATH, 7)]), now=10_100.0)
    assert rt.window_uses(1) == WindowUses(aborted=False, used=frozenset({(MATH, 7, "t")}))


def test_an_aborted_window_says_so(tmp_path):
    rt = runtime(tmp_path)
    rt.reconcile_archive(archive(), aborted=True, now=10_100.0)
    assert rt.window_uses(1) == WindowUses(aborted=True, used=frozenset())


async def test_the_wiring_builds_the_ledger_as_the_outcomes_and_settles_from_the_runtime(tmp_path):
    uses = {}
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3", window_uses=uses.get)
    assert services.ledger is not None and services.intake._outcomes is services.ledger
    assert any(isinstance(listener, LedgerEvents) for listener in services.issuer._events.listeners)
    assert services.issuer._record_faults
    assert services.issuer._open_refusal_check == services.fleet.open_refusal_valid
    record = _record("s-1", sha=SHA, state=SUBMITTED)
    services.book.add(record)
    services.ledger.note_opened(record)
    services.ledger.group_settled(hotkey=record.hotkey, precommit_sha256=SHA, session_ids=("s-1",), accepted=True)
    assert await services.settle_yield() == 0                         # window 1 not settled yet
    uses[1] = WindowUses(aborted=False, used=frozenset({(EPISODE, record.index, record.hotkey)}))
    assert await services.settle_yield() == 1
    assert services.ledger.stats(record.hotkey) == YieldStats(
        useful=1, decided=1, useful_24h=1, useful_s_24h=services.ledger.stats(record.hotkey).useful_s_24h)
    assert services.ledger.stats(record.hotkey).useful_s_24h > 0      # its box ran: a graded final
    assert services.policies[EPISODE].sandbox_env == SANDBOX_ENV and services.policies[EPISODE].split == SPLIT
    await services.stop()
    assert services.ledger._closed                                   # closed after its last flush


async def test_a_session_a_machine_listed_counts_its_held_seconds(tmp_path):
    """A live or lapsed session with no machine final counts its held seconds only when a machine report
    listed it."""
    uses = {1: WindowUses(aborted=False, used=frozenset({(EPISODE, 3, "5Hot")}))}
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3", window_uses=uses.get)
    listed = _record("s-listed", sha=SHA, state="lapsed")
    unlisted = _record("s-unlisted", sha=SHA, state="lapsed", seed=1)
    for record in (listed, unlisted):
        services.book.add(record)
        services.ledger.note_opened(record)
    services.fleet.session_reported = lambda session_id: session_id == "s-listed"
    held = {}
    original = services.ledger.decide

    def decide(sha, rows, **kwargs):
        rows = list(rows)
        held.update({row[0]: row[4] for row in rows})
        return original(sha, rows, **kwargs)

    services.ledger.decide = decide
    assert await services.settle_yield() == 2
    assert held["s-unlisted"] == 0.0 and held["s-listed"] > 0.0
    await services.stop()


async def test_a_refusal_of_ours_reaches_the_ledger_with_its_stage(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3", window_uses=lambda w: None)
    record = _record("s-1", sha=SHA)
    services.ledger.note_opened(record)
    services.intake._report(record.hotkey, SHA, ("s-1",), accepted=False, stage="episode_window_resumed")
    assert SHA not in services.ledger._verdicts                       # before the check: no verdict
    services.intake._report(record.hotkey, SHA, ("s-1",), accepted=False, stage="episode_prompt_source")
    assert SHA not in services.ledger._verdicts                       # our prompt source failed in the check
    services.intake._report(record.hotkey, SHA, ("s-1",), accepted=False, stage="episode_persist_failed")
    assert services.ledger._verdicts[SHA][0] == "ours"


async def test_a_claimed_group_is_settled_with_the_servers_stage(tmp_path):
    from reliquary.validator.episode_intake import EpisodeClaim

    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3", window_uses=lambda w: None)
    record = _record("s-1", sha=SHA)
    services.ledger.note_opened(record)
    claim = EpisodeClaim(record.hotkey, SHA, ("s-1",))
    await services.intake.settle(claim, accepted=False, stage="admission_timeout")
    assert services.ledger._verdicts[SHA][0] == "ours"
    await services.intake.settle(claim, accepted=False, stage="proof")
    assert services.ledger._verdicts[SHA][0] == "refused"             # a real refusal outranks ours


@pytest.mark.parametrize("stage, verdict", [("content_identity", "ours"), ("service_eligibility", "ours"),
                                            ("admission_identity", "refused"), ("proof", "refused")])
async def test_the_batchers_refusals_of_a_claimed_group_are_ours_only_when_they_are(tmp_path, stage, verdict):
    """A missing content digest and an inactive runtime are ours; a proof or identity refusal is real."""
    from reliquary.validator.episode_intake import EpisodeClaim

    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3", window_uses=lambda w: None)
    record = _record("s-1", sha=SHA)
    services.ledger.note_opened(record)
    await services.intake.settle(EpisodeClaim(record.hotkey, SHA, ("s-1",)), accepted=False, stage=stage)
    assert services.ledger._verdicts[SHA][0] == verdict


def test_a_refusal_before_the_claim_never_reports_a_stage_of_ours(tmp_path):
    from types import SimpleNamespace

    from reliquary.protocol.submission import RejectReason

    calls = []
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    services.intake._outcomes = SimpleNamespace(group_settled=lambda **kw: calls.append(kw))
    prepared = SimpleNamespace(request=SimpleNamespace(miner_hotkey="5Hot", precommit_sha256=SHA),
                               episode_pending=True, reject_reason=None, reject_stage=None)
    services.intake._refuse(prepared, RejectReason.WORKER_DROPPED, "episode_persist_failed")
    services.intake._refuse(prepared, RejectReason.WORKER_DROPPED, "episode_checker_busy")
    assert [call["stage"] for call in calls] == ["episode_checker_busy"]      # ours before the claim: no verdict


async def test_the_server_passes_its_stage_when_it_settles_a_claim():
    from reliquary.validator.server import ValidatorServer

    seen = []

    class Intake:
        async def settle(self, claim, *, accepted, stage=None):
            seen.append((accepted, stage))

    server = ValidatorServer.__new__(ValidatorServer)
    server._episode_intake = Intake()
    await server._start_episode_settlement(object(), accepted=False, stage="episode_persist_failed")
    await server._start_episode_settlement(object(), accepted=True)
    assert seen == [(False, "episode_persist_failed"), (True, None)]


async def test_a_raising_decision_still_flushes_the_journal(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3",
                            window_uses=lambda w: WindowUses(aborted=False, used=frozenset()))
    record = _record("s-1", sha=SHA, state="lapsed")
    services.book.add(record)
    services.ledger.note_opened(record)

    def boom(*args, **kwargs):
        raise RuntimeError("decide failed")

    services.ledger.decide = boom
    with pytest.raises(RuntimeError):
        await services.settle_yield()
    assert services.ledger._journal == [] and services.ledger.version == 1      # written all the same
    await services.stop()


async def test_a_raising_drain_at_stop_still_flushes_and_closes_the_ledger(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3",
                            window_uses=lambda w: WindowUses(aborted=False, used=frozenset()))
    record = _record("s-1", sha=SHA, state="lapsed")
    services.book.add(record)
    services.ledger.note_opened(record)
    assert services.ledger._journal                                   # something to flush

    async def drain(timeout=None):
        raise RuntimeError("a state write failed while draining")

    services.issuer.drain = drain
    with pytest.raises(RuntimeError, match="draining"):               # the drain's error is not swallowed
        await services.stop()
    assert services.ledger._journal == [] and services.ledger.version >= 1   # the final flush ran
    assert services.ledger._closed                                   # and the ledger is closed


async def test_when_the_drain_and_the_close_both_raise_the_drains_error_wins(tmp_path, caplog):
    import logging

    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3",
                            window_uses=lambda w: WindowUses(aborted=False, used=frozenset()))

    async def drain(timeout=None):
        raise RuntimeError("a state write failed while draining")

    def close():
        raise OSError("the ledger file could not be closed")

    services.issuer.drain = drain
    real_close, services.ledger.close = services.ledger.close, close
    with caplog.at_level(logging.ERROR), pytest.raises(RuntimeError, match="draining"):
        await services.stop()
    assert "ledger not closed at shutdown" in caplog.text           # logged, never hiding the drain's error
    services.ledger.close = real_close
    services.ledger.close()


def test_a_capacity_refusal_of_an_open_reaches_the_ledger(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3")
    record = _record("s-1", sha=SHA)
    services.ledger.note_opened(record)
    events = LedgerEvents(services.ledger)
    engagement = {"kind": "rl_precommit", "precommit": {"precommit_sha256": SHA, "seed_index": 3}}
    events.refused(record.hotkey, engagement, Refusal("quota_live", {}))      # the miner's own limit: nothing
    assert SHA not in services.ledger._verdicts
    events.refused(record.hotkey, engagement, Refusal("sandbox_saturated", {}))
    assert services.ledger._verdicts[SHA][0] == "capacity"


@pytest.mark.parametrize("accepting, resumed", [(True, True), (False, False)])
async def test_start_marks_the_window_a_restart_interrupted(tmp_path, accepting, resumed):
    from reliquary.infrastructure.sandbox_store import MemorySessionStore
    from reliquary.sandbox.rl_yield import SessionYieldLedger
    from reliquary.sandbox.sessions import LIVE

    store = MemorySessionStore()
    record = _record("s-live", sha=SHA, state=LIVE)                      # a live session of an abandoned precommit
    store.documents[record.session_id] = record.to_document()
    before = SessionYieldLedger(tmp_path / "rl-yield.sqlite3")           # the crashed process's last flush
    before.note_accepting(record_window(record) if accepting else None, 1.0)
    before.flush(1.0)
    before.close()
    services, _ = _services(tmp_path, store=store, yield_path=tmp_path / "rl-yield.sqlite3",
                            window_started_at=lambda window: None, window_uses=lambda window: WindowUses(
                                aborted=False, used=frozenset()))
    await services.start()
    assert services.resumed_windows == ({record_window(record)} if resumed else set())
    services.book.get("s-live").state = "lapsed"                       # it lapsed: nothing holds the precommit
    assert await services.settle_yield() == 1
    assert services.ledger.stats(record.hotkey).decided == (0 if resumed else 1)   # the restart's window: neither
    await services.stop()


async def test_the_wiring_records_the_window_accepting_groups(tmp_path):
    services, _ = _services(tmp_path, yield_path=tmp_path / "rl-yield.sqlite3", window_uses=lambda w: None,
                            current_window=lambda: 4)
    await services.settle_yield()
    assert services.ledger.accepting_at_last_flush() == 4


def record_window(record):
    from reliquary.protocol.service_episode import parse_rl_engagement

    return parse_rl_engagement(record.engagement)[0]


async def test_start_reconciles_the_restored_book(tmp_path):
    from reliquary.infrastructure.sandbox_store import MemorySessionStore

    store = MemorySessionStore()
    record = _record("s-restored", sha=SHA, state=SUBMITTED)
    store.documents[record.session_id] = record.to_document()
    services, _ = _services(tmp_path, store=store, yield_path=tmp_path / "rl-yield.sqlite3")
    await services.start()
    assert SHA in services.ledger._pending and services.ledger._verdicts[SHA][0] == "accepted"
    assert services.issuer._restored == {"s-restored"}
    await services.stop()


def test_without_a_yield_path_nothing_changes(tmp_path):
    services, _ = _services(tmp_path)
    assert services.ledger is None and services.issuer._events is None and not services.issuer._record_faults
    assert services.issuer._operator_of is None and services.issuer._entitled is None


# --- the validator opens the ledger beside its runtime ---
async def test_the_validator_opens_the_ledger_beside_its_runtime(tmp_path, monkeypatch):
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
    from tests.unit.episode_v2_fixtures import episode_runtime
    from tests.unit.test_rl_episode_wiring import _service

    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    service = _service(tmp_path, monkeypatch, runtime=episode_runtime(tmp_path / "rt"))
    service._service_folder = tmp_path / "folder"
    service._service_folder.mkdir()
    service.server._operator_by_hotkey = {"5Hot": "5Cold"}
    import threading

    from reliquary.sandbox import rl_yield

    built_on = []
    real = rl_yield.SessionYieldLedger

    def ledger(*args, **kwargs):
        built_on.append(threading.get_ident())
        return real(*args, **kwargs)

    monkeypatch.setattr(rl_yield, "SessionYieldLedger", ledger)
    loop_thread = threading.get_ident()
    await service._start_episode_services()
    assert built_on and loop_thread not in built_on                  # SQLite opened off the event loop
    try:
        services = service._episode_services
        assert services.ledger is not None and services.intake._outcomes is services.ledger
        assert (tmp_path / "folder" / "rl-yield.sqlite3").exists()
        assert services.window_uses == service._service_runtime.window_uses
        assert services.operator_of("5Hot") == "5Cold" and services.operator_of("5Other") is None
        assert services.issuer._operator_of("5Hot") == "5Cold"     # a saturation is judged by operator
        assert services.issuer._entitled == services.quota.sessions_allowed    # within it, a saturation is ours
        # and the yield settlement, the served-env watch and the saturation ALERTs
        assert len(service._episode_tasks) == 6
        assert services.saturation is not None
    finally:
        await service._stop_episode_services()
    assert services.ledger._closed


async def test_an_unusable_ledger_stops_the_start_with_an_alert(tmp_path, monkeypatch, caplog):
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
    from reliquary.sandbox.rl_yield import YieldLedgerUnusable
    from tests.unit.episode_v2_fixtures import episode_runtime
    from tests.unit.test_rl_episode_wiring import _service

    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    service = _service(tmp_path, monkeypatch, runtime=episode_runtime(tmp_path / "rt"))
    service._service_folder = tmp_path / "folder"
    service._service_folder.mkdir()
    garbage = b"not a database" * 100
    (service._service_folder / "rl-yield.sqlite3").write_bytes(garbage)
    with pytest.raises(YieldLedgerUnusable):
        await service._start_episode_services()
    assert any("ALERT" in r.getMessage() and "yield ledger" in r.getMessage() for r in caplog.records)
    assert (service._service_folder / "rl-yield.sqlite3").read_bytes() == garbage     # never replaced
    assert getattr(service, "_episode_services", None) is None and service.server._episode_intake is None


async def test_a_failed_start_closes_the_ledger(tmp_path, monkeypatch):
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
    from reliquary.sandbox import rl_yield
    from reliquary.validator.rl_sandbox_wiring import RlEpisodeServices
    from tests.unit.episode_v2_fixtures import episode_runtime
    from tests.unit.test_rl_episode_wiring import _service

    opened = []
    real = rl_yield.SessionYieldLedger

    def ledger(*args, **kwargs):
        opened.append(real(*args, **kwargs))
        return opened[-1]

    async def broken_start(self):
        raise OSError("store down")

    monkeypatch.setattr(rl_yield, "SessionYieldLedger", ledger)
    monkeypatch.setattr(RlEpisodeServices, "start", broken_start)
    monkeypatch.setattr("reliquary.protocol.profiles.toploc_proof", lambda profile: TOPLOC_DEPLOYED_DEFAULTS)
    service = _service(tmp_path, monkeypatch, runtime=episode_runtime(tmp_path / "rt"))
    service._service_folder = tmp_path / "folder"
    service._service_folder.mkdir()
    with pytest.raises(OSError):
        await service._start_episode_services()
    assert len(opened) == 1 and opened[0]._closed
