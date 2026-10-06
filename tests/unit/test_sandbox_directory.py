"""The machine directory: R2 documents changed under their ETag, and the snapshot the
verifier and the heartbeat check read keys from."""

import asyncio
import logging

import pytest

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.infrastructure import sandbox_store as store  # noqa: E402
from reliquary.sandbox.machines import (  # noqa: E402
    DirectorySnapshot, MachineEntry, snapshot_from_documents,
)
from tests.unit.sandbox_fixtures import (  # noqa: E402
    ADDRESS, MACHINE, machine_document, public_key_b64, signer,
)
from tests.unit.test_corpus_job_store import _FakeMultiObjectR2  # noqa: E402


@pytest.fixture
def bucket(monkeypatch):
    fake = _FakeMultiObjectR2()
    monkeypatch.setattr(store, "get_s3_client", lambda **kw: fake)
    return fake


def register(machine_id=MACHINE, **kw):
    fields = dict(machine_id=machine_id, address=ADDRESS, provider="hetzner", capacity=16,
                  key_id="k1", public_key_b64=public_key_b64(1), valid_from=100, now=50.0)
    fields.update(kw)
    return asyncio.run(store.register_machine(**fields))


def test_a_registered_machine_reads_back(bucket):
    document, created = register()
    assert created and asyncio.run(store.read_machine(MACHINE)) == document
    assert document["status"] == "active"
    assert document["keys"] == [{"key_id": "k1", "public_key_b64": public_key_b64(1),
                                 "valid_from": 100, "valid_until": None}]
    assert f"reliquary/sandbox/machines/{MACHINE}.json" in bucket.objects


def test_registering_twice_is_idempotent_and_differently_a_conflict(bucket):
    first, _ = register()
    again, created = register()
    assert not created and again == first
    with pytest.raises(store.MachineConflict):
        register(address="http://10.0.0.6:8080")


@pytest.mark.parametrize("field,value", [
    ("machine_id", "../x"), ("address", "ftp://h"), ("address", "http://h/"),
    ("address", "http://h/path"), ("public_key_b64", "not base64"),
    ("public_key_b64", public_key_b64(1)[:-4]), ("capacity", 0), ("provider", ""),
])
def test_bad_registrations_are_refused(bucket, field, value):
    with pytest.raises(ValueError):
        register(**{field: value})


def test_a_key_is_added_and_ended_but_an_end_never_moves_later(bucket):
    register()
    asyncio.run(store.add_machine_key(MACHINE, key_id="k2", public_key_b64=public_key_b64(2),
                                      valid_from=500))
    asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=500))
    with pytest.raises(store.MachineConflict):
        asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=900))
    # Compromise: backdated before it was even valid.
    ended = asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=10))
    assert [k["valid_until"] for k in ended["keys"]] == [10, None]
    with pytest.raises(store.MachineConflict):
        asyncio.run(store.add_machine_key(MACHINE, key_id="k2", public_key_b64=public_key_b64(3),
                                          valid_from=600))


def test_status_and_heartbeat_summaries_never_undo_each_other(bucket):
    register()
    asyncio.run(store.set_machine_status(MACHINE, "draining", reason="kernel update"))
    after = asyncio.run(store.record_machine_heartbeat(MACHINE, at=700.0, summary={"free": 3}))
    assert after["status"] == "draining" and after["last_heartbeat"] == 700.0
    assert after["heartbeat"] == {"free": 3}
    with pytest.raises(ValueError):
        asyncio.run(store.set_machine_status(MACHINE, "busy"))
    assert asyncio.run(store.set_machine_status("ghost", "revoked")) is None


def test_every_machine_is_listed(bucket):
    register("m-1")
    register("m-2", key_id="k9", public_key_b64=public_key_b64(9))
    assert sorted(d["machine_id"] for d in asyncio.run(store.list_machines())) == ["m-1", "m-2"]


def test_a_key_answers_only_inside_its_window(tmp_path):
    machine = signer(tmp_path, "m", "k1")
    snapshot = DirectorySnapshot([MachineEntry.from_document(
        machine_document(machine, valid_from=100, valid_until=500))])
    assert snapshot.public_key(MACHINE, "k1", 99) is None
    assert snapshot.public_key(MACHINE, "k1", 100) is not None
    assert snapshot.public_key(MACHINE, "k1", 499) is not None      # rotation: earlier opens verify
    assert snapshot.public_key(MACHINE, "k1", 500) is None          # compromise: later ones do not
    assert snapshot.public_key(MACHINE, "k2", 200) is None
    assert snapshot.public_key("other", "k1", 200) is None


def test_a_revoked_machine_keeps_its_keys_for_past_transcripts(tmp_path):
    machine = signer(tmp_path, "m", "k1")
    snapshot = DirectorySnapshot([MachineEntry.from_document(
        machine_document(machine, status="revoked"))])
    assert snapshot.public_key(MACHINE, "k1", 200) is not None
    assert snapshot.entry(MACHINE).status == "revoked"


def test_a_malformed_document_is_skipped_not_fatal(tmp_path, caplog):
    machine = signer(tmp_path, "m", "k1")
    # bittensor's logging sets every logger that already exists to CRITICAL.
    with caplog.at_level(logging.ERROR, logger="reliquary.sandbox.machines"):
        snapshot = snapshot_from_documents([machine_document(machine), {"schema": "x"}])
    assert [e.machine_id for e in snapshot.entries()] == [MACHINE]
    assert "unreadable" in caplog.text


def test_the_admin_commands_reach_the_store(bucket):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["sandbox", "machines", "register", "--machine-id", MACHINE,
                                 "--address", ADDRESS, "--provider", "hetzner", "--capacity", "8",
                                 "--key-id", "k1", "--public-key", public_key_b64(1),
                                 "--valid-from", "100"])
    assert result.exit_code == 0, result.output
    result = runner.invoke(app, ["sandbox", "machines", "end-key", "--machine-id", MACHINE,
                                 "--key-id", "k1", "--valid-until", "50"])
    assert result.exit_code == 0, result.output
    assert asyncio.run(store.read_machine(MACHINE))["keys"][0]["valid_until"] == 50


# --- Invariants -------------------------------------------------------------------------


class _RacingR2(_FakeMultiObjectR2):
    """Every read yields to the loop before returning, so concurrent writers all read
    the same version and race their conditional puts."""

    async def get_object(self, Bucket, Key):
        response = await super().get_object(Bucket, Key)
        await asyncio.sleep(0)
        return response


def test_an_end_never_moves_later_and_is_never_reopened(bucket):
    register()
    asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=500))
    asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=500))   # same: fine
    for later in (501, 10**10):
        with pytest.raises(store.MachineConflict):
            asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=later))
    for reopen in (None, -1, True, 500.5, "500"):
        with pytest.raises(ValueError):
            asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=reopen))
    with pytest.raises(store.MachineConflict):          # an unknown key is not created by an end
        asyncio.run(store.end_machine_key(MACHINE, key_id="k7", valid_until=10))
    assert asyncio.run(store.read_machine(MACHINE))["keys"][0]["valid_until"] == 500
    # Nothing else touches a key's window: status and heartbeats leave it as it is.
    asyncio.run(store.set_machine_status(MACHINE, "active"))
    after = asyncio.run(store.record_machine_heartbeat(
        MACHINE, at=900.0, summary={"keys": [], "valid_until": None}))
    assert after["keys"][0]["valid_until"] == 500


def test_concurrent_writers_never_lose_an_update(monkeypatch):
    racing = _RacingR2()
    monkeypatch.setattr(store, "get_s3_client", lambda **kw: racing)
    register()

    async def all_at_once():
        return await asyncio.gather(
            store.add_machine_key(MACHINE, key_id="k2", public_key_b64=public_key_b64(2),
                                  valid_from=500),
            store.end_machine_key(MACHINE, key_id="k1", valid_until=500),
            store.set_machine_status(MACHINE, "draining", reason="r"),
            store.record_machine_heartbeat(MACHINE, at=700.0, summary={"free": 1}))

    asyncio.run(all_at_once())
    stored = asyncio.run(store.read_machine(MACHINE))
    assert [(k["key_id"], k["valid_until"]) for k in stored["keys"]] == [("k1", 500), ("k2", None)]
    assert stored["status"] == "draining" and stored["last_heartbeat"] == 700.0


def test_a_writer_that_keeps_losing_gives_up_after_bounded_retries(monkeypatch):
    class _AlwaysMoved(_FakeMultiObjectR2):
        puts = 0

        async def put_object(self, Bucket, Key, Body, **condition):
            if "IfMatch" in condition:
                type(self).puts += 1
                stored, _ = self.objects[Key]
                await super().put_object(Bucket, Key, stored)      # someone else wrote
            return await super().put_object(Bucket, Key, Body, **condition)

    fake = _AlwaysMoved()
    monkeypatch.setattr(store, "get_s3_client", lambda **kw: fake)
    register()
    with pytest.raises(store.MachineConflict):
        asyncio.run(store.set_machine_status(MACHINE, "draining"))
    assert _AlwaysMoved.puts == store.WRITE_ATTEMPTS
    assert asyncio.run(store.read_machine(MACHINE))["status"] == "active"


def test_every_write_is_conditional(monkeypatch):
    seen = []

    class _Recording(_FakeMultiObjectR2):
        async def put_object(self, Bucket, Key, Body, **condition):
            seen.append(set(condition))
            return await super().put_object(Bucket, Key, Body, **condition)

    fake = _Recording()
    monkeypatch.setattr(store, "get_s3_client", lambda **kw: fake)
    register()
    asyncio.run(store.add_machine_key(MACHINE, key_id="k2", public_key_b64=public_key_b64(2),
                                      valid_from=500))
    asyncio.run(store.end_machine_key(MACHINE, key_id="k1", valid_until=500))
    asyncio.run(store.set_machine_status(MACHINE, "revoked"))
    asyncio.run(store.record_machine_heartbeat(MACHINE, at=1.0, summary={}))
    assert seen == [{"IfNoneMatch"}] + [{"IfMatch"}] * 4


def test_a_heartbeat_never_changes_the_address(bucket, tmp_path):
    register()
    after = asyncio.run(store.record_machine_heartbeat(
        MACHINE, at=700.0, summary={"public_base_url": "http://evil:1", "address": "http://evil:1"}))
    assert after["address"] == ADDRESS
    snapshot = snapshot_from_documents([after])
    assert snapshot.entry(MACHINE).address == ADDRESS


def test_the_snapshot_is_pure_and_deterministic(tmp_path):
    a, b = signer(tmp_path, "a", "ka"), signer(tmp_path, "b", "kb")
    documents = [machine_document(a, machine_id="m-a"), machine_document(b, machine_id="m-b")]
    frozen = [dict(d) for d in documents]
    one = snapshot_from_documents(documents)
    two = snapshot_from_documents(list(reversed(documents)))
    assert one.entries() == two.entries()
    assert [e.machine_id for e in one.entries()] == ["m-a", "m-b"]
    assert documents == frozen                             # the input is not touched
    documents[0]["keys"][0]["valid_until"] = 0             # nor read again later
    assert one.public_key("m-a", "ka", 200) is not None


def test_verify_transcript_reads_keys_from_the_snapshot(tmp_path):
    from reliquary_sandbox.attest import Ed25519TokenVerifier, Expected, Reason, verify_transcript

    from tests.unit.sandbox_fixtures import JOB_ID, claims, transcript
    from reliquary.corpus.signed_reasons import corpus_engagement

    validator, machine = signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1")
    session = claims()
    signed = transcript(validator, machine, session)
    tokens = Ed25519TokenVerifier({"v1": validator.public_key_b64})
    expected = Expected(hotkey="5Hot", engagement=corpus_engagement(JOB_ID, 0), env=session.env,
                        split=session.split, index=0, checkpoint=session.checkpoint)
    opened = session.issued_at + 10

    def check(**kw):
        snapshot = snapshot_from_documents([machine_document(machine, **kw)])
        return verify_transcript(signed, snapshot, tokens, expected)

    assert check().ok
    assert check(valid_until=opened + 1).ok                  # rotation after the open
    assert check(status="revoked").ok                        # revoked: past work stands
    assert Reason.UNKNOWN_KEY in check(valid_until=opened).reasons      # compromise
    assert Reason.UNKNOWN_KEY in check(valid_until=5).reasons


def test_admin_commands_print_only_public_documents(bucket, tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    machine = signer(tmp_path, "m", "k1")
    pem = (tmp_path / "m.pem").read_text()
    runner = CliRunner()
    outputs = []
    for args in (
        ["register", "--machine-id", MACHINE, "--address", ADDRESS, "--provider", "hetzner",
         "--capacity", "8", "--key-id", "k1", "--public-key", machine.public_key_b64,
         "--valid-from", "100"],
        ["add-key", "--machine-id", MACHINE, "--key-id", "k2", "--public-key", public_key_b64(2),
         "--valid-from", "500"],
        ["end-key", "--machine-id", MACHINE, "--key-id", "k1", "--valid-until", "500"],
        ["status", "--machine-id", MACHINE, "--status", "draining", "--reason", "upgrade"],
        ["list"],
    ):
        result = runner.invoke(app, ["sandbox", "machines", *args])
        assert result.exit_code == 0, result.output
        outputs.append(result.output)
    text = "".join(outputs)
    body = "".join(pem.splitlines()[1:-1])
    assert body[:40] not in text and "PRIVATE" not in text
    for line in outputs[-1].splitlines():
        document = __import__("json").loads(line)
        assert set(document) <= {"schema", "machine_id", "address", "provider", "capacity",
                                 "status", "status_reason", "keys", "registered_at",
                                 "last_heartbeat", "heartbeat"}
    result = runner.invoke(app, ["sandbox", "machines", "status", "--machine-id", "ghost",
                                 "--status", "revoked"])
    assert result.exit_code == 2


def test_a_refused_admin_command_exits_cleanly(bucket):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    runner = CliRunner()
    base = ["sandbox", "machines", "register", "--machine-id", MACHINE, "--provider", "hetzner",
            "--capacity", "8", "--key-id", "k1", "--public-key", public_key_b64(1),
            "--valid-from", "100"]
    assert runner.invoke(app, [*base, "--address", "http://h/path"]).exit_code == 1
    assert runner.invoke(app, [*base, "--address", ADDRESS]).exit_code == 0
    assert runner.invoke(app, [*base, "--address", "http://10.0.0.6:8080"]).exit_code == 1
    result = runner.invoke(app, ["sandbox", "machines", "end-key", "--machine-id", MACHINE,
                                 "--key-id", "k1", "--valid-until", "50"])
    assert result.exit_code == 0
    result = runner.invoke(app, ["sandbox", "machines", "end-key", "--machine-id", MACHINE,
                                 "--key-id", "k1", "--valid-until", "60"])
    assert result.exit_code == 1 and "earlier" in result.output
