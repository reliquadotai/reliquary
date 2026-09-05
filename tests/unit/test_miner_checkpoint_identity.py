"""Durable miner checkpoint identity survives restart without rebinding."""

import json
from types import SimpleNamespace

import pytest

from reliquary.miner.checkpoint_identity import (
    ActivatedCheckpoint,
    CheckpointIdentityError,
    MinerCheckpointIdentityStore,
    checkpoint_identity_from_state,
)
from reliquary.miner.engine import maybe_pull_checkpoint


REPO = "org/repo"
OTHER_REPO = "org/other"
REV_4 = "4" * 40
REV_5 = "5" * 40
REV_6 = "6" * 40


def _identity(number=5, repo=REPO, oid=REV_5):
    return ActivatedCheckpoint(number, repo, oid)


def test_store_create_reload_and_idempotent_compare(tmp_path):
    path = tmp_path / "checkpoint.json"
    first = MinerCheckpointIdentityStore(path)
    first.commit(_identity())
    committed = path.read_bytes()

    restarted = MinerCheckpointIdentityStore(path)
    assert restarted.load() == _identity()
    restarted.commit(_identity())
    assert path.read_bytes() == committed
    assert path.stat().st_mode & 0o777 == 0o600


def test_store_advances_monotonically(tmp_path):
    store = MinerCheckpointIdentityStore(tmp_path / "checkpoint.json")
    store.commit(_identity(number=5, oid=REV_5))
    store.commit(_identity(number=6, oid=REV_6))

    assert store.load() == _identity(number=6, oid=REV_6)
    with pytest.raises(CheckpointIdentityError, match="rolled back"):
        store.commit(_identity(number=5, oid=REV_5))


@pytest.mark.parametrize(
    "candidate",
    [
        _identity(repo=OTHER_REPO),
        _identity(oid=REV_6),
    ],
)
def test_store_rejects_same_number_rebinding(tmp_path, candidate):
    store = MinerCheckpointIdentityStore(tmp_path / "checkpoint.json")
    store.commit(_identity())

    with pytest.raises(CheckpointIdentityError, match="rebound"):
        MinerCheckpointIdentityStore(store.path).commit(candidate)


@pytest.mark.parametrize(
    "record",
    [
        b"not-json",
        (
            b'{"schema_version":1,"schema_version":1,"checkpoint_n":5,'
            b'"repo_id":"org/repo","oid":"' + REV_5.encode() + b'"}'
        ),
        json.dumps(
            {
                "schema_version": True,
                "checkpoint_n": 5,
                "repo_id": REPO,
                "oid": REV_5,
            }
        ).encode(),
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_n": True,
                "repo_id": REPO,
                "oid": REV_5,
            }
        ).encode(),
        json.dumps(
            {
                "schema_version": 1,
                "checkpoint_n": 5,
                "repo_id": REPO,
                "oid": "main",
            }
        ).encode(),
    ],
)
def test_corrupt_record_fails_closed(tmp_path, record):
    path = tmp_path / "checkpoint.json"
    path.write_bytes(record)

    with pytest.raises(CheckpointIdentityError, match="corrupt"):
        MinerCheckpointIdentityStore(path).load()


def test_state_identity_is_complete_and_canonical():
    assert checkpoint_identity_from_state(
        SimpleNamespace(
            checkpoint_n=0,
            checkpoint_repo_id=None,
            checkpoint_revision=None,
        )
    ) is None
    assert checkpoint_identity_from_state(
        SimpleNamespace(
            checkpoint_n=5,
            checkpoint_repo_id=REPO,
            checkpoint_revision=REV_5,
        )
    ) == _identity()

    for revision in (None, "main", "A" * 40):
        with pytest.raises(CheckpointIdentityError, match="invalid"):
            checkpoint_identity_from_state(
                SimpleNamespace(
                    checkpoint_n=5,
                    checkpoint_repo_id=REPO,
                    checkpoint_revision=revision,
                )
            )


@pytest.mark.asyncio
async def test_restart_record_detects_same_number_rebind_before_download(tmp_path):
    store = MinerCheckpointIdentityStore(tmp_path / "checkpoint.json")
    store.commit(_identity())
    restarted = MinerCheckpointIdentityStore(store.path).load()
    state = SimpleNamespace(
        checkpoint_n=5,
        checkpoint_repo_id=REPO,
        checkpoint_revision=REV_6,
    )
    download_calls = []

    async def download(repo, revision):
        download_calls.append((repo, revision))
        return "/unused"

    with pytest.raises(CheckpointIdentityError, match="rebound"):
        await maybe_pull_checkpoint(
            state,
            local_n=restarted.checkpoint_n,
            local_repo_id=restarted.repo_id,
            local_hash=restarted.oid,
            local_model=object(),
            download_fn=download,
            load_fn=lambda path: object(),
        )

    assert download_calls == []


def test_uppercase_and_mutable_oids_are_rejected():
    for oid in ("main", "A" * 40, "a" * 39):
        with pytest.raises(ValueError, match="40-character commit OID"):
            ActivatedCheckpoint(4, REPO, oid)
