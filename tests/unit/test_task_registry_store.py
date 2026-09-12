"""Two launches racing must not both succeed."""

from __future__ import annotations

import pytest

from reliquary.shared.task_registry import (
    MECHANISM_RL_DISCOVERED_PRICE,
    RegistryError,
    TaskEntry,
    parse_registry,
    render_registry,
)
from reliquary.infrastructure import task_registry_store as store

PARAMS = {
    "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
    "deadband": 0.80, "snap": 1.20, "floor": 0.05, "cap": 0.6,
    "median_rounds": 4800,
}


def _entry(task_id: str, cap: float) -> TaskEntry:
    return TaskEntry(
        task_id=task_id,
        profile_id="qwen3-4b-base-dapo-fill-closed-v6",
        profile_sha256="a" * 64,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params={**PARAMS, "cap": cap},
        status="active",
        retired_at=None,
    )


class FakeR2:
    """An object store with ETags and real conditional-put semantics."""

    def __init__(self, body: bytes | None = None):
        self.body = body
        self.etag = '"v1"' if body is not None else None
        self.puts = 0
        self.steal_once: bytes | None = None

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_object(self, Bucket, Key):
        if self.body is None:
            raise _client_error("NoSuchKey")

        class _Body:
            def __init__(self, data):
                self._data = data

            async def read(self):
                return self._data

        return {"Body": _Body(self.body), "ETag": self.etag}

    async def put_object(self, Bucket, Key, Body, **condition):
        # A concurrent writer lands between our read and our write, exactly once.
        if self.steal_once is not None:
            self.body, self.etag, self.steal_once = self.steal_once, '"v9"', None
            raise _client_error("PreconditionFailed")
        if "IfMatch" in condition and condition["IfMatch"] != self.etag:
            raise _client_error("PreconditionFailed")
        if "IfNoneMatch" in condition and self.body is not None:
            raise _client_error("PreconditionFailed")
        self.puts += 1
        self.body = Body
        self.etag = f'"v{self.puts + 1}"'
        return {"ETag": self.etag}


def _client_error(code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code}}, "PutObject")


@pytest.fixture
def fake(monkeypatch):
    holder = {}

    def _install(body: bytes | None = None) -> FakeR2:
        client = FakeR2(body)
        holder["client"] = client
        monkeypatch.setattr(store, "get_s3_client", lambda **kw: client)
        return client

    return _install


@pytest.mark.asyncio
async def test_an_absent_registry_reads_as_empty(fake):
    fake(None)

    entries, etag = await store.read_registry()

    assert entries == {}
    assert etag is None


@pytest.mark.asyncio
async def test_creating_the_first_task_writes_the_object(fake):
    client = fake(None)

    await store.create_task(_entry("default", 1.0))

    assert parse_registry(client.body)["default"].params["cap"] == 1.0


@pytest.mark.asyncio
async def test_a_lost_race_recomputes_and_refuses_when_it_no_longer_fits(fake):
    # We read a registry holding 0.5 free, but a rival takes 0.6 first.
    client = fake(render_registry({"a": _entry("a", 0.5)}))
    client.steal_once = render_registry(
        {"a": _entry("a", 0.5), "rival": _entry("rival", 0.4)}
    )

    with pytest.raises(RegistryError):
        await store.create_task(_entry("b", 0.4))


@pytest.mark.asyncio
async def test_a_lost_race_retries_and_succeeds_when_it_still_fits(fake):
    client = fake(render_registry({"a": _entry("a", 0.2)}))
    client.steal_once = render_registry(
        {"a": _entry("a", 0.2), "rival": _entry("rival", 0.2)}
    )

    await store.create_task(_entry("b", 0.2))

    entries = parse_registry(client.body)
    assert set(entries) == {"a", "rival", "b"}


@pytest.mark.asyncio
async def test_retiring_keeps_the_cap_reserved(fake):
    client = fake(render_registry({"a": _entry("a", 0.5)}))

    await store.retire_task_entry("a", retired_at=999)

    entry = parse_registry(client.body)["a"]
    assert entry.status == "retired"
    assert entry.params["cap"] == 0.5


@pytest.mark.asyncio
async def test_an_oversubscribed_registry_is_refused_by_default(fake):
    fake(render_registry({"a": _entry("a", 0.9), "b": _entry("b", 0.9)}))

    with pytest.raises(RegistryError):
        await store.read_registry()


@pytest.mark.asyncio
async def test_an_oversubscribed_registry_is_readable_when_not_strict(fake):
    fake(render_registry({"a": _entry("a", 0.9), "b": _entry("b", 0.9)}))

    entries, _ = await store.read_registry(strict=False)

    assert set(entries) == {"a", "b"}
