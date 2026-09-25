"""Lowering the RL task's cap is the step that makes room for a corpus task."""

from __future__ import annotations

from dataclasses import replace

import pytest

from reliquary.infrastructure import task_registry_store as store
from reliquary.shared.task_registry import (
    MECHANISM_CORPUS_GENERATION,
    RegistryError,
    parse_registry,
    render_registry,
)
from tests.unit.test_task_registry_store import _entry, fake  # noqa: F401  (fixture)


def _corpus(task_id: str = "corpus-run", cap: float = 0.1):
    return replace(
        _entry(task_id, cap),
        mechanism=MECHANISM_CORPUS_GENERATION,
        params={**_entry(task_id, cap).params, "floor": cap},
        job_id="swe-v1",
    )


@pytest.mark.asyncio
async def test_lowering_default_makes_room_for_a_corpus_task(fake):
    client = fake(render_registry({"default": _entry("default", 1.0)}))
    with pytest.raises(RegistryError, match="total"):
        await store.create_task(_corpus())

    await store.set_task_cap("default", 0.9)
    await store.create_task(_corpus())

    entries = parse_registry(client.body)
    assert entries["default"].params["cap"] == 0.9
    assert entries["corpus-run"].params["cap"] == 0.1


@pytest.mark.asyncio
async def test_only_the_params_change_never_the_contract_or_its_digest(fake):
    before = _entry("default", 1.0)
    client = fake(render_registry({"default": before}))

    await store.set_task_cap("default", 0.9)

    after = parse_registry(client.body)["default"]
    assert after.params == {**before.params, "cap": 0.9}
    assert (after.profile_id, after.profile_sha256, after.contract, after.mechanism) == (
        before.profile_id, before.profile_sha256, before.contract, before.mechanism)


@pytest.mark.asyncio
async def test_raising_a_cap_past_the_pool_is_refused(fake):
    client = fake(render_registry({"default": _entry("default", 0.9), "corpus-run": _corpus()}))
    body = client.body

    with pytest.raises(RegistryError, match="total"):
        await store.set_task_cap("default", 0.95)
    assert client.body == body


@pytest.mark.asyncio
async def test_a_retired_task_is_refused(fake):
    fake(render_registry({"default": replace(_entry("default", 0.5), status="retired", retired_at=7)}))

    with pytest.raises(RegistryError, match="retired"):
        await store.set_task_cap("default", 0.4)


@pytest.mark.asyncio
async def test_an_unknown_task_is_refused(fake):
    fake(render_registry({"default": _entry("default", 0.5)}))

    with pytest.raises(RegistryError, match="not in the registry"):
        await store.set_task_cap("nope", 0.4)


@pytest.mark.asyncio
async def test_a_corpus_task_keeps_its_price_pinned(fake):
    client = fake(render_registry({"default": _entry("default", 0.8), "corpus-run": _corpus()}))

    with pytest.raises(RegistryError, match="floor"):
        await store.set_task_cap("corpus-run", 0.2, floor=0.1)

    # Without an explicit floor, the pin follows the cap.
    await store.set_task_cap("corpus-run", 0.2)
    params = parse_registry(client.body)["corpus-run"].params
    assert params["cap"] == params["floor"] == 0.2


@pytest.mark.asyncio
async def test_an_rl_floor_above_the_new_cap_is_refused(fake):
    fake(render_registry({"default": _entry("default", 1.0)}))

    with pytest.raises(RegistryError, match="floor"):
        await store.set_task_cap("default", 0.9, floor=0.95)


@pytest.mark.asyncio
async def test_a_lost_race_is_reapplied_against_the_winner(fake):
    client = fake(render_registry({"default": _entry("default", 1.0)}))
    # The rival lowered default itself and declared a task in the room.
    client.steal_once = render_registry(
        {"default": _entry("default", 0.8), "rival": _entry("rival", 0.2)})

    await store.set_task_cap("default", 0.7)

    entries = parse_registry(client.body)
    assert set(entries) == {"default", "rival"}
    assert entries["default"].params["cap"] == 0.7


@pytest.mark.asyncio
async def test_a_lost_race_that_no_longer_fits_is_refused(fake):
    client = fake(render_registry({"default": _entry("default", 0.5)}))
    client.steal_once = render_registry(
        {"default": _entry("default", 0.5), "rival": _entry("rival", 0.5)})

    with pytest.raises(RegistryError, match="total"):
        await store.set_task_cap("default", 0.6)


def test_the_cli_sets_the_cap(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    calls = []

    async def _set(task_id, cap, *, floor=None, **kwargs):
        calls.append((task_id, cap, floor))

    monkeypatch.setattr(store, "set_task_cap", _set)
    result = CliRunner().invoke(app, ["tasks", "set-cap", "--task-id", "default", "--cap", "0.9"])

    assert result.exit_code == 0, result.output
    assert calls == [("default", 0.9, None)]


def test_the_cli_names_a_refusal_and_exits_non_zero(monkeypatch):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    async def _set(task_id, cap, *, floor=None, **kwargs):
        raise RegistryError("task 'nope' is not in the registry")

    monkeypatch.setattr(store, "set_task_cap", _set)
    result = CliRunner().invoke(app, ["tasks", "set-cap", "--task-id", "nope", "--cap", "0.9"])

    assert result.exit_code == 1
    assert "not in the registry" in result.output
