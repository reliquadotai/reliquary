"""Checkpoint adoption of a service-contract/v2 task: validate first, close after an installation fault."""

import json
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reliquary.protocol.release_contract import canonical_sha256
from reliquary.services.runtime import ServiceRuntime
from reliquary.trainer.publisher import PUBLICATION_RECEIPT
from reliquary.validator.errors import FatalProofPlaneError
from reliquary.validator.service import ValidationService
from reliquary.protocol.service_contract import ServiceContract
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2, contract_v2_dict, qualification_v2

ROOT = "d" * 40
NEXT = "f" * 40
THIRD = "a" * 40
FILES = {"config.json": {"size": 2, "sha256": "e" * 64, "blob_id": "e" * 40}}


def pinned_contract(files=FILES):
    """An order whose root pin is the digest the validator computes for the published root files."""
    value = contract_v2_dict()
    value["checkpoint"]["sha256"] = canonical_sha256(files)
    return ServiceContract.from_dict(value)


def build(path, contract=None, now=0):
    contract = contract or contract_v2()
    return ServiceRuntime(path, contract, qualification_v2(contract), now=now)


def staged_service(tmp_path, runtime, *, repo="models/test", revision=NEXT, parent=ROOT, checkpoint_n=1,
                   files=FILES, installed=(0, ROOT)):
    manifest = {"checkpoint_n": checkpoint_n, "repo_id": repo, "revision": revision, "trained_window_cursor": 0}
    stage = tmp_path / f"stage-{revision[:6]}-{len(list(tmp_path.glob('stage-*')))}"
    stage.mkdir()
    receipt = {"publication_id": "unit-publication", "parent_revision": parent,
               "manifest": {key: value for key, value in manifest.items() if key != "revision"},
               "files": files}
    (stage / PUBLICATION_RECEIPT).write_text(json.dumps(receipt))
    old = SimpleNamespace(checkpoint_n=installed[0], revision=installed[1])
    store = SimpleNamespace(repo_id=manifest["repo_id"], current=old)

    def install(number, revision):
        store.current = SimpleNamespace(checkpoint_n=number, revision=revision)
        return store.current
    store.install_external = MagicMock(side_effect=install)
    intake = SimpleNamespace(take_staged=MagicMock(return_value=(manifest, stage)), mark_installed=MagicMock())
    service = SimpleNamespace(
        _checkpoint_intake=intake, _checkpoint_store=store, _service_runtime=runtime,
        _record_fill_closed_checkpoint_candidate=MagicMock(), _network_proof=True,
        _proof_worker_pool=SimpleNamespace(bind_checkpoint=MagicMock()), proof_scheduler=None,
        server=SimpleNamespace(set_current_checkpoint=MagicMock()), _checkpoint_n=0,
        _training_accumulator=SimpleNamespace(reset=MagicMock()), _windows_since_checkpoint_swap=10,
    )
    return service, manifest, stage, receipt


def assert_nothing_moved(service, runtime, stage):
    service._record_fill_closed_checkpoint_candidate.assert_not_called()
    service._proof_worker_pool.bind_checkpoint.assert_not_called()
    service._checkpoint_store.install_external.assert_not_called()
    service.server.set_current_checkpoint.assert_not_called()
    assert service._checkpoint_n == 0 and not stage.exists()
    assert runtime.checkpoint["revision"] == ROOT
    assert runtime.db.execute("SELECT COUNT(*) FROM service_checkpoints").fetchone()[0] == 1
    assert runtime.db.execute("SELECT COUNT(*) FROM service_consumption").fetchone()[0] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["missing", "json", "manifest", "files"])
async def test_service_receipt_error_precedes_all_checkpoint_mutations(tmp_path, invalid):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    service, _, stage, receipt = staged_service(tmp_path, runtime)
    path = stage / PUBLICATION_RECEIPT
    if invalid == "missing":
        path.unlink()
    elif invalid == "json":
        path.write_text("invalid JSON")
    else:
        receipt[invalid] = {} if invalid == "files" else {"trained_window_cursor": 2}
        path.write_text(json.dumps(receipt))
    await ValidationService._swap_staged_checkpoint(service, 1)
    assert_nothing_moved(service, runtime, stage)
    runtime.close()


@pytest.mark.asyncio
async def test_a_checkpoint_of_another_repository_is_refused_before_any_mutation(tmp_path):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    service, _, stage, _ = staged_service(tmp_path, runtime, repo="models/other")
    await ValidationService._swap_staged_checkpoint(service, 1)
    assert_nothing_moved(service, runtime, stage)
    runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["consumption", "adopt", "announce", "mark"])
async def test_service_error_after_installation_never_returns_old_revision_fallback(tmp_path, monkeypatch, failing):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)   # NEXT is a child of the root
    service, manifest, stage, _ = staged_service(tmp_path, runtime)
    if failing == "consumption":
        monkeypatch.setattr(runtime, "record_consumption", MagicMock(side_effect=OSError("unit journal failure")))
    elif failing == "adopt":
        monkeypatch.setattr(runtime, "adopt", MagicMock(side_effect=OSError("unit journal failure")))
    elif failing == "announce":
        service.server.set_current_checkpoint.side_effect = OSError("unit server failure")
    else:
        service._checkpoint_intake.mark_installed.side_effect = OSError("unit intake failure")
    with pytest.raises(FatalProofPlaneError, match="installation/adoption failed.*restart"):
        await ValidationService._swap_staged_checkpoint(service, 1)
    assert service._checkpoint_store.current.revision == manifest["revision"]
    assert stage.exists()
    service._training_accumulator.reset.assert_not_called()
    runtime.close()


@pytest.mark.asyncio
async def test_service_adoption_follows_installation_keeps_the_run_and_runs_off_the_event_loop(tmp_path, monkeypatch):
    contract = contract_v2()
    runtime = build(tmp_path / "runtime.sqlite3", contract)
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    schedule = runtime.schedule
    service, manifest, stage, receipt = staged_service(tmp_path, runtime)
    order, threads = [], []
    real_adopt, real_consumption = runtime.adopt, runtime.record_consumption
    service._checkpoint_store.install_external.side_effect = (
        lambda number, revision, _real=service._checkpoint_store.install_external.side_effect:
        (order.append("install"), _real(number, revision))[1])
    monkeypatch.setattr(runtime, "record_consumption", lambda cursor: (
        order.append("consumption"), threads.append(threading.get_ident()), real_consumption(cursor))[2])
    monkeypatch.setattr(runtime, "adopt", lambda **kwargs: (
        order.append("adopt"), threads.append(threading.get_ident()), real_adopt(**kwargs))[2])
    service.server.set_current_checkpoint.side_effect = lambda entry: order.append("announce")

    await ValidationService._swap_staged_checkpoint(service, 1)

    assert order == ["install", "consumption", "adopt", "announce"]
    assert threading.get_ident() not in threads                       # SQLite commits leave the event loop
    assert runtime.checkpoint == {"checkpoint_n": 1, "repo": "models/test", "revision": NEXT,
                                  "sha256": canonical_sha256(receipt["files"])}
    # One order for the whole run: the contract, its schedule and its lineage root are untouched.
    assert runtime.contract == runtime.order_contract == contract and runtime.schedule == schedule
    assert runtime.db.execute("SELECT revision FROM service_checkpoints ORDER BY seq").fetchall() == [(ROOT,), (NEXT,)]
    assert service._checkpoint_n == 1
    service.server.set_current_checkpoint.assert_called_once_with(service._checkpoint_store.current)
    service._checkpoint_intake.mark_installed.assert_called_once_with(manifest["revision"], stage)
    service._training_accumulator.reset.assert_called_once()
    assert runtime.db.execute("SELECT cursor FROM service_consumption").fetchone()[0] == 0
    runtime.close()


def assert_not_installed(service, stage):
    service._record_fill_closed_checkpoint_candidate.assert_not_called()
    service._proof_worker_pool.bind_checkpoint.assert_not_called()
    service._checkpoint_store.install_external.assert_not_called()
    service.server.set_current_checkpoint.assert_not_called()
    service._checkpoint_intake.mark_installed.assert_not_called()
    assert not stage.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ["not_the_root", "root_with_another_digest"])
async def test_the_first_adoption_must_be_the_orders_pinned_root(tmp_path, case):
    """I1: a wrong --resume-from or a stale candidate never becomes the lineage root."""
    runtime = build(tmp_path / "runtime.sqlite3", pinned_contract())            # nothing adopted yet
    if case == "not_the_root":
        service, _, stage, _ = staged_service(tmp_path, runtime, revision=NEXT, parent=ROOT, checkpoint_n=1)
    else:
        other = {"config.json": {"size": 3, "sha256": "1" * 64, "blob_id": "1" * 40}}
        service, _, stage, _ = staged_service(tmp_path, runtime, revision=ROOT, parent="9" * 40, checkpoint_n=0,
                                              files=other)
    await ValidationService._swap_staged_checkpoint(service, 1)                  # refused: old revision kept
    assert_not_installed(service, stage)
    assert service._checkpoint_n == 0 and service._checkpoint_store.current.revision == ROOT
    assert runtime.db.execute("SELECT COUNT(*) FROM service_checkpoints").fetchone()[0] == 0
    assert runtime.db.execute("SELECT COUNT(*) FROM service_consumption").fetchone()[0] == 0
    with pytest.raises(ValueError, match="no service checkpoint has been adopted"):
        runtime.checkpoint
    runtime.close()


@pytest.mark.asyncio
async def test_a_checkpoint_whose_parent_is_not_the_current_one_is_refused_before_any_mutation(tmp_path):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    service, _, stage, _ = staged_service(tmp_path, runtime, revision=THIRD, parent=NEXT, checkpoint_n=2)
    await ValidationService._swap_staged_checkpoint(service, 1)
    assert_nothing_moved(service, runtime, stage)
    # A receipt without a parent at all is no better.
    service, _, stage, receipt = staged_service(tmp_path, runtime)
    del receipt["parent_revision"]
    (stage / PUBLICATION_RECEIPT).write_text(json.dumps(receipt))
    await ValidationService._swap_staged_checkpoint(service, 1)
    assert_nothing_moved(service, runtime, stage)
    runtime.close()


@pytest.mark.asyncio
async def test_the_legitimate_chain_root_then_child_then_restart_is_adopted(tmp_path):
    contract = pinned_contract()
    path = tmp_path / "runtime.sqlite3"
    runtime = build(path, contract)
    digest = canonical_sha256(FILES)
    # First boot: --resume-from is the order's root, published with the pinned files.
    service, _, _, _ = staged_service(tmp_path, runtime, revision=ROOT, parent="9" * 40, checkpoint_n=0,
                                      installed=(0, "0" * 40))
    await ValidationService._swap_staged_checkpoint(service, 0)
    assert runtime.checkpoint == {"checkpoint_n": 0, "repo": "models/test", "revision": ROOT, "sha256": digest}
    # The trainer publishes a child of the root, then a child of that child.
    child = {"model.safetensors": {"size": 9, "sha256": "2" * 64, "blob_id": "2" * 40}}
    service, _, _, _ = staged_service(tmp_path, runtime, revision=NEXT, parent=ROOT, checkpoint_n=1, files=child)
    await ValidationService._swap_staged_checkpoint(service, 1)
    service, _, _, _ = staged_service(tmp_path, runtime, revision=THIRD, parent=NEXT, checkpoint_n=2, files=child,
                                      installed=(1, NEXT))
    await ValidationService._swap_staged_checkpoint(service, 2)
    assert runtime.db.execute("SELECT revision FROM service_checkpoints ORDER BY seq").fetchall() == [
        (ROOT,), (NEXT,), (THIRD,)]
    runtime.close()

    # Restart on --resume-from NEXT (an adopted revision, not a child of the current one): re-selected.
    restarted = build(path, contract, now=1)
    service, _, _, _ = staged_service(tmp_path, restarted, revision=NEXT, parent=ROOT, checkpoint_n=1, files=child,
                                      installed=(2, THIRD))
    await ValidationService._swap_staged_checkpoint(service, 3)
    service._checkpoint_store.install_external.assert_called_once_with(1, NEXT)
    assert restarted.checkpoint == {"checkpoint_n": 1, "repo": "models/test", "revision": NEXT,
                                    "sha256": canonical_sha256(child)}
    assert restarted.db.execute("SELECT COUNT(*) FROM service_checkpoints").fetchone()[0] == 3
    # The same revision published with other files is not the revision that was adopted.
    service, _, stage, _ = staged_service(tmp_path, restarted, revision=THIRD, parent=NEXT, checkpoint_n=2,
                                          files=FILES, installed=(1, NEXT))
    await ValidationService._swap_staged_checkpoint(service, 4)
    assert_not_installed(service, stage)
    assert restarted.checkpoint["revision"] == NEXT
    restarted.close()


@pytest.mark.asyncio
async def test_the_lineage_check_runs_off_the_event_loop(tmp_path, monkeypatch):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    service, _, _, _ = staged_service(tmp_path, runtime)
    threads, real = [], runtime.require_adoptable
    monkeypatch.setattr(runtime, "require_adoptable",
                        lambda **kwargs: (threads.append(threading.get_ident()), real(**kwargs))[1])
    await ValidationService._swap_staged_checkpoint(service, 1)
    assert threads and threading.get_ident() not in threads
    assert runtime.checkpoint["revision"] == NEXT
    runtime.close()


@pytest.mark.asyncio
async def test_legacy_swap_preserves_path_without_service_receipt(tmp_path):
    service, manifest, stage, _ = staged_service(tmp_path, None)
    (stage / PUBLICATION_RECEIPT).unlink()
    await ValidationService._swap_staged_checkpoint(service, 1)
    assert service._checkpoint_n == 1
    service._checkpoint_store.install_external.assert_called_once_with(1, manifest["revision"])
    service._checkpoint_intake.mark_installed.assert_called_once_with(manifest["revision"], stage)


@pytest.mark.parametrize("saved", [False, True])
def test_restart_restores_only_a_revision_adopted_in_the_lineage(tmp_path, saved):
    contract = contract_v2()
    path = tmp_path / "runtime.sqlite3"
    runtime = build(path, contract)
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    if saved:
        runtime.adopt(checkpoint_n=1, repo="models/test", revision=NEXT, sha256="e" * 64)
        runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)   # NEXT is not the latest row
    runtime.close()

    restored = build(path, contract, now=1)
    installed = SimpleNamespace(repo_id="models/test", revision=NEXT, checkpoint_n=1)
    service = SimpleNamespace(
        _service_runtime=restored, _service_schedule_store=SimpleNamespace(take=lambda: None),
        _checkpoint_store=SimpleNamespace(current_manifest=lambda: installed),
        env_mix=[(MATH, 16), (CODE, 16)], _emission_cap=0.5,
        _service_activation_version=lambda name: None, _require_service_environments=lambda schedule: None,
        _service_window_pool=lambda schedule, order: {name: 0.25 for name in order},
    )
    if saved:
        plan = ValidationService._service_window_plan(service, 2)
        assert plan["checkpoint_revision"] == NEXT and plan["window"] == 2
        assert restored.checkpoint == {"checkpoint_n": 1, "repo": "models/test", "revision": NEXT, "sha256": "e" * 64}
    else:
        with pytest.raises(ValueError, match="no adopted service lineage entry"):
            ValidationService._service_window_plan(service, 2)
        assert restored.checkpoint["revision"] == ROOT
    with pytest.raises(ValueError, match="another identity"):         # a known revision keeps its number
        restored.ensure_checkpoint(checkpoint_n=7, repo="models/test",
                                   revision=NEXT if saved else ROOT)
    restored.close()
