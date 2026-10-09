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
@pytest.mark.parametrize("failing", ["adopt", "announce", "mark"])
async def test_service_error_after_installation_never_returns_old_revision_fallback(tmp_path, monkeypatch, failing):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)   # NEXT is a child of the root
    service, manifest, stage, _ = staged_service(tmp_path, runtime)
    if failing == "adopt":
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

    assert order == ["install", "adopt", "consumption", "announce"]          # I2: adoption first
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

    # Restart on --resume-from THIRD (the head): re-selected, idempotent.
    restarted = build(path, contract, now=1)
    service, _, _, _ = staged_service(tmp_path, restarted, revision=THIRD, parent=NEXT, checkpoint_n=2, files=child,
                                      installed=(1, NEXT))
    await ValidationService._swap_staged_checkpoint(service, 3)
    service._checkpoint_store.install_external.assert_called_once_with(2, THIRD)
    assert restarted.checkpoint == {"checkpoint_n": 2, "repo": "models/test", "revision": THIRD,
                                    "sha256": canonical_sha256(child)}
    assert restarted.db.execute("SELECT COUNT(*) FROM service_checkpoints").fetchone()[0] == 3
    # The same revision published with other files is not the revision that was adopted.
    service, _, stage, _ = staged_service(tmp_path, restarted, revision=THIRD, parent=NEXT, checkpoint_n=2,
                                          files=FILES, installed=(1, NEXT))
    await ValidationService._swap_staged_checkpoint(service, 4)
    assert_not_installed(service, stage)
    assert restarted.checkpoint["revision"] == THIRD
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
        plan = ValidationService._service_window_plan(service, 2)       # NEXT is the head: idempotent
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


def _three_deep(tmp_path):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    runtime.adopt(checkpoint_n=1, repo="models/test", revision=NEXT, sha256=canonical_sha256(FILES))
    runtime.adopt(checkpoint_n=2, repo="models/test", revision=THIRD, sha256=canonical_sha256(FILES))
    return runtime


def _order(runtime):
    return runtime.db.execute("SELECT revision, seq FROM service_checkpoints ORDER BY seq").fetchall()


@pytest.mark.asyncio
async def test_j1_a_staged_swap_naming_an_ancestor_is_refused_before_any_mutation(tmp_path):
    runtime = _three_deep(tmp_path)
    before = _order(runtime)
    service, _, stage, _ = staged_service(tmp_path, runtime, revision=NEXT, parent=ROOT, checkpoint_n=1,
                                          installed=(2, THIRD))
    await ValidationService._swap_staged_checkpoint(service, 3)
    assert_not_installed(service, stage)
    assert runtime.checkpoint["revision"] == THIRD and _order(runtime) == before
    with pytest.raises(ValueError, match="ancestor of the current lineage head"):
        runtime.require_adoptable(checkpoint_n=1, repo="models/test", revision=NEXT,
                                  sha256=canonical_sha256(FILES), parent_revision=ROOT)
    runtime.close()


def test_j1_adopt_and_ensure_checkpoint_never_move_a_row_back_to_the_head(tmp_path):
    runtime = _three_deep(tmp_path)
    before = _order(runtime)
    for revision, number in ((ROOT, 0), (NEXT, 1)):
        with pytest.raises(ValueError, match="ancestor of the current lineage head"):
            runtime.ensure_checkpoint(checkpoint_n=number, repo="models/test", revision=revision)
        with pytest.raises(ValueError, match="ancestor of the current lineage head"):
            runtime.adopt(checkpoint_n=number, repo="models/test", revision=revision,
                          sha256=canonical_sha256(FILES) if revision == NEXT else runtime.contract.to_dict()["checkpoint"]["sha256"])
    assert _order(runtime) == before and runtime.checkpoint["revision"] == THIRD
    runtime.close()


def test_j1_re_adopting_the_head_is_idempotent(tmp_path):
    runtime = _three_deep(tmp_path)
    before = _order(runtime)
    digest = canonical_sha256(FILES)
    assert runtime.adopt(checkpoint_n=2, repo="models/test", revision=THIRD, sha256=digest)["revision"] == THIRD
    assert runtime.ensure_checkpoint(checkpoint_n=2, repo="models/test", revision=THIRD)["revision"] == THIRD
    runtime.require_adoptable(checkpoint_n=2, repo="models/test", revision=THIRD, sha256=digest,
                              parent_revision=NEXT)
    assert _order(runtime) == before
    runtime.close()


def test_j1_a_restart_with_resume_from_an_ancestor_is_refused_at_the_boundary(tmp_path):
    runtime = _three_deep(tmp_path)
    before = _order(runtime)
    installed = SimpleNamespace(repo_id="models/test", revision=NEXT, checkpoint_n=1)
    service = SimpleNamespace(
        _service_runtime=runtime, _service_schedule_store=SimpleNamespace(take=lambda: None),
        _checkpoint_store=SimpleNamespace(current_manifest=lambda: installed),
        env_mix=[(MATH, 16), (CODE, 16)], _emission_cap=0.5,
        _service_activation_version=lambda name: None, _require_service_environments=lambda schedule: None,
        _service_window_pool=lambda schedule, order: {name: 0.25 for name in order},
    )
    with pytest.raises(ValueError, match="ancestor of the current lineage head"):
        ValidationService._service_window_plan(service, 4)
    assert _order(runtime) == before and runtime.checkpoint["revision"] == THIRD
    runtime.close()


def test_a_refused_ancestor_leaves_the_operators_schedule_request_unapplied(tmp_path):
    """Item 10: the lineage check comes BEFORE the schedule request is consumed for the window."""
    runtime = _three_deep(tmp_path)
    runtime.apply_pending_schedule_request = MagicMock(side_effect=AssertionError("request consumed before the lineage check"))
    installed = SimpleNamespace(repo_id="models/test", revision=NEXT, checkpoint_n=1)
    service = SimpleNamespace(
        _service_runtime=runtime, _service_schedule_store=SimpleNamespace(take=lambda: None),
        _checkpoint_store=SimpleNamespace(current_manifest=lambda: installed),
        env_mix=[(MATH, 16), (CODE, 16)], _emission_cap=0.5,
        _service_activation_version=lambda name: None, _require_service_environments=lambda schedule: None,
        _service_window_pool=lambda schedule, order: {name: 0.25 for name in order},
    )
    with pytest.raises(ValueError, match="ancestor of the current lineage head"):
        ValidationService._service_window_plan(service, 4)
    runtime.apply_pending_schedule_request.assert_not_called()
    runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("revision, number, message", [
    (NEXT, 1, "ancestor of the current lineage head"),
    ("c" * 40, 9, "no adopted service lineage entry"),
])
async def test_boot_refuses_a_resume_target_outside_the_lineage_head(tmp_path, monkeypatch, revision, number, message):
    """Item 10: the configured resume target is checked at start, before any weights are loaded."""
    runtime = _three_deep(tmp_path)
    monkeypatch.setattr("reliquary.validator.resume.resolve_resume_source",
                        lambda source, **kw: (str(tmp_path / "weights"), number))
    loaded = MagicMock(side_effect=AssertionError("weights loaded before the lineage check"))
    service = SimpleNamespace(_resume_from=f"sha:{revision}", _service_runtime=runtime,
                              _checkpoint_store=SimpleNamespace(repo_id="models/test"), _load_model_fn=loaded)
    with pytest.raises(ValueError, match=message) as refused:
        await ValidationService._apply_resume_from(service)
    # M3: the operator is told which revision to resume from (the full lineage head).
    assert f"resume from the lineage head {THIRD}" in str(refused.value)
    loaded.assert_not_called()
    runtime.require_resumable(checkpoint_n=2, repo="models/test", revision=THIRD)   # the head itself is fine
    runtime.close()


# ---- I2: an installed checkpoint is never left unadopted for good ------------------------------------------

def _boundary_service(runtime, installed, receipt_memory=None):
    return SimpleNamespace(
        _service_runtime=runtime, _service_schedule_store=SimpleNamespace(take=lambda: None),
        _checkpoint_store=SimpleNamespace(current_manifest=lambda: installed),
        env_mix=[(MATH, 16), (CODE, 16)], _emission_cap=0.5,
        _service_activation_version=lambda name: None, _require_service_environments=lambda schedule: None,
        _service_window_pool=lambda schedule, order: {name: 0.25 for name in order},
        _service_installed_receipt=receipt_memory,
    )


@pytest.mark.asyncio
async def test_record_consumption_raising_does_not_prevent_adoption(tmp_path, monkeypatch, caplog):
    import logging
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    service, manifest, stage, receipt = staged_service(tmp_path, runtime)
    monkeypatch.setattr(runtime, "record_consumption", MagicMock(side_effect=OSError("unit journal failure")))
    with caplog.at_level(logging.ERROR, logger="reliquary"):
        await ValidationService._swap_staged_checkpoint(service, 1)              # no FatalProofPlaneError
    assert runtime.checkpoint["revision"] == NEXT and runtime.checkpoint["sha256"] == canonical_sha256(FILES)
    assert service._checkpoint_n == 1
    service.server.set_current_checkpoint.assert_called_once()
    service._checkpoint_intake.mark_installed.assert_called_once_with(manifest["revision"], stage)
    assert any("consumption not recorded" in r.getMessage() for r in caplog.records)
    runtime.close()


@pytest.mark.asyncio
async def test_a_crash_after_install_before_adopt_heals_at_the_next_boundary(tmp_path, monkeypatch):
    path = tmp_path / "runtime.sqlite3"
    runtime = build(path)
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    service, manifest, stage, receipt = staged_service(tmp_path, runtime)
    real_adopt = runtime.adopt
    monkeypatch.setattr(runtime, "adopt", MagicMock(side_effect=OSError("crash between install and adopt")))
    with pytest.raises(FatalProofPlaneError):
        await ValidationService._swap_staged_checkpoint(service, 1)
    assert service._checkpoint_store.current.revision == NEXT                    # installed ...
    assert runtime.db.execute("SELECT COUNT(*) FROM service_checkpoints").fetchone()[0] == 1   # ... never adopted
    assert service._service_installed_receipt == (NEXT, receipt)
    assert runtime.pending_install() == (NEXT, receipt)                         # N1: persisted before the install
    monkeypatch.setattr(runtime, "adopt", real_adopt)
    installed = SimpleNamespace(repo_id="models/test", revision=NEXT, checkpoint_n=1)
    # With the receipt, the next boundary adopts the installed child of the head, loudly.
    from reliquary.services import runtime as runtime_module
    warnings = []
    monkeypatch.setattr(runtime_module.logger, "warning", lambda *args: warnings.append(args[0] % args[1:]))
    plan = ValidationService._service_window_plan(
        _boundary_service(runtime, installed, service._service_installed_receipt), 2)
    assert plan["checkpoint_revision"] == NEXT
    assert runtime.checkpoint == {"checkpoint_n": 1, "repo": "models/test", "revision": NEXT,
                                  "sha256": canonical_sha256(FILES)}
    assert any("healed" in message for message in warnings)
    assert runtime.pending_install() is None                                    # cleared by the adoption
    runtime.close()
    # A restart reads the same lineage: the healed head is re-selected, idempotent.
    restarted = build(path, now=1)
    assert ValidationService._service_window_plan(_boundary_service(restarted, installed), 3)["checkpoint_revision"] == NEXT
    restarted.close()


@pytest.mark.parametrize("case", ["not_a_child", "ancestor_parent", "other_number", "no_files"])
def test_the_heal_never_adopts_a_revision_that_is_not_the_heads_child(tmp_path, case):
    runtime = _three_deep(tmp_path)                                              # ROOT -> NEXT -> THIRD
    before = _order(runtime)
    revision, number = "c" * 40, 3
    receipt = {"parent_revision": THIRD, "manifest": {"checkpoint_n": number, "repo_id": "models/test"},
               "files": FILES}
    if case == "not_a_child":
        receipt["parent_revision"] = "9" * 40
    elif case == "ancestor_parent":
        receipt["parent_revision"] = NEXT
    elif case == "other_number":
        receipt["manifest"]["checkpoint_n"] = 9
    else:
        receipt["files"] = {}
    with pytest.raises(ValueError):
        runtime.ensure_checkpoint(checkpoint_n=number, repo="models/test", revision=revision, receipt=receipt)
    with pytest.raises(ValueError):
        runtime.require_resumable(checkpoint_n=number, repo="models/test", revision=revision, receipt=receipt)
    assert _order(runtime) == before and runtime.checkpoint["revision"] == THIRD
    # the same receipt naming the head as its parent heals
    receipt.update(parent_revision=THIRD, manifest={"checkpoint_n": number, "repo_id": "models/test"}, files=FILES)
    runtime.require_resumable(checkpoint_n=number, repo="models/test", revision=revision, receipt=receipt)
    assert _order(runtime) == before                                             # the boot check only reads
    assert runtime.ensure_checkpoint(checkpoint_n=number, repo="models/test", revision=revision,
                                     receipt=receipt)["revision"] == revision
    runtime.close()


@pytest.mark.asyncio
async def test_boot_accepts_an_unadopted_child_of_the_head_with_its_receipt_and_keeps_it(tmp_path, monkeypatch):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    weights = tmp_path / "weights"
    weights.mkdir()
    receipt = {"parent_revision": ROOT, "manifest": {"checkpoint_n": 1, "repo_id": "models/test",
                                                     "trained_window_cursor": 0}, "files": FILES}
    (weights / PUBLICATION_RECEIPT).write_text(json.dumps(receipt))
    monkeypatch.setattr("reliquary.validator.resume.resolve_resume_source", lambda source, **kw: (str(weights), 1))

    class Reached(Exception):
        pass

    def past_the_lineage_check(*args, **kwargs):
        raise Reached()
    monkeypatch.setattr("reliquary.validator.checkpoint_profile.validate_checkpoint_profile", past_the_lineage_check)
    service = SimpleNamespace(_resume_from=f"sha:{NEXT}", _service_runtime=runtime,
                              _checkpoint_store=SimpleNamespace(repo_id="models/test"),
                              _service_installed_receipt=None)
    with pytest.raises(Reached):
        await ValidationService._apply_resume_from(service)
    assert service._service_installed_receipt == (NEXT, receipt)
    assert runtime.checkpoint["revision"] == ROOT                                # adopted at the boundary, not here
    runtime.close()


# ---- N1: the heal survives the restart that a failed adoption forces -------------------------------------

def test_n1_the_persisted_install_heals_in_a_new_runtime_without_any_receipt_and_is_cleared(tmp_path):
    path = tmp_path / "runtime.sqlite3"
    runtime = build(path)
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    receipt = {"parent_revision": ROOT, "manifest": {"checkpoint_n": 1, "repo_id": "models/test"}, "files": FILES}
    runtime.record_pending_install(revision=NEXT, receipt=receipt)
    runtime.close()                                                              # the process dies here
    restarted = build(path, now=1)
    assert restarted.pending_install() == (NEXT, receipt)
    restarted.require_resumable(checkpoint_n=1, repo="models/test", revision=NEXT)   # the boot twin agrees
    assert restarted.ensure_checkpoint(checkpoint_n=1, repo="models/test", revision=NEXT)["revision"] == NEXT
    assert restarted.checkpoint["sha256"] == canonical_sha256(FILES)
    assert restarted.pending_install() is None
    restarted.close()


def test_n1_without_a_persisted_install_for_that_revision_the_refusal_stands(tmp_path):
    runtime = build(tmp_path / "runtime.sqlite3")
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    with pytest.raises(ValueError, match="no adopted service lineage entry"):
        runtime.ensure_checkpoint(checkpoint_n=1, repo="models/test", revision=NEXT)
    receipt = {"parent_revision": ROOT, "manifest": {"checkpoint_n": 1, "repo_id": "models/test"}, "files": FILES}
    runtime.record_pending_install(revision=THIRD, receipt=receipt)            # another revision's row
    with pytest.raises(ValueError, match="no adopted service lineage entry"):
        runtime.ensure_checkpoint(checkpoint_n=1, repo="models/test", revision=NEXT)
    assert runtime.checkpoint["revision"] == ROOT
    runtime.close()


@pytest.mark.parametrize("case", ["not_a_child", "other_number", "no_files"])
def test_n1_a_persisted_receipt_goes_through_the_same_heal_checks(tmp_path, case):
    path = tmp_path / "runtime.sqlite3"
    runtime = build(path)
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    receipt = {"parent_revision": ROOT, "manifest": {"checkpoint_n": 1, "repo_id": "models/test"}, "files": FILES}
    if case == "not_a_child":
        receipt["parent_revision"] = "9" * 40
    elif case == "other_number":
        receipt["manifest"]["checkpoint_n"] = 7
    else:
        receipt["files"] = {}
    runtime.record_pending_install(revision=NEXT, receipt=receipt)
    runtime.close()
    restarted = build(path, now=1)
    with pytest.raises(ValueError):
        restarted.ensure_checkpoint(checkpoint_n=1, repo="models/test", revision=NEXT)
    with pytest.raises(ValueError):
        restarted.require_resumable(checkpoint_n=1, repo="models/test", revision=NEXT)
    assert restarted.checkpoint["revision"] == ROOT
    restarted.close()


@pytest.mark.asyncio
async def test_n1_a_failed_adoption_heals_at_the_first_boundary_of_a_new_validation_service(tmp_path, monkeypatch):
    from tests.unit.test_service_window_build import _boot, _bootable_contract, _persisted_folder

    contract = _bootable_contract()
    folder = _persisted_folder(tmp_path)
    folder.chmod(0o700)                                                          # the operator's folder mode
    path = folder / "runtime.sqlite3"
    runtime = ServiceRuntime(path, contract, qualification_v2(contract), now=0)
    runtime.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=ROOT)
    service, manifest, stage, receipt = staged_service(tmp_path, runtime)
    monkeypatch.setattr(runtime, "adopt", MagicMock(side_effect=OSError("crash between install and adopt")))
    with pytest.raises(FatalProofPlaneError):
        await ValidationService._swap_staged_checkpoint(service, 1)
    assert service._checkpoint_store.current.revision == NEXT                    # installed, never adopted
    runtime.close()                                                              # the process restarts

    svc = _boot(monkeypatch, tmp_path, contract, loaded=[MATH, CODE])          # the real constructor, scoped
    assert svc._service_installed_receipt is None                                # the memory is gone
    installed = SimpleNamespace(repo_id="models/test", revision=NEXT, checkpoint_n=1)
    svc._checkpoint_store = SimpleNamespace(current_manifest=lambda: installed)
    plan = svc._service_window_plan(2)                                           # the first boundary
    assert plan["checkpoint_revision"] == NEXT
    assert svc._service_runtime.checkpoint == {"checkpoint_n": 1, "repo": "models/test", "revision": NEXT,
                                               "sha256": canonical_sha256(FILES)}
    assert svc._service_runtime.pending_install() is None
    svc._service_runtime.close()
