"""Service policy adoption validates first and closes after installation faults."""

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reliquary.protocol.release_contract import canonical_sha256
from reliquary.services.runtime import ServiceRuntime
from reliquary.trainer.publisher import PUBLICATION_RECEIPT
from reliquary.validator.errors import FatalProofPlaneError
from reliquary.validator.service import ValidationService
from tests.unit.test_service_runtime import example


def staged_service(tmp_path, runtime):
    contract, _ = example()
    manifest = {"checkpoint_n": 1, "repo_id": contract.to_dict()["checkpoint"]["repo"],
                "revision": "f" * 40, "trained_window_cursor": 0}
    stage = tmp_path / "stage"
    stage.mkdir()
    receipt = {"publication_id": "unit-publication", "parent_revision": contract.to_dict()["checkpoint"]["revision"],
               "manifest": {key: value for key, value in manifest.items() if key != "revision"},
               "files": {"config.json": {"size": 2, "sha256": "e" * 64, "blob_id": "e" * 40}}}
    (stage / PUBLICATION_RECEIPT).write_text(json.dumps(receipt))
    old = SimpleNamespace(checkpoint_n=0, revision=contract.to_dict()["checkpoint"]["revision"])
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


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["missing", "json", "manifest", "files"])
async def test_service_receipt_error_precedes_all_checkpoint_mutations(tmp_path, invalid):
    contract, qualification = example()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
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
    service._record_fill_closed_checkpoint_candidate.assert_not_called()
    service._proof_worker_pool.bind_checkpoint.assert_not_called()
    service._checkpoint_store.install_external.assert_not_called()
    service.server.set_current_checkpoint.assert_not_called()
    assert runtime.contract == contract
    assert service._checkpoint_n == 0
    assert not stage.exists()
    runtime.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["consumption", "adopt", "announce", "mark"])
async def test_service_error_after_installation_never_returns_old_revision_fallback(tmp_path, monkeypatch, failing):
    contract, qualification = example()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
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
async def test_service_context_adoption_uses_prevalidated_receipt_after_installation(tmp_path):
    contract, qualification = example()
    runtime = ServiceRuntime(tmp_path / "runtime.sqlite", contract, qualification, now=0)
    service, manifest, stage, receipt = staged_service(tmp_path, runtime)
    await ValidationService._swap_staged_checkpoint(service, 1)
    assert runtime.contract.to_dict()["checkpoint"] == {
        "repo": manifest["repo_id"], "revision": manifest["revision"], "sha256": canonical_sha256(receipt["files"])
    }
    assert runtime.order_contract == contract
    assert service._checkpoint_n == 1
    service.server.set_current_checkpoint.assert_called_once_with(service._checkpoint_store.current)
    service._checkpoint_intake.mark_installed.assert_called_once_with(manifest["revision"], stage)
    service._training_accumulator.reset.assert_called_once()
    assert runtime.db.execute("SELECT cursor FROM service_consumption").fetchone()[0] == 0
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
def test_restart_restores_only_a_context_saved_for_the_installed_checkpoint(tmp_path, saved):
    contract, qualification = example()
    path = tmp_path / "runtime.sqlite"
    runtime = ServiceRuntime(path, contract, qualification, now=0)
    successor = None
    if saved:
        successor = runtime.adopt(repo=contract.to_dict()["checkpoint"]["repo"], revision="f" * 40, sha256="e" * 64)
    runtime.close()
    restored = ServiceRuntime(path, contract, qualification, now=1)
    assert restored.contract == contract
    checkpoint = SimpleNamespace(repo_id=contract.to_dict()["checkpoint"]["repo"], revision="f" * 40)
    probe = MagicMock(side_effect=RuntimeError("construction gate passed"))
    service = SimpleNamespace(_candidate_activation_nonce=b"bound", _service_runtime=restored,
                              _set_window_preparation_stage=MagicMock(), proof_scheduler=None,
                              _checkpoint_store=SimpleNamespace(current_manifest=lambda: checkpoint),
                              envs={contract.to_dict()["environment"]["id"]: object()},
                              server=SimpleNamespace(operator_by_hotkey_snapshot=probe))
    if saved:
        with pytest.raises(RuntimeError, match="construction gate passed"):
            ValidationService._build_window_batchers(service, 2)
        assert restored.contract == successor
        probe.assert_called_once()
    else:
        with pytest.raises(ValueError, match="no qualified service context"):
            ValidationService._build_window_batchers(service, 2)
        assert restored.contract == contract
        probe.assert_not_called()
    restored.close()
