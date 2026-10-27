"""Public HF transport cannot change the original signed native checkpoint."""

from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import huggingface_hub
import pytest

from reliquary.integrations import affine_hf_checkpoint as hf
from reliquary.integrations import affine_native_evaluation as native
from reliquary.integrations.affine import AffineRuntimeError
from tests.unit.test_affine_native_evaluation import intake, save, sign

REVISION = "a" * 40


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    config_path, config, job, manifest = intake(tmp_path)
    checkpoint = Path(config["checkpointDirectory"])
    cache = tmp_path / "cache"
    checkpoint.rename(cache)
    info = SimpleNamespace(
        sha=REVISION, private=False, gated=False,
        siblings=[SimpleNamespace(rfilename=path.name, size=path.stat().st_size) for path in cache.iterdir()],
    )
    calls = []

    class Api:
        def __init__(self, **kwargs):
            assert kwargs == {"token": False}

        def model_info(self, repository, **kwargs):
            calls.append(("metadata", repository, kwargs))
            return info

    def download(repository, filename, **kwargs):
        calls.append(("download", repository, filename, kwargs))
        return str(cache / filename)

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(cache))
    return SimpleNamespace(path=config_path, config=config, job=job, manifest=manifest,
                           checkpoint=checkpoint, cache=cache, info=info, calls=calls)


def materialize(value, **kwargs):
    return hf.materialize(value.path, repository="owner/model", revision=REVISION, max_bytes=1024, **kwargs)


def test_pinned_cache_symlinks_become_exact_private_regular_files_with_bound_provenance(inputs, tmp_path):
    cached = inputs.cache / "model.safetensors"
    blob = inputs.cache / "blob"
    cached.rename(blob)
    cached.symlink_to(blob)
    result = materialize(inputs)
    assert set(result) == {"schema", "repository", "revision", "checkpointId", "checkpointInventoryDigest"}
    assert result["schema"] == "hf-checkpoint-source/v1"
    assert result["repository"] == "owner/model" and result["revision"] == REVISION
    assert result["checkpointId"] == inputs.manifest["checkpoint"]["id"]
    assert {path.name for path in inputs.checkpoint.iterdir()} == set(inputs.manifest["checkpoint"]["files"])
    for path in inputs.checkpoint.iterdir():
        assert not path.is_symlink() and path.stat().st_mode & 0o777 == 0o600
        assert path.stat().st_ino != (inputs.cache / path.name).stat().st_ino
    assert inputs.checkpoint.stat().st_mode & 0o777 == 0o700
    source = inputs.checkpoint.with_name(inputs.checkpoint.name + ".hf-source.private.json")
    assert json.loads(source.read_bytes()) == result and source.stat().st_mode & 0o777 == 0o600
    bundle = tmp_path / "bundle"
    native.prepare(inputs.path, bundle)
    descriptor, *_ = native.load_bundle(bundle)
    assert descriptor["checkpointInventoryDigest"] == result["checkpointInventoryDigest"]
    assert all(call[-1]["revision"] == REVISION for call in inputs.calls)
    assert not list(tmp_path.glob(".hf-checkpoint-*"))


@pytest.mark.parametrize("revision", ["main", "a" * 39, "A" * 40, "a" * 41, "../revision"])
def test_mutable_or_invalid_revision_is_refused_before_network(inputs, revision):
    with pytest.raises(ValueError, match="immutable revision"):
        hf.materialize(inputs.path, repository="owner/model", revision=revision, max_bytes=1024)
    assert inputs.calls == [] and not inputs.checkpoint.exists()


@pytest.mark.parametrize("change", ["private", "gated", "revision", "missing", "unknown-size", "budget", "disk"])
def test_metadata_or_capacity_mismatch_is_refused_before_download(inputs, monkeypatch, change):
    if change in ("private", "gated"):
        setattr(inputs.info, change, True)
    elif change == "revision":
        inputs.info.sha = "b" * 40
    elif change == "missing":
        inputs.info.siblings.pop()
    elif change == "unknown-size":
        inputs.info.siblings[0].size = None
    elif change == "budget":
        inputs.info.siblings[0].size = 2048
    else:
        monkeypatch.setattr(hf.shutil, "disk_usage", lambda _: SimpleNamespace(free=0))
    with pytest.raises(ValueError):
        materialize(inputs)
    assert len(inputs.calls) == 1 and not inputs.checkpoint.exists()


@pytest.mark.parametrize("change", ["signature", "expired"])
def test_original_job_admission_precedes_network(inputs, change):
    job = deepcopy(inputs.job)
    if change == "expired":
        job.update(created_at=1, expires_at=2)
        envelope = sign(job)
    else:
        envelope = sign(job)
        envelope["payload"]["job_id"] = "changed"
    save(Path(inputs.config["jobEnvelope"]), envelope)
    with pytest.raises(ValueError):
        materialize(inputs)
    assert inputs.calls == [] and not inputs.checkpoint.exists()


def test_shared_cache_must_stay_outside_repositories(inputs):
    (inputs.cache / ".git").write_text("repository-marker")
    with pytest.raises(AffineRuntimeError, match="outside repositories"):
        materialize(inputs)
    assert len(inputs.calls) == 1 and not inputs.checkpoint.exists()


@pytest.mark.parametrize("change", ["hash", "size", "download", "inventory"])
def test_failed_or_changed_download_cleans_partial_export_without_modifying_cache(inputs, tmp_path, monkeypatch, change):
    original = huggingface_hub.hf_hub_download

    def changed(repository, filename, **kwargs):
        path = Path(original(repository, filename, **kwargs))
        if filename == "model.safetensors":
            if change == "hash":
                path.write_bytes(b"x" * path.stat().st_size)
            elif change == "size":
                path.write_bytes(b"short")
            elif change == "download":
                raise OSError("interrupted transport")
            else:
                job = deepcopy(inputs.job)
                job["job_id"] = "different-authorized-job"
                save(Path(inputs.config["jobEnvelope"]), sign(job))
        return str(path)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", changed)
    with pytest.raises((ValueError, OSError)):
        materialize(inputs)
    assert not inputs.checkpoint.exists()
    assert not inputs.checkpoint.with_name(inputs.checkpoint.name + ".hf-source.private.json").exists()
    assert not list(tmp_path.glob(".hf-checkpoint-*"))
    assert (inputs.cache / "config.json").read_bytes() == b"{}"


def test_existing_export_is_preserved_and_cli_can_materialize_then_prepare(inputs, tmp_path, capsys):
    from reliquary.integrations.affine_native_evaluation_cli import main

    bundle = tmp_path / "bundle"
    assert main(["prepare", "--config", str(inputs.path), "--out", str(bundle),
                 "--hf-repository", "owner/model", "--hf-revision", REVISION,
                 "--max-checkpoint-bytes", "1024"]) == 0
    assert json.loads(capsys.readouterr().out)["stage"] == "prepared"
    previous = {path.name: path.read_bytes() for path in inputs.checkpoint.iterdir()}
    calls = len(inputs.calls)
    with pytest.raises(ValueError, match="new checkpoint"):
        materialize(inputs)
    assert len(inputs.calls) == calls
    assert {path.name: path.read_bytes() for path in inputs.checkpoint.iterdir()} == previous
    shutil.rmtree(bundle)


def test_failed_atomic_publication_cleans_only_its_partial_export(inputs, tmp_path, monkeypatch):
    original = hf.os.link
    sidecar = inputs.checkpoint.with_name(inputs.checkpoint.name + ".hf-source.private.json")

    def collision(source, destination):
        if Path(destination) == sidecar:
            sidecar.write_bytes(b"concurrent-owner-file")
            sidecar.chmod(0o600)
        return original(source, destination)

    monkeypatch.setattr(hf.os, "link", collision)
    with pytest.raises(FileExistsError):
        materialize(inputs)
    assert not inputs.checkpoint.exists()
    assert sidecar.read_bytes() == b"concurrent-owner-file"
    assert not list(tmp_path.glob(".hf-checkpoint-*"))
