from __future__ import annotations

import hashlib
import json

import torch

from scripts.qualify_episode_suite import (
    _artifact_digest,
    qualify_adversarial,
    qualify_cpu,
    summarize_model_environment,
)
from scripts.qualify_episode_training_gpu import _model_weight_sha256


def _git_blob_sha1(value: bytes) -> str:
    digest = hashlib.sha1(usedforsecurity=False)
    digest.update(f"blob {len(value)}\0".encode("ascii"))
    digest.update(value)
    return digest.hexdigest()


def _write_hf_receipt(root, revision: str, files: dict[str, bytes]) -> None:
    tree = root / ".cache" / "huggingface" / "trees"
    tree.mkdir(parents=True)
    receipt = {
        "format_version": 1,
        "files": {
            name: {"size": len(value), "blob_id": _git_blob_sha1(value)}
            for name, value in files.items()
        },
    }
    (tree / f"{revision}.json").write_text(json.dumps(receipt))


def test_local_artifact_requires_and_verifies_immutable_revision_receipt(tmp_path):
    revision = "a" * 40
    files = {"config.json": b"{}", "tokenizer.json": b"tokens"}
    for name, value in files.items():
        (tmp_path / name).write_bytes(value)
    _write_hf_receipt(tmp_path, revision, files)

    artifact = _artifact_digest(str(tmp_path), revision)

    assert artifact["files"] == 2
    assert artifact["revision_verified"] is True
    assert artifact["revision_receipt"]["verified"] is True


def test_artifact_digest_ignores_cache_but_receipt_detects_tampering(tmp_path):
    revision = "b" * 40
    files = {"config.json": b"original"}
    (tmp_path / "config.json").write_bytes(files["config.json"])
    _write_hf_receipt(tmp_path, revision, files)
    first = _artifact_digest(str(tmp_path), revision)

    (tmp_path / ".cache" / "transient.lock").write_text("changed")
    second = _artifact_digest(str(tmp_path), revision)
    assert second["sha256"] == first["sha256"]
    assert second["revision_verified"] is True

    (tmp_path / "config.json").write_bytes(b"tampered")
    tampered = _artifact_digest(str(tmp_path), revision)
    assert tampered["revision_verified"] is False
    assert "mismatch" in tampered["revision_receipt"]["reason"]


def test_receipt_may_include_intentionally_undownloaded_metadata(tmp_path):
    revision = "c" * 40
    downloaded = {"config.json": b"{}"}
    receipt_files = {**downloaded, "README.md": b"not downloaded"}
    (tmp_path / "config.json").write_bytes(downloaded["config.json"])
    _write_hf_receipt(tmp_path, revision, receipt_files)

    artifact = _artifact_digest(str(tmp_path), revision)

    assert artifact["revision_verified"] is True
    assert artifact["revision_receipt"]["files"] == 1
    assert artifact["revision_receipt"]["receipt_files"] == 2


def test_complete_model_weight_digest_detects_an_optimizer_change():
    model = torch.nn.Linear(4, 3)
    before = _model_weight_sha256(model)
    with torch.no_grad():
        model.weight[0, 0].add_(1.0)
    after = _model_weight_sha256(model)

    assert len(before) == 64
    assert before != after


def _row(reward: float, *, error=None, exact_replay=True):
    return {
        "reward": reward,
        "error": error,
        "exact_replay": exact_replay,
        "invalid_actions": 0,
        "elapsed_seconds": 1.0,
    }


def test_cpu_and_adversarial_qualification_pass_all_episode_environments():
    names = (
        "reliquary_stateful_tools_v1",
        "reliquary_retrieval_tools_v1",
        "reliquary_workspace_tools_v1",
    )
    assert all(value["passed"] for value in qualify_cpu(names, 2).values())
    assert all(value["passed"] for value in qualify_adversarial(names).values())


def test_model_gate_rejects_uniform_success_or_uniform_failure():
    for reward in (0.0, 1.0):
        rows = [_row(reward), _row(reward)]
        summary = summarize_model_environment(
            rows,
            {0: [reward, reward]},
            sigma_min=0.24,
        )
        assert summary["passed"] is False
        assert summary["grpo_eligible_groups"] == 0


def test_model_gate_requires_exact_replay_and_training_frontier():
    rows = [_row(0.0), _row(1.0)]
    assert summarize_model_environment(
        rows,
        {0: [0.0, 1.0]},
        sigma_min=0.24,
    )["passed"] is True
    rows[1]["exact_replay"] = False
    assert summarize_model_environment(
        rows,
        {0: [0.0, 1.0]},
        sigma_min=0.24,
    )["passed"] is False
