"""An explicit metadata bridge preserves training progress, not V5 wire identity."""

import hashlib
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import prepare_v5_fill_checkpoint as prepare
from reliquary.protocol.profiles import resolve_protocol_profile


def _profile(name):
    p = resolve_protocol_profile(name)
    return {
        "schema_version": 2, "profile_id": p.profile_id,
        "protocol_version": p.protocol_version,
        "base_model_id": p.model_id, "base_model_revision": p.model_revision,
        "generation_contract_sha256": hashlib.sha256(json.dumps(
            p.to_generation_contract(), sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest(),
        "training_run_id": "continuation-test",
    }


@pytest.fixture
def source(monkeypatch):
    import reliquary.infrastructure.training_payload_queue as queue
    monkeypatch.setattr(prepare, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(queue, "FILL_CLOSED_EMISSIONS_PER_WINDOW", 16)
    monkeypatch.setattr(prepare, "active_checkpoint_profile", lambda: _profile(
        "qwen3-4b-base-dapo-fill-closed-v6",
    ))
    return {**_profile("qwen3-4b-base-dapo-reasoning-v5"),
            "lr_schedule_step": 1234, "trained_window_cursor": 50}


def _plan(source, **overrides):
    args = dict(repo_id="owner/model", revision="a" * 40,
                checkpoint_n=42, last_archived_window=50)
    args.update(overrides)
    return prepare.prepare_transition(source, **args)


def test_bridge_inherits_weights_and_preserves_progress(source):
    plan = _plan(source)
    assert plan["parent_commit"] == "a" * 40
    assert plan["commit_message"] == "checkpoint 43 (v5-to-fill-v6)"
    assert set(plan["files"]) == {prepare.CHECKPOINT_PROFILE_NAME, prepare.TRANSITION_NAME}
    target = plan["files"][prepare.CHECKPOINT_PROFILE_NAME]
    assert target["training_run_id"] == source["training_run_id"]
    assert target["lr_schedule_step"] == 1234
    assert target["trained_window_cursor"] == 51 * 16 - 1
    assert target["journal_key_space"] == "fill_closed"
    assert plan["files"][prepare.TRANSITION_NAME]["source_profile"] == source


@pytest.mark.parametrize("key,value", [
    ("profile_id", "qwen3-4b-base-dapo-v4"), ("protocol_version", 4),
    ("generation_contract_sha256", "0" * 64), ("base_model_revision", "0" * 40),
    ("training_run_id", "another-run"), ("training_run_id", None),
    ("lr_schedule_step", None), ("lr_schedule_step", True),
    ("lr_schedule_step", -1), ("trained_window_cursor", None),
    ("trained_window_cursor", 50.0), ("journal_key_space", "fill_closed"),
])
def test_bridge_rejects_unproven_lineage_or_progress(source, key, value):
    source[key] = value
    with pytest.raises(ValueError):
        _plan(source)


@pytest.mark.parametrize("overrides", [
    {"revision": "main"}, {"checkpoint_n": True},
    {"last_archived_window": 49}, {"last_archived_window": 51},
])
def test_bridge_rejects_mutable_source_or_undrained_boundary(source, overrides):
    with pytest.raises(ValueError):
        _plan(source, **overrides)


def test_preparation_requires_explicit_fill_activation(source, monkeypatch):
    monkeypatch.setattr(prepare, "FILL_CLOSED_ENABLED", False)
    with pytest.raises(ValueError, match="exact fill-closed"):
        _plan(source)


def test_repository_head_and_checkpoint_number_cannot_move_or_rebind():
    head = SimpleNamespace(commit_id="a" * 40, title="checkpoint 42 (cadence)")
    old = SimpleNamespace(commit_id="b" * 40, title="checkpoint 41")
    assert prepare.source_checkpoint_number([head, old], "a" * 40) == 42
    for commits in ([old, head], [head, head], [head, SimpleNamespace(
        commit_id="c" * 40, title="checkpoint 43",
    )]):
        with pytest.raises(ValueError):
            prepare.source_checkpoint_number(commits, "a" * 40)


@pytest.mark.parametrize("case", ["simple", "sharded", "missing_model", "missing_shard", "moved"])
def test_cli_reads_immutable_metadata_and_never_publishes(source, tmp_path, monkeypatch, case):
    import huggingface_hub

    revision = "a" * 40
    profile_path = tmp_path / prepare.CHECKPOINT_PROFILE_NAME
    profile_path.write_text(json.dumps(source))
    index_name = "model.safetensors.index.json"
    (tmp_path / index_name).write_text(json.dumps({
        "weight_map": {"layer.weight": "model-00001-of-00001.safetensors"},
    }))
    names = ["config.json", "tokenizer_config.json", "tokenizer.json"]
    names += {
        "simple": ["model.safetensors"], "moved": ["model.safetensors"],
        "sharded": [index_name, "model-00001-of-00001.safetensors"],
        "missing_model": ["optimizer.safetensors"], "missing_shard": [index_name],
    }[case]
    heads = []
    downloads = []

    def commits(repo):
        assert repo == "owner/model"
        heads.append(repo)
        oid = "b" * 40 if case == "moved" and len(heads) == 2 else revision
        return [SimpleNamespace(commit_id=oid, title="checkpoint 42 (cadence)")]

    def info(repo, *, revision):
        assert repo == "owner/model" and revision == "a" * 40
        return SimpleNamespace(sha=revision, siblings=[
            SimpleNamespace(rfilename=name) for name in names
        ])

    def download(repo, name, *, revision):
        assert repo == "owner/model" and revision == "a" * 40
        assert name in {prepare.CHECKPOINT_PROFILE_NAME, index_name}
        downloads.append(name)
        return str(tmp_path / name)

    # Deliberately expose only reads: any upload/create_commit call fails.
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        list_repo_commits=commits, model_info=info,
    ))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", download)
    output = tmp_path / "prepared"
    monkeypatch.setattr(sys, "argv", ["prepare", "--repo-id", "owner/model",
        "--source-revision", revision, "--last-archived-window", "50",
        "--output-dir", str(output)])
    if case in {"missing_model", "missing_shard", "moved"}:
        with pytest.raises(ValueError):
            prepare.main()
        assert not output.exists()
    else:
        prepare.main()
        assert len(heads) == 2
        assert {p.name for p in output.iterdir()} == {
            prepare.CHECKPOINT_PROFILE_NAME, prepare.TRANSITION_NAME, "commit-plan.json",
        }
        plan = json.loads((output / "commit-plan.json").read_text())
        assert plan["parent_commit"] == revision
        assert plan["status"] == "prepared_only_no_remote_write"
        for name, digest in plan["add_files"].items():
            assert hashlib.sha256((output / name).read_bytes()).hexdigest() == digest
    assert downloads[0] == prepare.CHECKPOINT_PROFILE_NAME


def test_real_v6_resume_consumes_batch_zero_once_and_rejects_old_payloads(source, tmp_path):
    source_path = tmp_path / "source.json"
    source_path.write_text(json.dumps(source))
    code = r'''
import json, sys
from pathlib import Path
from scripts.prepare_v5_fill_checkpoint import prepare_transition
from reliquary.validator.checkpoint_profile import (
    CHECKPOINT_PROFILE_NAME, validate_checkpoint_profile, CheckpointProfileMismatch,
)
from reliquary.trainer.resume import resolve_resume_point
from reliquary.trainer.journal import WindowJournal, migrate_journal_cursor
from reliquary.trainer.worker import TrainerWorker
from reliquary.shared.training_payload import (
    active_training_identity, encode_training_payload, encode_tombstone,
    TrainingPayloadProtocolMismatch,
)
from reliquary.infrastructure.training_payload_queue import payload_key, tombstone_key
from tests.unit.test_training_payload_codec import _window_batches
source = json.loads(Path(sys.argv[1]).read_text())
plan = prepare_transition(source, repo_id='owner/model', revision='a'*40,
                          checkpoint_n=42, last_archived_window=50)
profile = plan['files'][CHECKPOINT_PROFILE_NAME]
directory = Path(sys.argv[1]).parent
(directory / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(profile))
assert validate_checkpoint_profile(directory, required=True) == profile
(directory / CHECKPOINT_PROFILE_NAME).write_text(json.dumps(source))
try:
    validate_checkpoint_profile(directory, required=True)
except CheckpointProfileMismatch:
    pass
else:
    raise AssertionError('ordinary V6 resume accepted a V5 checkpoint')
old_identity = dict(protocol_profile_id=source['profile_id'], protocol_version=5,
    training_run_id=source['training_run_id'],
    generation_contract_sha256=source['generation_contract_sha256'])
identity = active_training_identity()
revision, cursor, number = resolve_resume_point(
    lambda _: json.dumps(old_identity).encode(),
    env=dict(RELIQUARY_TRAINER_BOOTSTRAP_REVISION='b'*40,
             RELIQUARY_TRAINER_BOOTSTRAP_CURSOR=str(profile['trained_window_cursor']),
             RELIQUARY_TRAINER_CHECKPOINT_N='43'), expected_identity=identity)
assert (revision, number) == ('b'*40, 43)
assert migrate_journal_cursor(cursor, profile['journal_key_space']) == (cursor, 'fill_closed')
first_key = 51*16
assert cursor == first_key-1
store = {payload_key(first_key): encode_training_payload(_window_batches(),
    window_start=51, checkpoint_revision=revision,
    env_order=['openmathinstruct','opencodeinstruct'], window_quarantine={})}
received = []
worker = TrainerWorker(journal=WindowJournal(store.get, expected_identity=identity),
    train_fn=lambda payload: received.append(payload.window_start) or True,
    publish_fn=lambda reason: (_ for _ in ()).throw(AssertionError('unexpected publish')),
    head_revision_fn=lambda: revision, cursor=cursor, stride=1,
    publish_every=16, last_published_revision=revision)
assert worker.run_once() == 'trained'
assert received == [51] and worker.cursor == first_key
assert worker.run_once() == 'waited' and received == [51]
old = json.loads(encode_tombstone(window_start=51, failure_stage='test', failure_type='test'))
old.update(old_identity)
store[tombstone_key(first_key+1)] = json.dumps(old).encode()
try:
    worker.run_once()
except TrainingPayloadProtocolMismatch:
    pass
else:
    raise AssertionError('V6 consumed an old-protocol entry')
assert worker.cursor == first_key
'''
    result = subprocess.run([sys.executable, "-c", code, str(source_path)],
                            env={**os.environ,
                                 "RELIQUARY_PROTOCOL_PROFILE": "qwen3-4b-base-dapo-fill-closed-v6",
                                 "RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED": "1",
                                 "RELIQUARY_TRAINING_RUN_ID": "continuation-test"},
                            text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
