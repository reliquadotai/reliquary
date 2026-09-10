"""Exercise CLI wiring with the real journal/codec, without model or network I/O."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from reliquary import constants
from reliquary.shared.training_payload import (
    TrainingPayloadProtocolMismatch,
    active_training_identity,
    encode_tombstone,
    encode_training_payload,
)
from reliquary.trainer import cli, journal, publisher, resume, train_runner, worker
from reliquary.shared import modeling
from reliquary.validator import telemetry, training
from tests.unit.test_training_payload_codec import _window_batches


class _JournalChecked(BaseException):
    pass


@pytest.mark.parametrize("protocol_version", [4, 5, 6])
@pytest.mark.parametrize("kind", ["payload", "tombstone"])
@pytest.mark.parametrize("foreign_run", [False, True])
def test_cli_consumes_its_codec_and_keeps_manifest_repo_guard(
    monkeypatch, tmp_path, protocol_version, kind, foreign_run,
):
    monkeypatch.setattr(constants, "PROTOCOL_VERSION", protocol_version)
    monkeypatch.setattr(journal, "FILL_CLOSED_ENABLED", False)
    expected = active_training_identity() if protocol_version >= 5 else {}
    if foreign_run:
        monkeypatch.setattr(constants, "TRAINING_RUN_ID", "foreign-run")
    if kind == "payload":
        data = encode_training_payload(
            _window_batches(), window_start=101, checkpoint_revision="rev",
            env_order=["openmathinstruct", "opencodeinstruct"],
            window_quarantine={"quarantined": False},
        )
        key = "reliquary/training/window-101.npz"
    else:
        data = encode_tombstone(window_start=101, failure_stage="test", failure_type="test")
        key = "reliquary/training/window-101.tombstone.json"
    if foreign_run:
        monkeypatch.setattr(constants, "TRAINING_RUN_ID", expected.get("training_run_id", "test"))

    monkeypatch.setenv("RELIQUARY_HF_REPO_ID", "owner/model")
    monkeypatch.setenv("RELIQUARY_TRAINER_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(cli, "_r2_client", lambda: None)
    monkeypatch.setattr(journal, "r2_fetch_fn", lambda *_: {key: data}.get)

    def resolve(_fetch, *, env, expected_identity):
        assert expected_identity == {**expected, "repo_id": "owner/model"}
        return None, 100, 0

    monkeypatch.setattr(resume, "resolve_resume_point", resolve)
    monkeypatch.setattr(modeling, "load_tokenizer", MagicMock())
    monkeypatch.setattr(modeling, "load_text_generation_model", MagicMock())
    monkeypatch.setattr(telemetry, "init", MagicMock())
    monkeypatch.setattr(training, "reset_training_state", MagicMock())
    monkeypatch.setattr(train_runner, "TrainRunner", MagicMock())
    monkeypatch.setattr(publisher, "TrainerPublisher", MagicMock())
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "HfApi", lambda: SimpleNamespace(
        model_info=lambda _: SimpleNamespace(sha=None)))

    def check_journal(**kwargs):
        actual_kind, decoded = kwargs["journal"].next_entry(100, stride=1)
        assert actual_kind == kind
        assert (decoded.window_start if kind == "payload" else decoded["window_start"]) == 101
        raise _JournalChecked

    monkeypatch.setattr(worker, "TrainerWorker", check_journal)
    if foreign_run and protocol_version >= 5:
        with pytest.raises(TrainingPayloadProtocolMismatch, match="training_run_id"):
            cli.run_train_worker(shadow=True)
    else:
        with pytest.raises(_JournalChecked):
            cli.run_train_worker(shadow=True)
