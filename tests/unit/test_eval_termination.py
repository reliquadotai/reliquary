"""Owned inference workers stop even when heartbeat cleanup is interrupted."""

from types import SimpleNamespace

import pytest

from reliquary.eval.model_process import ModelProcess
from reliquary.eval.platform_client import LeaseLost
from reliquary.eval.runner import EvalRunner, _Heartbeat


class IdleModel:
    def load(self):
        return {}


@pytest.mark.parametrize("failure", [
    RuntimeError("heartbeat cleanup failed"),
    LeaseLost("evaluation executor stopping"),
])
def test_interrupted_heartbeat_cleanup_still_joins_owned_model(tmp_path, monkeypatch, failure):
    model = ModelProcess(IdleModel, timeout_seconds=10)
    try:
        model.load()
        assert model._process.is_alive()
        monkeypatch.setattr(_Heartbeat, "start", lambda self: None)

        def interrupted(self):
            raise failure

        monkeypatch.setattr(_Heartbeat, "stop", interrupted)
        monkeypatch.setattr(EvalRunner, "_run", lambda *args: {"rows": 0})
        client = SimpleNamespace(claim=lambda: {"task_id": "fixture"})
        with pytest.raises(type(failure)) as raised:
            EvalRunner(client, model, work_dir=tmp_path).run()
        assert raised.value is failure
        assert not model._process.is_alive()
        assert model._process.exitcode is not None
    finally:
        model.close()
