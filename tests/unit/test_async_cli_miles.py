"""``RELIQUARY_TRAINER_BACKEND=miles``: settings, gateway launch and the async trainer's wiring."""
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.trainer import miles_gateway
from reliquary.trainer.distributed import trainer_backend
from reliquary.trainer.miles_gateway import GatewayProcess, gateway_command, miles_settings

ENV = {"RELIQUARY_MILES_LAUNCHER": "/opt/lab/serve_tinker_full.py", "RELIQUARY_MILES_MODEL_TYPE": "qwen3.5-9B",
       "RELIQUARY_MILES_OUTPUT_DIR": "/data/miles"}


def test_backend_accepts_miles():
    assert trainer_backend({"RELIQUARY_TRAINER_BACKEND": "miles"}) == "miles"


@pytest.mark.parametrize("missing", sorted(ENV))
def test_settings_require_launcher_model_type_and_output(missing):
    env = {k: v for k, v in ENV.items() if k != missing}
    with pytest.raises(ValueError, match=missing):
        miles_settings(env)


def test_settings_defaults_and_command():
    settings = miles_settings(ENV)
    assert (settings["tp"], settings["gpus"], settings["port"]) == (2, 8, 10613)
    command = gateway_command("/ckpt/model", settings, base_model="org/policy", grad_norm_skip=100.0)
    assert command[:3] == [sys.executable, "/opt/lab/serve_tinker_full.py", "serve"]
    joined = " ".join(command)
    for part in ("--mode full", "--no-inference", "--hf-checkpoint /ckpt/model", "--model-type qwen3.5-9B",
                 "--tp 2", "--actor-num-gpus 8", "--num-gpus-per-node 8", "--tinker-port 10613",
                 "--output-dir /data/miles"):
        assert part in joined
    extra = command[command.index("--extra-args") + 1]
    assert "--tinker-base-model org/policy" in extra and "--tinker-grad-norm-skip 100.0" in extra


def test_gateway_waits_for_readiness_and_stops(tmp_path):
    script = tmp_path / "fake_gateway.py"
    script.write_text("import time\nprint('booting', flush=True)\ntime.sleep(0.3)\n"
                      "print('tinker gateway serving on 0.0.0.0:1', flush=True)\ntime.sleep(60)\n")
    stops = []
    gateway = GatewayProcess([sys.executable, str(script)], log_path=tmp_path / "gw.log", env=dict(os.environ),
                             cleanup=lambda: stops.append(1))
    gateway.wait_ready(timeout_s=20, poll_s=0.1)
    gateway.stop()
    assert gateway.process.poll() is not None and stops == [1]


def test_gateway_that_dies_raises(tmp_path):
    script = tmp_path / "dies.py"
    script.write_text("raise SystemExit(3)\n")
    gateway = GatewayProcess([sys.executable, str(script)], log_path=tmp_path / "gw.log", env=dict(os.environ),
                             cleanup=lambda: None)
    with pytest.raises(RuntimeError, match="exited"):
        gateway.wait_ready(timeout_s=20, poll_s=0.1)


def test_miles_backend_never_loads_a_torch_model(tmp_path, monkeypatch):
    """The wiring: gateway launched with the resumed weights, trainer built on the gateway, gateway stopped."""
    pytest.importorskip("tinker")
    from reliquary.shared import modeling
    from reliquary.trainer import async_cli

    events = []
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)   # restored after the test
    monkeypatch.setenv("RELIQUARY_TRAINER_BACKEND", "miles")
    for k, v in ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("RELIQUARY_TRAINER_STATE_DIR", str(tmp_path))
    monkeypatch.setattr(modeling, "load_text_generation_model", lambda *a, **k: pytest.fail("loaded a torch model"))
    monkeypatch.setattr(async_cli, "async_trainer_settings", lambda: {
        **{k: None for k in ("hf_prefix", "hf_every", "hf_milestone_every", "base_model")},
        "url": "u", "token": "t", "validator_hotkey": "h", "state_dir": tmp_path, "batch_groups": 4,
        "steps_per_checkpoint": 1, "anchor_interval": 50, "hf_keep_last": 3, "audits": True,
        "audits_per_step": 8, "audit_budget_s": 10.0, "prefetch": False, "report_workers": 1, "heartbeat_s": 60})
    link = SimpleNamespace(config=lambda: {"task_id": 1, "run": "r", "targets": {}}, requeue=lambda: 0,
                           latest_checkpoint=3)
    monkeypatch.setattr("reliquary.trainer.rl_link.TrainerLink", lambda url, token: link)
    monkeypatch.setattr("reliquary.validator.telemetry.init", lambda *a, **k: None)
    monkeypatch.setattr("reliquary.validator.telemetry.finish", lambda: None)
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text("{}")
    monkeypatch.setattr(async_cli, "resume_weights", lambda *a, **k: ({}, 3, model_dir))
    monkeypatch.setattr("reliquary.miner.rl_async.write_hf_dir", lambda state, src, dst: model_dir)
    monkeypatch.setattr(async_cli, "_eos_ids", lambda path, kw: (9,))
    monkeypatch.setattr("reliquary.validator.verifier.draw_settings", lambda: SimpleNamespace(
        draw="gumbel_max", t=1.0, top_k=0, top_p=1.0, boundary_epsilon=0.05))
    monkeypatch.setattr("reliquary.trainer.miles_engine.expected_weight_keys", lambda path: {"w"})
    monkeypatch.setattr("reliquary.infrastructure.storage.public_checkpoint_client", lambda: None)
    monkeypatch.setattr("reliquary.infrastructure.storage.public_checkpoint_bucket", lambda: "b")
    monkeypatch.setattr("reliquary.trainer.ckpt_publisher.DeltaPublisher", lambda **kw: SimpleNamespace())

    class Gateway:
        base_url = "http://127.0.0.1:10613"

        def stop(self):
            events.append("stop")

    def launch(model_path, settings, **kw):
        events.append(("launch", str(model_path), os.environ.get("CUDA_VISIBLE_DEVICES")))
        return Gateway()

    monkeypatch.setattr(miles_gateway, "launch_gateway", launch)
    monkeypatch.setattr(miles_gateway, "create_full_training_client", lambda *a: SimpleNamespace(name="client"))
    built = {}

    class Trainer:
        version = 3

        def __init__(self, model, **kw):
            built.update(kw, model=model)

        def close(self):
            events.append("close")

    monkeypatch.setattr("reliquary.trainer.async_trainer.AsyncTrainer", Trainer)
    monkeypatch.setattr(async_cli, "run_forever", lambda trainer: events.append("run"))
    monkeypatch.setattr("reliquary.trainer.rl_link.TrainerHeartbeat",
                        lambda link, s: SimpleNamespace(start=lambda: SimpleNamespace(stop=lambda: None)))
    async_cli.run_async_train_worker()
    assert events[0] == ("launch", str(model_dir), None) and events[-1] == "stop"
    assert "run" in events and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert type(built["train_step_fn"]).__name__ == "MilesTrainStep" and built["train_step_fn"].hint == 3
    assert type(built["state_fn"]).__name__ == "MilesState" and built["model"] is None
    assert built["audits"] is not None and built["global_step_hint"] == 3
