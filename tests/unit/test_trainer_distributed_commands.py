import pytest
import pickle
import torch

from tests.unit.trainer_dist_helpers import run_ranks, tiny_qwen35


def _channel_roundtrip():
    from reliquary.trainer.distributed import CommandChannel, init_distributed, run_follower

    ctx = init_distributed(device_type="cpu")
    channel = CommandChannel(ctx)
    if ctx.is_main:
        channel.send("LOAD", {"path": "p"})
        channel.send("TRAIN", {"batches": [[1, 2]], "kwargs": {"window_index": 5}})
        channel.send("GATHER")
        channel.send("STOP")
        return "sent"
    seen = []
    run_follower(
        channel,
        load=lambda payload: seen.append(("LOAD", payload)) or "model",
        train=lambda model, payload: seen.append(("TRAIN", model, payload)),
        gather=lambda model: seen.append(("GATHER", model)),
    )
    return seen


def test_follower_stops_on_stop():
    out = run_ranks(_channel_roundtrip, 2)
    assert out[1] == [
        ("LOAD", {"path": "p"}),
        ("TRAIN", "model", {"batches": [[1, 2]], "kwargs": {"window_index": 5}}),
        ("GATHER", "model"),
    ]


def test_saved_checkpoint_reloads_bf16(tmp_path):
    """The gathered-state export matches today's save_pretrained publication:
    same config (dtype bfloat16), same files, same reloaded logits."""
    import json
    from transformers import AutoModelForCausalLM
    from reliquary.trainer.distributed import save_state_dict_hf

    model = tiny_qwen35().to(torch.bfloat16)
    model.save_pretrained(tmp_path / "today")
    state = {k: v.clone() for k, v in model.state_dict().items()}
    save_state_dict_hf(model, state, None, tmp_path / "fsdp2")
    today = json.loads((tmp_path / "today" / "config.json").read_text())
    ours = json.loads((tmp_path / "fsdp2" / "config.json").read_text())
    assert ours == today and ours["dtype"] == "bfloat16"
    assert sorted(p.name for p in (tmp_path / "fsdp2").iterdir()) == \
        sorted(p.name for p in (tmp_path / "today").iterdir())
    x = torch.randint(0, 300, (1, 32))
    want = AutoModelForCausalLM.from_pretrained(tmp_path / "today", dtype=torch.bfloat16)(x).logits
    got = AutoModelForCausalLM.from_pretrained(tmp_path / "fsdp2", dtype=torch.bfloat16)(x).logits
    assert torch.equal(got, want)




def test_a_failed_gather_is_fatal_not_retried(tmp_path, monkeypatch):
    """Rank 0 failing mid-gather must stop the trainer: a retried GATHER would
    pair its collectives with the followers' unfinished ones."""
    from reliquary.trainer import distributed
    from reliquary.trainer.distributed import TrainerStateUnsafe

    sent = []

    class _Channel:
        def send(self, command, payload=None):
            sent.append(command)

    def _boom(model, ctx):
        raise RuntimeError("host out of memory")

    monkeypatch.setattr(distributed, "gather_full_state_dict", _boom)
    save = distributed.sharded_save_fn(lambda: object(), _Channel(), object())
    with pytest.raises(TrainerStateUnsafe):
        save(None, None, tmp_path)
    assert sent == ["GATHER"]


def test_launch_refuses_several_processes_without_fsdp2():
    from reliquary.trainer.distributed import check_trainer_launch

    check_trainer_launch("single", {"WORLD_SIZE": "1"})
    check_trainer_launch("fsdp2", {"WORLD_SIZE": "4"})
    with pytest.raises(SystemExit):
        check_trainer_launch("single", {"WORLD_SIZE": "4"})
