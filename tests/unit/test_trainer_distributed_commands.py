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


def test_decoded_batches_pickle():
    """A batch decoded from a real journal payload survives the broadcast."""
    from reliquary.shared.training_payload import decode_training_payload
    from tests.unit.test_trainer_journal import _payload_bytes

    batches = decode_training_payload(_payload_bytes(3)).batches()
    copy_ = pickle.loads(pickle.dumps(batches))
    tokens = lambda b: [[r.commit["tokens"] for g in groups for r in g.rollouts] for groups in b.values()]
    assert tokens(copy_) == tokens(batches)
