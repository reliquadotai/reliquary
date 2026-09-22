"""The store a streamed replica reads from must hold exactly what the checkpoint held.

Fusing is where a layer's experts get stacked, and a store written once is read on every
traversal from then on. A store that differs from its checkpoint by a single tensor would reject
honest miners, so what is asserted here is equality with the checkpoint, not merely that files
appeared.
"""

import json

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.shared.forward import forward_single_layer
from reliquary.shared.fused_layers import (
    STAMP_FILE,
    build_fused_store,
    install_fused_store,
    is_fused_store,
    layer_file,
    prune_fused_stores,
)

MODEL_TYPES = ("qwen3", "qwen3_moe")


def _tiny(model_type: str, tmp_path):
    common = dict(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
        eos_token_id=2, tie_word_embeddings=True,
    )
    if model_type == "qwen3_moe":
        common |= dict(
            num_experts=4, num_experts_per_tok=2, moe_intermediate_size=32,
            decoder_sparse_step=1, norm_topk_prob=True, mlp_only_layers=[],
        )
    config = AutoConfig.for_model(model_type, **common)
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config).to(torch.float32).eval()
    path = tmp_path / model_type
    model.save_pretrained(str(path))
    return model, path


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_a_replica_on_the_store_verifies_like_one_on_the_checkpoint(model_type, tmp_path):
    from reliquary.shared.streaming_forward import StreamedReplica

    resident, path = _tiny(model_type, tmp_path)
    store = build_fused_store(path, tmp_path / "store" / "rev")
    tokens = torch.randint(0, 256, (2, 24), generator=torch.Generator().manual_seed(1))

    hidden, logits = forward_single_layer(resident, tokens, None, -1)
    replica = StreamedReplica.from_checkpoint(str(store), device="cpu")
    streamed_hidden, streamed_logits = forward_single_layer(replica, tokens, None, -1)

    assert torch.equal(streamed_hidden, hidden)
    assert torch.equal(streamed_logits, logits)


def test_a_store_is_a_file_per_layer_and_is_read_without_being_told(tmp_path):
    """The replica recognises a store on sight: nothing has to carry the layout around."""
    from reliquary.shared.layer_source import CheckpointLayers

    _, path = _tiny("qwen3_moe", tmp_path)
    store = build_fused_store(path, tmp_path / "store" / "rev", revision="abc")

    assert is_fused_store(store) and not is_fused_store(path)
    assert json.loads((store / STAMP_FILE).read_text()) == {"layers": 3, "revision": "abc"}
    assert all((store / layer_file(i)).exists() for i in range(3))
    state = CheckpointLayers(store, fused_dir=store).state(0)
    assert state["mlp.experts.gate_up_proj"].shape[0] == 4, "the experts arrive already stacked"


def test_the_store_survives_the_staged_download_it_was_built_from(tmp_path):
    """Intake deletes the staged copy at swap; a streamed replica reads for the whole revision."""
    import shutil

    from reliquary.shared.streaming_forward import StreamedReplica

    resident, path = _tiny("qwen3_moe", tmp_path)
    store = install_fused_store(path, tmp_path / "root", "revision-one")
    shutil.rmtree(path)

    tokens = torch.randint(0, 256, (1, 16), generator=torch.Generator().manual_seed(3))
    hidden, _ = forward_single_layer(resident, tokens, None, -1)
    replica = StreamedReplica.from_checkpoint(str(store), device="cpu")
    streamed, _ = forward_single_layer(replica, tokens, None, -1)
    assert torch.equal(streamed, hidden)


def test_installing_twice_builds_once(tmp_path, monkeypatch):
    import reliquary.shared.fused_layers as fused_layers

    _, path = _tiny("qwen3", tmp_path)
    builds = []
    original = fused_layers.build_fused_store
    monkeypatch.setattr(
        fused_layers, "build_fused_store",
        lambda *a, **k: (builds.append(a), original(*a, **k))[1],
    )

    first = fused_layers.install_fused_store(path, tmp_path / "root", "rev")
    second = fused_layers.install_fused_store(path, tmp_path / "root", "rev")

    assert first == second and len(builds) == 1


def test_a_half_written_store_is_never_left_behind(tmp_path, monkeypatch):
    """A crash mid-build must not leave a directory a later traversal reads as complete."""
    import reliquary.shared.fused_layers as fused_layers

    _, path = _tiny("qwen3", tmp_path)
    root = tmp_path / "root"
    monkeypatch.setattr(fused_layers.shutil, "copy2", lambda *a, **k: 1 / 0)

    with pytest.raises(ZeroDivisionError):
        fused_layers.install_fused_store(path, root, "rev")

    assert not is_fused_store(root / "rev")
    assert list(root.iterdir()) == [], "nothing partial is left for the next run to find"


def test_pruning_spares_the_generation_a_slot_may_still_be_reading(tmp_path):
    _, path = _tiny("qwen3", tmp_path)
    root = tmp_path / "root"
    for revision in ("one", "two", "three"):
        build_fused_store(path, root / revision, revision=revision)

    prune_fused_stores(root, keep="three")

    assert (root / "three").is_dir(), "the one being proved against"
    assert (root / "two").is_dir(), "a slot that has not rotated yet still reads it"
    assert not (root / "one").exists()
