"""A replica that keeps one decoder layer on the device must verify like the resident model.

The equality asserted here is the oracle this design depends on: it exists while the model still
fits on one card, and it is gone once it does not. Every numerical claim about the streaming path
rests on these tests.
"""

import pytest
import torch
from transformers import AutoConfig, AutoModelForCausalLM

from reliquary.shared.forward import forward_single_layer


def _tiny(model_type: str, tmp_path):
    """A model small enough for CI, saved as a checkpoint the replica can stream from."""
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


MODEL_TYPES = ("qwen3", "qwen3_moe")


@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("batch", (1, 3))
def test_streamed_replica_matches_the_resident_model(model_type, batch, tmp_path):
    from reliquary.shared.streaming_forward import StreamedReplica

    resident, path = _tiny(model_type, tmp_path)
    replica = StreamedReplica.from_checkpoint(str(path), device="cpu")
    tokens = torch.randint(0, 256, (batch, 24), generator=torch.Generator().manual_seed(1))

    hidden, logits = forward_single_layer(resident, tokens, None, -1)
    streamed_hidden, streamed_logits = forward_single_layer(replica, tokens, None, -1)

    assert torch.equal(streamed_hidden, hidden)
    assert torch.equal(streamed_logits, logits)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_prefetching_changes_nothing(model_type, tmp_path):
    """Reading the next layer while the device works on this one is where races hide."""
    from reliquary.shared.streaming_forward import StreamedReplica

    resident, path = _tiny(model_type, tmp_path)
    tokens = torch.randint(0, 256, (2, 16), generator=torch.Generator().manual_seed(2))
    hidden, _ = forward_single_layer(resident, tokens, None, -1)

    for prefetch in (False, True):
        replica = StreamedReplica.from_checkpoint(str(path), device="cpu", prefetch=prefetch)
        streamed, _ = forward_single_layer(replica, tokens, None, -1)
        assert torch.equal(streamed, hidden), prefetch


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_the_replica_holds_one_layer_at_a_time(model_type, tmp_path):
    """The whole point: what stays resident is the fixed parts plus a single layer."""
    from reliquary.shared.streaming_forward import StreamedReplica

    resident, path = _tiny(model_type, tmp_path)
    replica = StreamedReplica.from_checkpoint(str(path), device="cpu")
    weight = lambda module: sum(p.numel() * p.element_size() for p in module.parameters())
    whole = weight(resident)
    one_layer = weight(resident.model.layers[0])
    fixed = weight(resident.model.embed_tokens) + weight(resident.model.norm) + weight(resident.lm_head)
    assert replica.resident_bytes == fixed + one_layer, "it holds the fixed parts and one layer"
    assert replica.resident_bytes < whole


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_the_replica_answers_what_verification_asks_of_a_model(model_type, tmp_path):
    """verify_commitment_proofs reads these off the model; a replica that lacks one is useless."""
    from reliquary.shared.hf_compat import resolve_hidden_size
    from reliquary.shared.streaming_forward import StreamedReplica

    resident, path = _tiny(model_type, tmp_path)
    replica = StreamedReplica.from_checkpoint(str(path), device="cpu")

    assert resolve_hidden_size(replica) == resolve_hidden_size(resident)
    assert next(replica.parameters()).device == torch.device("cpu")
    assert replica.lm_head is not None
    assert replica.config.vocab_size == resident.config.vocab_size


def test_a_padded_batch_is_refused_until_it_is_supported(tmp_path):
    """The proof path passes no mask today; a mask that arrives silently ignored would be a bug."""
    from reliquary.shared.streaming_forward import StreamedReplica

    _, path = _tiny("qwen3", tmp_path)
    replica = StreamedReplica.from_checkpoint(str(path), device="cpu")
    tokens = torch.randint(0, 256, (2, 8))
    mask = torch.ones_like(tokens)
    mask[0, :3] = 0
    with pytest.raises(NotImplementedError, match="attention mask"):
        forward_single_layer(replica, tokens, mask, -1)
