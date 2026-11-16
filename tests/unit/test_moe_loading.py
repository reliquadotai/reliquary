"""MoE checkpoints load whole in the verifier: a tiny random model saved in its released layout, loaded back through
``shared.modeling``'s text-only path (the audit executor's), gives the logits the saved model gives."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

_TEXT = dict(vocab_size=128, hidden_size=32, intermediate_size=64, num_hidden_layers=2, num_attention_heads=2,
             num_key_value_heads=1, head_dim=16, num_experts=8, num_experts_per_tok=2, moe_intermediate_size=16,
             max_position_embeddings=64, tie_word_embeddings=False)


def _qwen3_moe():
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    return Qwen3MoeForCausalLM(Qwen3MoeConfig(**_TEXT, decoder_sparse_step=1, norm_topk_prob=True))


def _qwen35_moe():
    from transformers import Qwen3_5MoeConfig, Qwen3_5MoeForConditionalGeneration

    text = dict(_TEXT, shared_expert_intermediate_size=16, layer_types=["linear_attention", "full_attention"],
                linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8, linear_value_head_dim=8,
                linear_conv_kernel_dim=4)
    vision = dict(depth=1, hidden_size=16, intermediate_size=32, num_heads=2, out_hidden_size=32, patch_size=2,
                  temporal_patch_size=1, spatial_merge_size=1)
    return Qwen3_5MoeForConditionalGeneration(Qwen3_5MoeConfig(text_config=text, vision_config=vision))


@pytest.mark.parametrize("build,architecture", [(_qwen3_moe, "Qwen3MoeForCausalLM"),
                                                (_qwen35_moe, "Qwen3_5MoeForConditionalGeneration")])
def test_a_moe_checkpoint_loads_text_only_with_every_weight(tmp_path, build, architecture):
    from reliquary.constants import SUPPORTED_ARCHITECTURES
    from reliquary.shared.modeling import load_text_only_model

    torch.manual_seed(0)
    saved = build().eval()
    saved.save_pretrained(tmp_path)
    assert saved.config.architectures == [architecture]
    assert architecture in SUPPORTED_ARCHITECTURES

    loaded = load_text_only_model(str(tmp_path), torch_dtype=torch.float32).eval()
    ids = torch.tensor([[1, 5, 9, 17, 33, 65, 3, 7]])
    with torch.no_grad():
        want = saved(input_ids=ids).logits
        got = loaded(input_ids=ids).logits
    # Every expert, router and (Qwen3.5) shared expert came from the checkpoint, none from a random init.
    assert torch.equal(got, want)
