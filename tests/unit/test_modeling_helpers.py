from types import SimpleNamespace
from unittest.mock import MagicMock

from reliquary.shared.hf_compat import resolve_max_context_length, resolve_vocab_size
from reliquary.shared.modeling import (
    MODEL_SNAPSHOT_ALLOW_PATTERNS,
    auto_model_class_for_config,
    first_eos_index,
    load_text_generation_model,
    resolve_eos_token_ids,
)


def test_qwen35_eos_union_includes_model_and_tokenizer_ids():
    model = SimpleNamespace(
        generation_config=SimpleNamespace(eos_token_id=248044),
        config=SimpleNamespace(
            eos_token_id=None,
            text_config=SimpleNamespace(eos_token_id=248044),
        ),
    )
    tokenizer = SimpleNamespace(eos_token_id=248046)

    assert resolve_eos_token_ids(model, tokenizer) == {248044, 248046}


def test_eos_resolver_ignores_magicmocks():
    assert resolve_eos_token_ids(MagicMock(), MagicMock()) == set()


def test_first_eos_index_uses_full_set():
    assert first_eos_index([1, 2, 248046, 248044], {248044, 248046}) == 2
    assert first_eos_index([1, 2, 3], {248044, 248046}) is None


def test_nested_text_config_dimensions_are_resolved():
    cfg = SimpleNamespace(
        vocab_size=None,
        max_position_embeddings=None,
        text_config=SimpleNamespace(
            vocab_size=248320,
            max_position_embeddings=262144,
        ),
    )

    assert resolve_vocab_size(cfg) == 248320
    assert resolve_max_context_length(cfg) == 262144


def test_qwen35_conditional_config_uses_image_text_auto_class():
    cfg = SimpleNamespace(
        model_type="qwen3_5",
        architectures=["Qwen3_5ForConditionalGeneration"],
    )

    assert auto_model_class_for_config(cfg).__name__ == "AutoModelForImageTextToText"


def test_legacy_config_uses_causal_lm_auto_class():
    cfg = SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLM"])

    assert auto_model_class_for_config(cfg).__name__ == "AutoModelForCausalLM"


def test_checkpoint_download_patterns_include_qwen35_shards():
    assert "model*.safetensors" in MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "model.safetensors.index.json" in MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "chat_template.jinja" in MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "special_tokens_map.json" in MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "processor_config.json" in MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "vocab.json" in MODEL_SNAPSHOT_ALLOW_PATTERNS
    assert "merges.txt" in MODEL_SNAPSHOT_ALLOW_PATTERNS


def test_has_think_close_detects_atomic_id():
    from reliquary.shared.modeling import has_think_close
    assert has_think_close([1, 2, 248069, 3], {248069}) is True
    assert has_think_close([1, 2, 3], {248069}) is False
    assert has_think_close([], {248069}) is False


def test_think_close_and_force_ids_from_tokenizer():
    from reliquary.shared.modeling import (
        force_close_token_ids,
        think_close_token_ids,
    )

    class _Tok:
        def convert_tokens_to_ids(self, t):
            return 248069 if t == "</think>" else None

        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [7, 8, 9]  # stand-in for "\n\nFinal Answer: \boxed{"

    assert think_close_token_ids(_Tok()) == [248069]
    assert force_close_token_ids(_Tok()) == [248069, 7, 8, 9]


def test_model_loader_normalizes_torch_dtype_for_transformers_5(monkeypatch):
    import reliquary.shared.modeling as modeling
    from transformers import AutoConfig

    seen: dict[str, dict] = {}

    def fake_config_from_pretrained(source, **kwargs):
        seen["config_kwargs"] = kwargs
        return SimpleNamespace(model_type="qwen3", architectures=["Qwen3ForCausalLM"])

    class FakeAutoModel:
        @staticmethod
        def from_pretrained(source, **kwargs):
            seen["model_kwargs"] = kwargs
            return "loaded"

    monkeypatch.setattr(AutoConfig, "from_pretrained", fake_config_from_pretrained)
    monkeypatch.setattr(modeling, "auto_model_class_for_config", lambda config: FakeAutoModel)

    assert load_text_generation_model("repo/model", torch_dtype="bf16", cache_dir="/tmp/hf") == "loaded"
    assert seen["config_kwargs"] == {"cache_dir": "/tmp/hf"}
    assert seen["model_kwargs"]["dtype"] == "bf16"
    assert "torch_dtype" not in seen["model_kwargs"]
