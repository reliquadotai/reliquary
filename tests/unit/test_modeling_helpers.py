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


def test_qwen35_text_config_uses_causal_lm_auto_class():
    # Teutonic-I with Qwen3.5 attention grafted: a text-only Qwen3.5 checkpoint.
    cfg = SimpleNamespace(model_type="qwen3_5_text", architectures=["Qwen3_5ForCausalLM"])

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


def test_a_text_only_load_takes_the_language_model_of_a_multimodal_checkpoint(monkeypatch):
    """The corpus audit proves text; a multimodal checkpoint whose vision config
    this transformers cannot read (Qwen3.8) must still load its language model."""
    import types

    import transformers

    from reliquary.shared import modeling

    text_config = object()
    composite = types.SimpleNamespace(text_config=text_config)
    calls = {}
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: composite))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        classmethod(lambda cls, source, **k: calls.setdefault("kw", (source, k))))
    modeling.load_text_only_model("/ckpt", torch_dtype="bf16")
    source, kwargs = calls["kw"]
    assert source == "/ckpt" and kwargs["config"] is text_config and kwargs["dtype"] == "bf16"


def test_a_text_only_load_of_a_text_checkpoint_is_the_usual_load(monkeypatch):
    import types

    import transformers

    from reliquary.shared import modeling

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: types.SimpleNamespace()))
    monkeypatch.setattr(modeling, "load_text_generation_model", lambda source, **k: ("usual", source))
    assert modeling.load_text_only_model("/ckpt") == ("usual", "/ckpt")


def test_gemma4_text_only_load_maps_language_model_weights(monkeypatch):
    """Gemma 4 keeps its text weights under model.language_model: without the mapping its causal LM would load
    nothing and keep random weights (every forced-draw sample refused, 2026-11-05). Its 512-wide global heads
    cannot run on FlashAttention 2, so the load takes sdpa."""
    import types

    import transformers

    from reliquary.shared import modeling

    text_config = types.SimpleNamespace(head_dim=256, global_head_dim=512, hidden_size=5376,
                                        num_attention_heads=32, final_logit_softcapping=30.0)
    composite = types.SimpleNamespace(model_type="gemma4", text_config=text_config)
    calls = {}
    head = types.SimpleNamespace()
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: composite))

    def load(cls, source, **k):
        calls["kw"] = k
        return types.SimpleNamespace(config=k["config"], lm_head=head)

    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", classmethod(load))
    modeling.load_text_only_model("/ckpt", torch_dtype="bf16", attn_implementation="flash_attention_2")
    kwargs = calls["kw"]
    assert kwargs["key_mapping"] == {r"^model\.language_model\.": "model."}
    assert kwargs["attn_implementation"] == "sdpa"
    assert getattr(head, modeling.LOGIT_SOFTCAP_ATTR) == 30.0


def test_a_qwen_text_only_load_keeps_its_weights_names_and_attention(monkeypatch):
    import types

    import transformers

    from reliquary.shared import modeling

    text_config = types.SimpleNamespace(head_dim=256, hidden_size=5120, num_attention_heads=24)
    composite = types.SimpleNamespace(model_type="qwen3_5", text_config=text_config)
    calls = {}
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained",
                        classmethod(lambda cls, *a, **k: composite))
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained",
                        classmethod(lambda cls, source, **k: calls.setdefault("kw", k)))
    modeling.load_text_only_model("/ckpt", attn_implementation="flash_attention_2")
    assert "key_mapping" not in calls["kw"] and calls["kw"]["attn_implementation"] == "flash_attention_2"


def test_a_softcapped_head_is_capped_by_the_verifier():
    import torch

    from reliquary.shared.modeling import LOGIT_SOFTCAP_ATTR
    from reliquary.validator.verifier import _LazyLogitRows

    head = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.tensor([[40.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, -50.0, 0]]))
    hidden = torch.ones(2, 4)
    plain = _LazyLogitRows(hidden, head, 3, fp32_head=False)[0:2]
    assert plain[0].tolist() == [40.0, 1.0, -50.0]
    setattr(head, LOGIT_SOFTCAP_ATTR, 30.0)
    capped = _LazyLogitRows(hidden, head, 3, fp32_head=False)[0:2]
    assert torch.allclose(capped[0], 30.0 * torch.tanh(torch.tensor([40.0, 1.0, -50.0]) / 30.0))


class _ChatTokenizer:
    """A tokenizer whose template closes an assistant turn with ``closer``; ids by a fixed table."""

    def __init__(self, eos_token_id, closer, special):
        self.eos_token_id, self._closer = eos_token_id, closer
        self.chat_template = "x"
        self.all_special_ids = list(special.values())
        self._special = special

    def apply_chat_template(self, messages, tokenize=False):
        return "".join(f"<{m['role']}>{m['content']}{self._closer}\n" for m in messages)

    def encode(self, text, add_special_tokens=False):
        for token, i in self._special.items():
            if text.startswith(token):
                return [i] + [9] * len(text[len(token):])
        return [9] * len(text)


def test_turn_end_token_prefers_a_declared_turn_terminator():
    from reliquary.shared.modeling import turn_end_token_id

    gemma = _ChatTokenizer(1, "<turn|>", {"<eos>": 1, "<turn|>": 106})
    assert turn_end_token_id(gemma, [1, 106, 50]) == 106
    # The model does not declare it a stop: the tokenizer's eos stands.
    assert turn_end_token_id(gemma, [1]) == 1
    assert turn_end_token_id(gemma, []) == 1
    # Qwen instruct: the turn closer is already the eos.
    qwen = _ChatTokenizer(151645, "<|im_end|>", {"<|im_end|>": 151645, "<|endoftext|>": 151643})
    assert turn_end_token_id(qwen, [151645, 151643]) == 151645
    # Qwen base: generation_config declares only <|endoftext|>, so the eos stays.
    base = _ChatTokenizer(151643, "<|im_end|>", {"<|im_end|>": 151645, "<|endoftext|>": 151643})
    assert turn_end_token_id(base, [151643]) == 151643
