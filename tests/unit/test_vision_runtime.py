"""Vision worker policy math, visual binding and response-only supervision."""

from types import SimpleNamespace

import pytest
import torch

from reliquary.vision.runtime import (
    _advantages,
    _assert_language_adapter,
    _optimizer_step,
    encode_example,
    frame_messages,
    generate_group,
    response_labels,
    response_logprobs,
)


def test_advantages_match_population_normalization_and_degenerate_groups():
    assert _advantages([0, 1, 2]) == pytest.approx([-1.2247448714, 0, 1.2247448714])
    assert _advantages([1, 1]) == [0, 0]
    for rewards in ([], [0, float("nan")], [float("inf")]):
        with pytest.raises(ValueError, match="finite"):
            _advantages(rewards)


def test_response_only_labels_preserve_input_and_reject_prefix_mismatch():
    full = torch.tensor([[1, 9, 2, 3]])
    labels = response_labels(full, torch.tensor([[1, 9]]))
    assert labels.tolist() == [[-100, -100, 2, 3]]
    assert full.tolist() == [[1, 9, 2, 3]]
    for prefix in (torch.tensor([[1, 8]]), full):
        with pytest.raises(ValueError, match="exact prompt"):
            response_labels(full, prefix)


def test_timestamps_preserve_microseconds_and_frame_order():
    example = {"prompt": "Describe.", "media": {"frames": [{"pts_us": 0}, {"pts_us": 33367}]}}
    messages = frame_messages(example, "Answer.")
    assert [item["type"] for item in messages[0]["content"]] == ["text", "image", "text", "image", "text"]
    assert messages[0]["content"][2]["text"].endswith("33367")
    assert messages[1]["content"][0]["text"] == "Answer."


def test_encoder_refuses_overflow_without_truncating_visual_prefix(monkeypatch):
    import io
    from PIL import Image
    import reliquary.vision.data

    buffer = io.BytesIO()
    Image.new("RGB", (4, 4)).save(buffer, format="PNG")
    monkeypatch.setattr(reliquary.vision.data, "load_verified_media_bytes",
                        lambda *_: [buffer.getvalue()])

    class Processor:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["add_generation_prompt"] is (len(messages) == 1)
            return "full" if len(messages) == 2 else "prefix"

        def __call__(self, *, text, images, common_kwargs, text_kwargs, images_kwargs):
            assert not text_kwargs["truncation"] and len(images[0]) == 1
            assert common_kwargs == {"return_tensors": "pt"}
            return {"input_ids": torch.tensor([[0, 1, 2, 3]] if text == ["full"] else [[0, 1]]),
                    "pixel_values": torch.ones((1, 3, 4, 4))}

    example = {"prompt": "Describe.", "media": {"frames": [{"pts_us": 0}]}}
    full = encode_example(Processor(), example, ".", 16, 4, "answer")
    assert full["labels"].tolist() == [[-100, -100, 2, 3]]
    with pytest.raises(ValueError, match="complete context"):
        encode_example(Processor(), example, ".", 16, 3, "answer")
    with pytest.raises(ValueError, match="pixel limit"):
        encode_example(Processor(), example, ".", 15, 4)


class TinyVision(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.forward_args = None
        self.generate_args = None

    def forward(self, **kwargs):
        self.forward_args = kwargs
        shape = (*kwargs["input_ids"].shape, 4)
        logits = self.scale * torch.arange(4).expand(shape) * kwargs["pixel_values"].mean()
        return SimpleNamespace(logits=logits)

    def generate(self, **kwargs):
        self.generate_args = kwargs
        kwargs["stopping_criteria"](kwargs["input_ids"], None)
        return torch.cat([kwargs["input_ids"], torch.tensor([[2, 3]])], dim=1)


def batch():
    return {
        "input_ids": torch.tensor([[0, 1]]), "attention_mask": torch.ones((1, 2), dtype=torch.long),
        "pixel_values": torch.ones((1, 3, 2, 2)), "mm_token_type_ids": torch.tensor([[0, 1]]),
    }


def test_recomputation_keeps_visual_inputs_and_extends_token_types():
    model = TinyVision()
    visual = batch()
    tokens = torch.tensor([[0, 1, 2, 3]])
    actual = response_logprobs(model, visual, tokens)
    expected = torch.log_softmax(torch.arange(4) * 0.5, -1)[torch.tensor([2, 3])]
    torch.testing.assert_close(actual, expected)
    assert model.forward_args["pixel_values"] is visual["pixel_values"]
    assert model.forward_args["mm_token_type_ids"].tolist() == [[0, 1, 0, 0]]
    assert actual.shape == (2,)
    actual.sum().backward()
    assert model.scale.grad is not None


def test_pixels_change_response_probabilities():
    model = TinyVision()
    visual = batch()
    first = response_logprobs(model, visual, torch.tensor([[0, 1, 2, 3]]))
    second = response_logprobs(model, {**visual, "pixel_values": visual["pixel_values"] * 0},
                              torch.tensor([[0, 1, 2, 3]]))
    assert not torch.allclose(first, second)


@pytest.mark.parametrize("tokens", [[[0, 2, 3]], [[0, 1]]])
def test_rollout_rejects_missing_response_or_changed_visual_prefix(tokens):
    with pytest.raises(ValueError):
        response_logprobs(TinyVision(), batch(), torch.tensor(tokens))


def test_generation_uses_full_policy_and_obeys_deadline():
    model = TinyVision()
    processor = SimpleNamespace(tokenizer=SimpleNamespace(
        eos_token_id=3, bos_token_id=0, pad_token_id=0,
        decode=lambda *_args, **_kwargs: "answer",
    ))
    calls = []
    rows = generate_group(model, processor, batch(), group_size=2, max_new_tokens=4,
                          sample=True, check_deadline=lambda: calls.append(True))
    assert len(rows) == 2 and all(not row["truncated"] for row in rows)
    config = model.generate_args["generation_config"]
    assert config.top_k == 0 and config.top_p == 1 and config.temperature == 1
    assert config.repetition_penalty == 1 and config.num_beams == 1
    assert config.pad_token_id == 0
    assert model.generation_config == SimpleNamespace(eos_token_id=3)
    assert all(not row["old"].requires_grad for row in rows)
    assert len(calls) >= 4
    with pytest.raises(TimeoutError):
        generate_group(model, processor, batch(), group_size=2, max_new_tokens=4,
                       sample=True, check_deadline=lambda: (_ for _ in ()).throw(TimeoutError()))


def test_trainable_vision_parameters_are_rejected():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    model = SimpleNamespace(named_parameters=lambda: [("vision_model.q_proj.lora_A", parameter)])
    with pytest.raises(ValueError, match="frozen"):
        _assert_language_adapter(model)
    model.named_parameters = lambda: [("model.language_model.q_proj.lora_A", parameter)]
    _assert_language_adapter(model)


def test_optimizer_requires_finite_nonzero_signal():
    model = TinyVision()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="nonzero"):
        _optimizer_step(model, optimizer)
    model.scale.grad = torch.tensor(float("nan"))
    with pytest.raises(RuntimeError, match="non-finite"):
        _optimizer_step(model, optimizer)
    model.scale.grad = torch.tensor(1.0)
    before = model.scale.detach().clone()
    assert _optimizer_step(model, optimizer) == 1
    assert not torch.equal(before, model.scale)


@pytest.fixture
def local_job(tmp_path, monkeypatch):
    """Exercise orchestration without downloading model weights in unit CI."""
    import json
    peft = pytest.importorskip("peft")
    import transformers
    from reliquary.vision import data, rewards, runtime

    class Adapter(TinyVision):
        config = SimpleNamespace(model_type="smolvlm",
                                 get_text_config=lambda: SimpleNamespace(max_position_embeddings=128))

        def named_parameters(self, *args, **kwargs):
            return iter([("model.text_model.q_proj.lora_A.weight", self.scale)])

        def forward(self, **kwargs):
            result = super().forward(**kwargs)
            result.loss = self.scale.square()
            return result

        def save_pretrained(self, directory, **kwargs):
            directory.mkdir()
            (directory / "adapter_config.json").write_text("{}")

        def unload(self):
            return self

    class Processor:
        def save_pretrained(self, directory):
            directory.mkdir()

    model = Adapter()
    monkeypatch.setattr(transformers.AutoModelForImageTextToText, "from_pretrained",
                        lambda *_args, **_kwargs: model)
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained",
                        lambda *_args, **_kwargs: Processor())
    monkeypatch.setattr(peft.PeftModel, "from_pretrained", lambda base, *_args, **_kwargs: base)
    monkeypatch.setattr(runtime, "language_lora", lambda base, rank: base)
    monkeypatch.setattr(data, "validate_manifest", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rewards, "reward", lambda example, answer, **kwargs: float(answer == "good"))
    monkeypatch.setattr(runtime, "_evaluate",
                        lambda *_args: ({"mean_reward": 0.5, "truncated": 0}, []))

    def encode(*args):
        visual = batch()
        if args[-1] is not None:
            visual["input_ids"] = torch.tensor([[0, 1, 2, 3]])
            visual["labels"] = torch.tensor([[-100, -100, 2, 3]])
        return visual

    monkeypatch.setattr(runtime, "encode_example", encode)

    def generate(model, processor, visual, **kwargs):
        rows = []
        for token, answer in ((2, "good"), (1, "bad")):
            tokens = torch.tensor([[0, 1, token, 3]])
            rows.append({"tokens": tokens, "old": response_logprobs(model, visual, tokens).detach(),
                         "answer": answer, "truncated": False})
        return rows

    monkeypatch.setattr(runtime, "generate_group", generate)
    request = {
        "processor_mode": runtime.PROCESSOR_MODE, "method": "sft_lora",
        "device": "cpu", "precision": "float32", "seed": 17, "max_steps": 1,
        "lora_rank": 2, "learning_rate": 1e-3, "max_pixels": 64,
        "max_context_tokens": 32, "max_new_tokens": 4, "max_eval_examples": 1,
        "group_size": 2, "model_id": "test/model", "model_revision": "a" * 40,
    }
    dataset = {"examples": [{"id": split, "split": split, "target": {"items": []}}
                            for split in ("train", "test")]}
    return runtime, request, dataset, tmp_path / "output", model


@pytest.mark.parametrize("method", ["sft_lora", "grpo_lora"])
def test_training_update_exports_portable_adapter_and_checks_reload(local_job, method):
    import json

    runtime, request, dataset, output, model = local_job
    request["method"] = method
    before = model.scale.detach().clone()
    result = runtime.run_job(request, dataset, ".", ".", output)
    assert result["steps"] == 1 and result["reloadVerified"] is True
    assert not torch.equal(before, model.scale)
    assert result["baseline"] == result["postRun"] == 0.5
    config = json.loads((output / "adapter" / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "test/model"
    assert config["revision"] == "a" * 40


@pytest.mark.parametrize("reason", ["truncated", "constant"])
def test_unusable_grpo_groups_never_publish_an_adapter(local_job, monkeypatch, reason):
    runtime, request, dataset, output, model = local_job
    request["method"] = "grpo_lora"
    original = runtime.generate_group

    def unusable(*args, **kwargs):
        rows = original(*args, **kwargs)
        for row in rows:
            row["truncated"] = reason == "truncated"
            row["answer"] = "bad"
        return rows

    monkeypatch.setattr(runtime, "generate_group", unusable)
    before = model.scale.detach().clone()
    with pytest.raises(ValueError, match="No usable training signal"):
        runtime.run_job(request, dataset, ".", ".", output)
    assert torch.equal(before, model.scale)
    assert not (output / "adapter").exists()
