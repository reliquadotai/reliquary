"""Bounded, local-only VLM LoRA execution for qualified private workers."""

from __future__ import annotations

import hashlib
import io
import json
import math
from pathlib import Path
from typing import Callable


PROCESSOR_MODE = "timestamped_frames_v1"
MODEL_TYPES = frozenset({"smolvlm", "qwen3_vl", "qwen2_5_vl"})


def _advantages(rewards: list[float]) -> list[float]:
    """Population-standardized GRPO advantages; constant groups have no signal."""
    if not rewards or not all(math.isfinite(x) for x in rewards):
        raise ValueError("Rewards must be a nonempty finite group")
    mean = sum(rewards) / len(rewards)
    std = (sum((x - mean) ** 2 for x in rewards) / len(rewards)) ** 0.5
    return [(x - mean) / std for x in rewards] if std >= 1e-8 else [0.0] * len(rewards)


def response_labels(full_ids, prefix_ids):
    import torch

    size = prefix_ids.shape[1]
    if size >= full_ids.shape[1] or not torch.equal(full_ids[:, :size], prefix_ids):
        raise ValueError("The supervised conversation must extend the exact prompt tokens")
    labels = full_ids.clone()
    labels[:, :size] = -100
    return labels


def frame_messages(example: dict, answer: str | None = None) -> list[dict]:
    content = []
    for frame in example["media"]["frames"]:
        # Integer microseconds avoid losing the original presentation timestamp.
        content.extend([
            {"type": "text", "text": f"Frame timestamp (clip-relative microseconds): {frame['pts_us']}"},
            {"type": "image"},
        ])
    content.append({"type": "text", "text": example["prompt"]})
    messages = [{"role": "user", "content": content}]
    if answer is not None:
        messages.append({"role": "assistant", "content": [{"type": "text", "text": answer}]})
    return messages


def encode_example(processor, example: dict, media_root, max_pixels: int,
                   max_context_tokens: int, answer: str | None = None) -> dict:
    """Keep the complete visual prefix; over-budget inputs are never truncated."""
    from PIL import Image
    from reliquary.vision.data import load_verified_media_bytes

    images = []
    for payload in load_verified_media_bytes(example, Path(media_root)):
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > max_pixels:
                raise ValueError("Frame exceeds the approved pixel limit")
            images.append(image.convert("RGB"))
    if not images:
        raise ValueError("At least one verified frame is required")

    def encode(text):
        options = {"do_image_splitting": False} if "SmolVLM" in type(processor).__name__ else {}
        return dict(processor(
            text=[text], images=[images], common_kwargs={"return_tensors": "pt"},
            text_kwargs={"padding": False, "truncation": False}, images_kwargs=options,
        ))

    is_smol = "SmolVLM" in type(processor).__name__
    template_options = {"processor_kwargs": {"do_sample_frames": False}} if is_smol else {}
    prefix = processor.apply_chat_template(frame_messages(example), tokenize=False,
                                           add_generation_prompt=True, **template_options)
    batch = encode(prefix)
    if answer is not None:
        full = processor.apply_chat_template(frame_messages(example, answer), tokenize=False,
                                             add_generation_prompt=False, **template_options)
        full_batch = encode(full)
        full_batch["labels"] = response_labels(full_batch["input_ids"], batch["input_ids"])
        batch = full_batch
    if "pixel_values" not in batch or batch["input_ids"].shape[1] > max_context_tokens:
        raise ValueError("Visual tensors are required and the complete context must fit")
    return batch


def _move(batch, model) -> dict:
    parameter = next(model.parameters())
    return {
        key: value.to(device=parameter.device,
                      dtype=parameter.dtype if value.is_floating_point() else value.dtype)
        for key, value in batch.items()
    }


def response_logprobs(model, batch: dict, tokens):
    """Recompute response probabilities with the exact images and visual prefix."""
    import torch

    prefix = batch["input_ids"].shape[1]
    if tokens.shape[0] != 1 or tokens.shape[1] <= prefix:
        raise ValueError("Expected one complete prompt followed by response tokens")
    if not torch.equal(tokens[:, :prefix], batch["input_ids"]):
        raise ValueError("Rollout prompt tokens differ from its visual input")
    kwargs = {key: value for key, value in batch.items() if key != "labels"}
    kwargs["input_ids"] = tokens
    kwargs["attention_mask"] = torch.ones_like(tokens)
    for key in ("token_type_ids", "mm_token_type_ids"):
        if key in kwargs:
            kwargs[key] = torch.cat([
                kwargs[key], kwargs[key].new_zeros((1, tokens.shape[1] - prefix))
            ], dim=1)
    logits = model(**kwargs, use_cache=False).logits[:, prefix - 1:-1, :].float()
    selected = torch.log_softmax(logits, -1).gather(-1, tokens[:, prefix:, None]).squeeze(-1)[0]
    if not torch.isfinite(selected).all():
        raise ValueError("Model returned non-finite response log probabilities")
    return selected


def generate_group(model, processor, batch: dict, *, group_size: int,
                   max_new_tokens: int, sample: bool,
                   check_deadline: Callable[[], None]) -> list[dict]:
    import torch
    from transformers import GenerationConfig, StoppingCriteria, StoppingCriteriaList

    class Deadline(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            check_deadline()
            return False

    eos = model.generation_config.eos_token_id
    if eos is None:
        eos = processor.tokenizer.eos_token_id
    eos_ids = eos if isinstance(eos, list) else [eos]
    if not eos_ids or any(type(token) is not int for token in eos_ids):
        raise ValueError("An explicit EOS token is required")
    # A fresh config prevents a checkpoint's top-k, repetition or forced-token rules
    # from silently changing the behavior policy used for GRPO recomputation.
    config = GenerationConfig(
        do_sample=sample, num_beams=1, temperature=1.0, top_p=1.0, top_k=0 if sample else 50,
        repetition_penalty=1.0,
        max_new_tokens=max_new_tokens, eos_token_id=eos,
        pad_token_id=(processor.tokenizer.pad_token_id
                      if processor.tokenizer.pad_token_id is not None else eos_ids[0]),
        bos_token_id=processor.tokenizer.bos_token_id,
    )
    model.eval()
    rows = []
    generator = model.get_base_model() if hasattr(model, "get_base_model") else model
    for _ in range(group_size):
        check_deadline()
        with torch.no_grad():
            original_config = generator.generation_config
            # Transformers fills unset options from model.generation_config even
            # when a caller supplies a new config; erase that implicit fallback.
            generator.generation_config = config
            try:
                tokens = model.generate(**batch, generation_config=config,
                                        stopping_criteria=StoppingCriteriaList([Deadline()]))
            finally:
                generator.generation_config = original_config
            old = response_logprobs(model, batch, tokens)
        completion = tokens[0, batch["input_ids"].shape[1]:]
        if not completion.numel():
            raise ValueError("Generation returned no response tokens")
        rows.append({
            "tokens": tokens, "old": old.detach(),
            "answer": processor.tokenizer.decode(completion, skip_special_tokens=True).strip(),
            "truncated": int(completion[-1]) not in eos_ids,
        })
    return rows


def _assert_language_adapter(model):
    trainable = [name for name, value in model.named_parameters() if value.requires_grad]
    if not trainable or any(
        "lora_" not in name or not (".text_model." in name or ".language_model." in name)
        or not (".q_proj.lora_" in name or ".v_proj.lora_" in name)
        for name in trainable
    ):
        raise ValueError("The vision encoder and base parameters must remain frozen")


def language_lora(model, rank: int):
    from peft import LoraConfig, get_peft_model

    if model.config.model_type not in MODEL_TYPES:
        raise ValueError("This model architecture has no approved language-only LoRA mapping")
    names = [
        name for name, _ in model.named_modules()
        if (".text_model." in name or ".language_model." in name)
        and name.endswith((".q_proj", ".v_proj"))
    ]
    if not names:
        raise ValueError("No language attention projection modules found")
    model = get_peft_model(model, LoraConfig(
        r=rank, lora_alpha=2 * rank, lora_dropout=0.0, target_modules=names, bias="none",
    ))
    _assert_language_adapter(model)
    return model


def _optimizer_step(model, optimizer) -> float:
    import torch

    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("No finite nonzero adapter gradient; no checkpoint published")
    optimizer.step()
    return norm


def _evaluate(model, processor, examples, encode, request, check_deadline):
    from reliquary.vision.rewards import reward

    predictions = []
    for example in examples:
        check_deadline()
        batch = encode(example)
        if batch["input_ids"].shape[1] + request["max_new_tokens"] > request["max_context_tokens"]:
            raise ValueError("The prompt plus completion budget exceeds the context limit")
        row = generate_group(
            model, processor, batch, group_size=1, max_new_tokens=request["max_new_tokens"],
            sample=False, check_deadline=check_deadline,
        )[0]
        score = 0.0 if row["truncated"] else reward(example, row["answer"], training=False)
        predictions.append({"id": example["id"], "answer": row["answer"],
                            "reward": score, "truncated": row["truncated"]})
    return {"mean_reward": sum(row["reward"] for row in predictions) / len(predictions),
            "truncated": sum(row["truncated"] for row in predictions)}, predictions


def run_job(request: dict, dataset: dict, media_root, model_directory,
            output_directory, adapter_directory=None, *,
            check_deadline: Callable[[], None] = lambda: None) -> dict:
    """Run one already validated, snapshot-bound private-worker job.

    The process supervisor owns the hard deadline, immutable snapshot checks,
    request bounds, result publication, and disposal of any failed work directory.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from reliquary.vision.data import validate_manifest
    from reliquary.vision.rewards import reward

    if request["processor_mode"] != PROCESSOR_MODE:
        raise ValueError("Unsupported multimodal processor contract")
    method = request["method"]
    if method not in {"sft_lora", "grpo_lora", "evaluate", "generate"}:
        raise ValueError("Unsupported vision method")
    validate_manifest(dataset, Path(media_root), verify_files=True)
    examples = dataset["examples"]
    training = method in {"sft_lora", "grpo_lora"}
    train = [example for example in examples if example["split"] == "train"]
    heldout = [example for example in examples if example["split"] in {"validation", "test"}]
    evaluation = heldout if training or method == "evaluate" else examples
    if not evaluation or len(evaluation) > request["max_eval_examples"]:
        raise ValueError("An explicit nonempty evaluation set within the approved bound is required")
    if training and not train:
        raise ValueError("Training needs examples in the train split")
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=False)
    check_deadline()
    torch.manual_seed(request["seed"])
    device = request["device"]
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError("Requested execution device is unavailable")
    if request["precision"] not in {"float32", "bf16"}:
        raise ValueError("Unsupported model precision")
    dtype = torch.float32 if request["precision"] == "float32" else torch.bfloat16
    if device == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise ValueError("Requested CUDA device does not support BF16")
    processor = AutoProcessor.from_pretrained(model_directory, local_files_only=True,
                                              trust_remote_code=False)
    model = AutoModelForImageTextToText.from_pretrained(
        model_directory, local_files_only=True, trust_remote_code=False,
        use_safetensors=True, dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    if model.config.model_type not in MODEL_TYPES:
        raise ValueError("Unsupported VLM architecture")
    position_limit = getattr(model.config.get_text_config(), "max_position_embeddings", None)
    if isinstance(position_limit, int) and request["max_context_tokens"] > position_limit:
        raise ValueError("Requested context exceeds the model's positional limit")
    if adapter_directory is not None:
        model = PeftModel.from_pretrained(model, adapter_directory, local_files_only=True,
                                         is_trainable=training)
        config = model.peft_config["default"]
        if training and (config.r != request["lora_rank"] or config.lora_dropout != 0
                         or config.bias != "none"):
            raise ValueError("Parent adapter differs from the requested LoRA training contract")
    elif training:
        model = language_lora(model, request["lora_rank"])
    if training:
        _assert_language_adapter(model)

    def encode(example, answer=None):
        check_deadline()
        return _move(encode_example(
            processor, example, media_root, request["max_pixels"],
            request["max_context_tokens"], answer,
        ), model)

    baseline, baseline_predictions = _evaluate(
        model, processor, evaluation, encode, request, check_deadline,
    )
    updates = 0
    skipped = {"truncated_groups": 0, "constant_reward_groups": 0}
    losses = []
    last_probe = None
    if training:
        parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(parameters, lr=request["learning_rate"])
        for step in range(request["max_steps"]):
            check_deadline()
            example = train[step % len(train)]
            optimizer.zero_grad(set_to_none=True)
            # Dropout stays disabled for both behavior and current policy forwards.
            model.eval()
            if method == "sft_lora":
                answer = json.dumps(example["target"], separators=(",", ":"), ensure_ascii=False)
                batch = encode(example, answer)
                prefix_batch = encode(example)
                if batch["input_ids"].shape[1] - prefix_batch["input_ids"].shape[1] > request["max_new_tokens"]:
                    raise ValueError("Supervised answer exceeds the approved completion budget")
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite supervised loss")
                loss.backward()
                last_probe = (prefix_batch, batch["input_ids"])
                loss_value = float(loss.detach())
            else:
                batch = encode(example)
                if batch["input_ids"].shape[1] + request["max_new_tokens"] > request["max_context_tokens"]:
                    raise ValueError("The rollout budget exceeds the complete context limit")
                rows = generate_group(
                    model, processor, batch, group_size=request["group_size"],
                    max_new_tokens=request["max_new_tokens"], sample=True,
                    check_deadline=check_deadline,
                )
                if any(row["truncated"] for row in rows):
                    skipped["truncated_groups"] += 1
                    continue
                advantages = _advantages([reward(example, row["answer"], training=True) for row in rows])
                if not any(advantages):
                    skipped["constant_reward_groups"] += 1
                    continue
                denominator = sum(row["old"].numel() for row in rows)
                loss_value = 0.0
                for row, advantage in zip(rows, advantages):
                    check_deadline()
                    current = response_logprobs(model, batch, row["tokens"])
                    ratio = (current - row["old"]).exp()
                    loss = -torch.minimum(ratio * advantage,
                                          ratio.clamp(0.8, 1.2) * advantage).sum() / denominator
                    if not torch.isfinite(loss):
                        raise ValueError("Non-finite GRPO loss")
                    loss.backward()
                    loss_value += float(loss.detach())
                last_probe = (batch, rows[0]["tokens"])
            _optimizer_step(model, optimizer)
            updates += 1
            losses.append(loss_value)
        if not updates:
            raise ValueError(f"No usable training signal; no adapter was produced ({skipped})")
        check_deadline()
        # Embeddings are frozen; explicit False also prevents PEFT from querying
        # the Hub for the parent model's vocabulary during an offline resume.
        model.save_pretrained(output / "adapter", safe_serialization=True, save_embedding_layers=False)
        adapter_config_path = output / "adapter" / "adapter_config.json"
        adapter_config = json.loads(adapter_config_path.read_text())
        adapter_config["base_model_name_or_path"] = request["model_id"]
        adapter_config["revision"] = request["model_revision"]
        adapter_config_path.write_text(json.dumps(adapter_config, indent=2) + "\n")
        processor.save_pretrained(output / "processor")

    post, predictions = (
        _evaluate(model, processor, evaluation, encode, request, check_deadline)
        if training else (baseline, baseline_predictions)
    )
    reload_verified = False
    if training:
        probe_batch, probe_tokens = last_probe
        with torch.no_grad():
            expected = response_logprobs(model, probe_batch, probe_tokens).detach().cpu()
        # Reload the exported adapter into the existing frozen base; this also
        # validates that only portable adapter weights are needed for delivery.
        base = model.unload()
        model = PeftModel.from_pretrained(base, output / "adapter",
                                         local_files_only=True).eval()
        with torch.no_grad():
            actual = response_logprobs(model, probe_batch, probe_tokens).detach().cpu()
        tolerance = 1e-5 if dtype == torch.float32 else 2e-2
        reload_verified = bool(torch.allclose(actual, expected, rtol=0, atol=tolerance))
        if not reload_verified:
            raise ValueError("Exported adapter failed reload parity")
    for name, rows in (("baseline-predictions.json", baseline_predictions),
                       ("predictions.json", predictions)):
        (output / name).write_text(json.dumps(rows, allow_nan=False) + "\n")
    evaluation_digest = hashlib.sha256(
        json.dumps(evaluation, sort_keys=True, separators=(",", ":"),
                   ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()
    check_deadline()
    return {
        "baseline": baseline["mean_reward"], "postRun": post["mean_reward"],
        "evaluationDiagnostics": {"baseline": baseline, "postRun": post},
        "samples": len(evaluation),
        "evaluationSetDigest": evaluation_digest, "steps": updates,
        "device": device, "precision": request["precision"], "metric": "mean_task_reward",
        "reloadVerified": reload_verified, "processorMode": PROCESSOR_MODE,
        "adapter": "adapter" if training else None, "losses": losses, "skipped": skipped,
    }
