"""The Miles gateway's settings out of the box: an RL-orders pod gets no RELIQUARY_MILES_* variable, so the card
count, tensor parallelism, Megatron model type, extra flags and chunk all have working defaults."""
import logging

import pytest

from reliquary.trainer.miles_gateway import (
    DEFAULT_CHUNK, DEFAULT_EXTRA_ARGS, miles_model_type, miles_settings, visible_gpu_count,
)

IMAGE = {"RELIQUARY_MILES_LAUNCHER": "/root/miles/scripts/run_reliquary_tinker.py",
         "RELIQUARY_MILES_OUTPUT_DIR": "/root/trainer/miles"}           # what Dockerfile.trainer-miles sets
POD = {**IMAGE, "RELIQUARY_RL_BASE_MODEL": "Qwen/Qwen3.5-4B",
       "RELIQUARY_MODEL": "Qwen/Qwen3.5-4B@851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"}


def smi(n):
    return lambda: "".join(f"GPU {i}: NVIDIA H200 (UUID: GPU-{i})\n" for i in range(n))


@pytest.mark.parametrize("cards", [2, 8])
def test_a_platform_pod_needs_no_miles_variable(cards):
    settings = miles_settings(POD, smi=smi(cards))
    assert settings["gpus"] == cards and settings["tp"] == 2
    assert settings["model_type"] == "qwen3.5-4B"
    assert settings["extra_args"] == DEFAULT_EXTRA_ARGS == (
        "--max-tokens-per-gpu 65536 --recompute-granularity full --recompute-method uniform --recompute-num-layers 1")
    assert settings["chunk"] == DEFAULT_CHUNK == 256


def test_the_card_count_follows_cuda_visible_devices_first():
    assert visible_gpu_count({"CUDA_VISIBLE_DEVICES": "0,3,5"}, smi=smi(8)) == 3
    assert visible_gpu_count({"CUDA_VISIBLE_DEVICES": " "}, smi=smi(2)) == 2      # blank: ask nvidia-smi
    assert visible_gpu_count({}, smi=smi(4)) == 4


def test_no_way_to_count_falls_back_to_eight_with_a_warning(caplog):
    def missing():
        raise FileNotFoundError("nvidia-smi")

    caplog.set_level(logging.WARNING, logger="reliquary.trainer.miles_gateway")
    assert visible_gpu_count({}, smi=missing) == 8
    assert visible_gpu_count({}, smi=lambda: "") == 8
    assert sum("8" in r.getMessage() for r in caplog.records) == 2


def test_tensor_parallelism_fits_the_cards():
    assert miles_settings({**POD, "RELIQUARY_MILES_GPUS": "1"}, smi=smi(8))["tp"] == 1      # default capped
    with pytest.raises(ValueError, match="divisible"):
        miles_settings({**POD, "RELIQUARY_MILES_GPUS": "6", "RELIQUARY_MILES_TP": "4"})
    with pytest.raises(ValueError, match="divisible"):
        miles_settings(POD, smi=smi(3))                                                    # 3 cards, TP2
    with pytest.raises(ValueError, match="RELIQUARY_MILES_TP"):
        miles_settings({**POD, "RELIQUARY_MILES_TP": "4"}, smi=smi(2))                      # explicit, over the cards
    assert miles_settings({**POD, "RELIQUARY_MILES_TP": "4"}, smi=smi(8))["tp"] == 4


@pytest.mark.parametrize("model,kind", [
    ("Qwen/Qwen3.5-4B", "qwen3.5-4B"), ("Qwen/Qwen3.5-9B", "qwen3.5-9B"), ("qwen/qwen3.5-0.8b", "qwen3.5-0.8B"),
    ("Qwen/Qwen3.5-27B@abc", "qwen3.5-27B"), ("Qwen/Qwen3.5-35B-A3B", "qwen3.5-35B-A3B"),
    ("/data/models/Qwen3.5-4B", "qwen3.5-4B")])
def test_the_megatron_model_type_comes_from_the_model_name(model, kind):
    assert miles_model_type(model) == kind


def test_the_model_type_reads_the_base_model_then_the_order_model_and_an_explicit_value_wins():
    assert miles_settings({**IMAGE, "RELIQUARY_MODEL": "Qwen/Qwen3.5-9B@x"}, smi=smi(2))["model_type"] == "qwen3.5-9B"
    assert miles_settings(IMAGE, smi=smi(2), base_model="Qwen/Qwen3.5-9B")["model_type"] == "qwen3.5-9B"
    explicit = {**POD, "RELIQUARY_MILES_MODEL_TYPE": "qwen3.5-9B"}
    assert miles_settings(explicit, smi=smi(2))["model_type"] == "qwen3.5-9B"


@pytest.mark.parametrize("model", ["Qwen/Qwen3.5-2B", "meta-llama/Llama-3-8B", None])
def test_an_unknown_model_names_the_variable_to_set(model):
    env = IMAGE if model is None else {**IMAGE, "RELIQUARY_RL_BASE_MODEL": model}
    with pytest.raises(ValueError, match="RELIQUARY_MILES_MODEL_TYPE"):
        miles_settings(env, smi=smi(2))


def test_explicit_extra_args_replace_the_defaults_entirely():
    settings = miles_settings({**POD, "RELIQUARY_MILES_EXTRA_ARGS": "--max-tokens-per-gpu 32768"}, smi=smi(2))
    assert settings["extra_args"] == "--max-tokens-per-gpu 32768"
    assert miles_settings({**POD, "RELIQUARY_MILES_EXTRA_ARGS": ""}, smi=smi(2))["extra_args"] == ""
    assert miles_settings({**POD, "RELIQUARY_MILES_CHUNK": "64"}, smi=smi(2))["chunk"] == 64
