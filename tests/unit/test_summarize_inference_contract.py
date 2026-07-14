import importlib.util
import json
import sys
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "summarize_inference_contract.py"
SPEC = importlib.util.spec_from_file_location("summarize_inference_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _artifact(profile, completion_hash):
    return {
        "model_revision_resolved": "revision",
        "checkpoint_hash": "checkpoint",
        "prompts_sha256": "corpus",
        "profile_label": profile,
        "replicate": 1,
        "config": {
            "batch_size": 1,
            "max_new_tokens": 1,
            "dtype": "bfloat16",
            "generation_use_cache": True,
            "bft_thinking_budget": 0,
            "bft_answer_budget": 0,
        },
        "runtime_profile": {"profile_hash": profile},
        "summary": {
            "n_positions": 1,
            "n_hard_mismatch": 0,
            "generated_tokens": 1,
            "elapsed_seconds": 1.0,
            "generation_seconds": 0.5,
            "teacher_force_seconds": 0.5,
            "cuda_peak_allocated_bytes": 1,
        },
        "rollouts": [
            {
                "prompt_idx": 7,
                "rollout_idx": 0,
                "completion_sha256": completion_hash,
                "n_stochastic": 1,
                "n_exact_match": 1,
                "ended_eos": False,
                "forced": False,
                "bft_termination_path": None,
                "unique_token_ratio": 1.0,
                "repeated_ngram_fraction": 0.0,
            }
        ],
    }


def test_summary_reports_pairwise_profile_completion_agreement(
    tmp_path, monkeypatch, capsys,
):
    left = tmp_path / "left.json"
    right = tmp_path / "right.json"
    left.write_text(json.dumps(_artifact("left", "same")), encoding="utf-8")
    right.write_text(json.dumps(_artifact("right", "same")), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(left), str(right)])

    MODULE.main()

    report = json.loads(capsys.readouterr().out)
    pair = report["pairwise_profile_completion_agreement"][0]
    assert pair["comparable_rollouts"] == 1
    assert pair["completion_agreement"] == 1.0
