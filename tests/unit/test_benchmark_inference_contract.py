import importlib.util
import json
from pathlib import Path


SCRIPT = Path(__file__).parents[2] / "scripts" / "benchmark_inference_contract.py"
SPEC = importlib.util.spec_from_file_location("benchmark_inference_contract", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_load_prompts_preserves_dataset_prompt_idx(tmp_path):
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        json.dumps({"prompt_idx": 10567250, "prompt": "problem"}) + "\n",
        encoding="utf-8",
    )

    assert MODULE._load_prompts(path, []) == [
        {"prompt_idx": 10567250, "prompt": "problem"}
    ]
