"""Measure TOPLOC's honest band (or a substitution's) on this machine.

Miner: vLLM, production defaults (CUDA graphs, torch.compile), proofs from decode
rows. Verifier: HF, bf16, prefill. Run once with --quantization none (honest)
and once with fp8 (the cheap substitution) per card, then compare the JSONs:

    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
      python scripts/toploc_band.py --model Qwen/Qwen3-4B-Base --out band.json
"""

import argparse
import gc
import json
import platform

import torch

from reliquary.miner.vllm_hidden_capture import capture_hidden_states, completion_rows
from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF
from reliquary.protocol.toploc import sequence_verdict
from reliquary.protocol.toploc_proof import build_chunk_proofs, verify_chunk_proofs
from reliquary.validator.corpus_audit import completion_hidden_states

PROMPTS = [
    "Question: Solve for x: 3x + 7 = 22. Show your work.\nAnswer:",
    "Write a Python function that returns the n-th Fibonacci number.\n\ndef fib(n):",
    "Explain why the sky appears blue, in three sentences.\n\n",
    "Question: What is the derivative of x^3 * sin(x)?\nAnswer:",
]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", required=True)
    parser.add_argument("--quantization", choices=["none", "fp8"], default="none")
    parser.add_argument("--rollouts", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=256)
    # Spec measurement 5: does a coarser chunk keep the separation at 4x less proof?
    parser.add_argument("--chunk-tokens", type=int, default=PROOF.chunk_tokens)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    from vllm import LLM, SamplingParams
    from transformers import AutoModelForCausalLM

    llm = LLM(model=args.model, dtype="bfloat16", seed=1234, max_model_len=2048,
              gpu_memory_utilization=0.5, enable_prefix_caching=False,
              quantization=None if args.quantization == "none" else args.quantization)
    prompts = [PROMPTS[i % len(PROMPTS)] for i in range(args.rollouts)]
    params = [SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.max_tokens, seed=1000 + i)
              for i in range(args.rollouts)]
    with capture_hidden_states() as capture:
        outputs = llm.generate(prompts, params, use_tqdm=False)
    miners = []
    for output in outputs:
        tokens = list(output.prompt_token_ids) + list(output.outputs[0].token_ids)
        prompt_len = len(output.prompt_token_ids)
        rows = completion_rows(capture.for_request(output.request_id), prompt_len, len(tokens))
        proofs = build_chunk_proofs(rows, chunk_tokens=args.chunk_tokens, topk=PROOF.topk)
        miners.append((tokens, prompt_len, proofs))
    del llm
    gc.collect()
    torch.cuda.empty_cache()

    verifier = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to("cuda").eval()
    report = []
    for tokens, prompt_len, proofs in miners:
        hidden = completion_hidden_states(verifier, tokens, prompt_len)
        results = verify_chunk_proofs(hidden, proofs, chunk_tokens=args.chunk_tokens, topk=PROOF.topk)
        passed, reason = sequence_verdict(results, PROOF.thresholds())
        report.append({"passed": passed, "reason": reason,
                       "chunks": [[r.exp_mismatches, r.mant_err_mean, r.mant_err_median] for r in results]})
    worst = max(c[0] for r in report for c in r["chunks"])
    with open(args.out, "w") as handle:
        json.dump({"gpu": torch.cuda.get_device_name(), "host": platform.node(),
                   "torch": torch.__version__, "model": args.model,
                   "quantization": args.quantization, "chunk_tokens": args.chunk_tokens,
                   "rollouts": report}, handle)
    print(f"{args.quantization}: {sum(r['passed'] for r in report)}/{len(report)} pass at "
          f"{PROOF.exp_mismatch_threshold}/{PROOF.mant_mean_threshold:g}/{PROOF.mant_median_threshold:g}; "
          f"worst chunk exp mismatches {worst}")


if __name__ == "__main__":
    main()
