"""Gate M1: do per-turn TOPLOC proofs built with vLLM's prefix cache ON pass an
HF prefill audit of the whole trajectory?

Generates multi-turn trajectories the way the agentic miner will (each turn's
prompt is the whole history, so later turns hit the cache), builds each turn's
proofs from ``turn_rows``, then verifies every span from one HF prefill.

    VLLM_ENABLE_V1_MULTIPROCESSING=0 VLLM_USE_V2_MODEL_RUNNER=0 \
      python scripts/agentic_proof_gate.py --model Qwen/Qwen3.8-27B --phase generate --work m1.json
    python scripts/agentic_proof_gate.py --model Qwen/Qwen3.8-27B --phase verify --work m1.json --out result.json
"""

import argparse
import base64
import json
import platform

import torch

from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS as PROOF

SEEDS = [
    "You are working in /testbed. Find where `parse_date` handles time zones and explain it.",
    "Write a bash command that counts Python files under src/, then refactor it into a function.",
    "A test fails with KeyError: 'id' in models.py. Describe how you would locate the bug.",
    "Explain the difference between a merge and a rebase, then show both commands.",
]
# A fixed observation between turns, like a tool result: the next turn's prompt
# extends the previous one, which is what makes vLLM serve it from cache.
OBSERVATION = "\n<tool_response>\n$ ls\nsrc  tests  setup.py  README.md\n</tool_response>\n"


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--model", required=True)
    p.add_argument("--phase", choices=["generate", "verify"], required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--out")
    p.add_argument("--trajectories", type=int, default=32)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--turn-tokens", type=int, default=512)
    p.add_argument("--no-prefix-cache", action="store_true", help="control run: prefix caching off")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return p.parse_args(argv)


def generate(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt

    from reliquary.miner.vllm_hidden_capture import capture_hidden_states, turn_rows
    from reliquary.protocol.toploc_proof import build_chunk_proofs

    tok = AutoTokenizer.from_pretrained(args.model)
    obs_ids = tok.encode(OBSERVATION, add_special_tokens=False)
    llm = LLM(model=args.model, dtype="bfloat16", enable_prefix_caching=not args.no_prefix_cache,
              max_model_len=args.turns * (args.turn_tokens + 64) + 1024,
              max_num_seqs=64, gpu_memory_utilization=args.gpu_memory_utilization,
              limit_mm_per_prompt={"image": 0, "video": 0})
    params = SamplingParams(temperature=1.0, top_p=1.0, max_tokens=args.turn_tokens)
    histories = [tok.encode(SEEDS[i % len(SEEDS)]) for i in range(args.trajectories)]
    prompt_lens = [len(h) for h in histories]
    spans = [[] for _ in histories]
    proofs = [[] for _ in histories]
    stats = {"cached_turns": 0, "turns": 0}
    with capture_hidden_states() as capture:
        for _turn in range(args.turns):
            outs = llm.generate([TokensPrompt(prompt_token_ids=h) for h in histories], params,
                                use_tqdm=False)
            for i, out in enumerate(outs):
                completion = list(out.outputs[0].token_ids)
                scheduled = len(histories[i]) - (out.num_cached_tokens or 0)
                rows = turn_rows(capture.pop(out.request_id), len(completion), prompt_rows=scheduled)
                stats["turns"] += 1
                stats["cached_turns"] += int(scheduled < len(histories[i]))
                start = len(histories[i])
                histories[i] = histories[i] + completion
                spans[i].append([start, len(histories[i])])
                proofs[i].append([base64.b64encode(p).decode() for p in
                                  build_chunk_proofs(rows, chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)])
                histories[i] = histories[i] + obs_ids
    with open(args.work, "w") as f:
        json.dump({"stats": stats, "trajectories": [
            {"tokens": h, "prompt_len": n, "spans": s, "proofs": p}
            for h, n, s, p in zip(histories, prompt_lens, spans, proofs)]}, f)
    print(f"generated {len(histories)} trajectories; {stats['cached_turns']}/{stats['turns']} turns hit the cache")


def verify(args):
    from huggingface_hub import snapshot_download

    from reliquary.protocol.toploc import sequence_verdict
    from reliquary.protocol.toploc_proof import verify_chunk_proofs
    from reliquary.shared.modeling import load_text_only_model
    from reliquary.validator.corpus_audit import span_hidden_states

    work = json.load(open(args.work))
    model = load_text_only_model(snapshot_download(args.model), torch_dtype=torch.bfloat16,
                                 attn_implementation="sdpa").to("cuda").eval()
    report = []
    for t in work["trajectories"]:
        rows = span_hidden_states(model, t["tokens"], [tuple(s) for s in t["spans"]])
        for span, hidden, proofs in zip(t["spans"], rows, t["proofs"]):
            results = verify_chunk_proofs(hidden, [base64.b64decode(p) for p in proofs],
                                          chunk_tokens=PROOF.chunk_tokens, topk=PROOF.topk)
            passed, reason = sequence_verdict(results, PROOF.thresholds())
            report.append({"span": span, "passed": passed, "reason": reason,
                           "chunks": [[r.exp_mismatches, r.mant_err_mean, r.mant_err_median] for r in results]})
    worst = max(c[0] for r in report for c in r["chunks"])
    result = {"gate": "M1", "model": args.model, "gpu": torch.cuda.get_device_name(),
              "host": platform.node(), "stats": work["stats"], "spans": len(report),
              "passed": sum(r["passed"] for r in report), "worst_exp": worst,
              "worst_mant_mean": max(c[1] for r in report for c in r["chunks"]),
              "worst_mant_median": max(c[2] for r in report for c in r["chunks"]),
              "report": report}
    json.dump(result, open(args.out, "w"))
    print(f"M1: {result['passed']}/{result['spans']} spans pass at 60/40/40; worst exp {worst}, "
          f"mant mean {result['worst_mant_mean']:.2f}, median {result['worst_mant_median']:.2f}")


def main(argv=None):
    args = parse_args(argv)
    generate(args) if args.phase == "generate" else verify(args)


if __name__ == "__main__":
    main()
