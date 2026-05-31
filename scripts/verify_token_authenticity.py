"""Acceptance: build ProofResult stats via HF forward, run the real check.

Asserts every measured-injected rollout is FLAGGED and every honest vLLM
completion PASSES. Run on the GPU box with the exact window checkpoint
(R0mAI/reliquary-sn-v23 @ 306c4af8). Inputs already staged on the box:
  /root/winners_replay.json  (16 measured injected rollouts)
  /root/vllm_gen.jsonl       (honest vLLM completions)
"""
import json

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from reliquary.constants import T_PROTO
from reliquary.validator.verifier import ProofResult, evaluate_token_authenticity

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/306c4af855889b3136765f7f7d589f4d7c133089"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, device_map="cuda").eval()
dev = next(model.parameters()).device


def stats(prefix, comp):
    ids = torch.tensor([prefix + comp], device=dev)
    with torch.no_grad():
        logits = model(ids).logits[0]
    P = len(prefix)
    probs = (logits[P - 1:P + len(comp) - 1].float() / T_PROTO).softmax(dim=-1)
    chosen = probs.gather(1, torch.tensor(comp, device=dev).unsqueeze(1)).squeeze(1)
    amax_p, amax_id = probs.max(dim=-1)
    return chosen.tolist(), amax_p.tolist(), amax_id.tolist()


def passes(prefix, comp):
    c, ap, ai = stats(prefix, comp)
    ok, _ = evaluate_token_authenticity(
        ProofResult(all_passed=True, passed=1, checked=1,
                    completion_chosen_probs=c, completion_argmax_probs=ap,
                    completion_argmax_ids=ai)
    )
    return ok


flagged = total_inj = 0
for it in json.load(open("/root/winners_replay.json")):
    for r in it["rollouts"]:
        if r["reward"] != 0.0:
            continue
        pl = len(r["tokens"]) - r["completion_length"]
        total_inj += 1
        if not passes(r["tokens"][:pl], r["tokens"][pl:]):
            flagged += 1
print(f"injected flagged: {flagged}/{total_inj}")

fp = total_h = 0
for line in list(open("/root/vllm_gen.jsonl"))[:200]:
    r = json.loads(line)
    pl, cl = r["prompt_length"], r["completion_length"]
    total_h += 1
    if not passes(r["tokens"][:pl], r["tokens"][pl:pl + cl]):
        fp += 1
print(f"honest false positives: {fp}/{total_h}")

assert flagged == total_inj, "some injected rollouts not flagged"
assert fp == 0, "false positive on honest completion"
print("ACCEPTANCE PASS")
