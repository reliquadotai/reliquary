# Rollout token authenticity — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reject submissions containing post-hoc token injection by flagging any completion position where the chosen-token probability collapsed while the model's argmax was near-certain.

**Architecture:** The validator's GPU forward (`verify_commitment_proofs`) already softmaxes every completion position. We add the argmax probability/id to `ProofResult`, then a pure check `evaluate_token_authenticity(proof)` flags `p_chosen < 1e-8 AND p_argmax >= 0.99`. Wired into the per-rollout verification loop in the batcher behind a shadow/enforce flag; any flagged rollout rejects the whole submission.

**Tech Stack:** Python, PyTorch (bf16 GPU forward), pytest. Files: `reliquary/constants.py`, `reliquary/protocol/submission.py`, `reliquary/validator/verifier.py`, `reliquary/validator/batcher.py`, `tests/unit/test_behavioural_validators.py`.

---

## File structure

- `reliquary/constants.py` — add `TOKEN_AUTH_THRESHOLD`, `TOKEN_AUTH_ARGMAX_CONF`, `TOKEN_AUTH_ENFORCE`.
- `reliquary/protocol/submission.py` — add `RejectReason.TOKEN_TAMPERED`.
- `reliquary/validator/verifier.py` — add 2 `ProofResult` fields; rename/extend `_gpu_completion_chosen_probs` → `_gpu_completion_token_stats` (also returns argmax prob+id); add `evaluate_token_authenticity`.
- `reliquary/validator/batcher.py` — import + call the check in the per-rollout loop with shadow/enforce gating.
- `tests/unit/test_behavioural_validators.py` — unit tests for the check and the GPU stats function.

Decision uses only `completion_chosen_probs` + `completion_argmax_probs` (both aligned). No tokenizer needed (the `argmax != token` condition is redundant given `p_chosen < threshold`, and the rule is intentionally not digit-restricted).

---

### Task 1: Constants

**Files:**
- Modify: `reliquary/constants.py` (near `BOXED_ANSWER_MIN_PROB`, ~line 515)

- [ ] **Step 1: Add the constants**

```python
# Token authenticity: a completion token whose chosen probability collapses
# below this while the model's argmax sits at >= TOKEN_AUTH_ARGMAX_CONF was not
# sampled — it was injected. Calibrated on 550k honest vLLM->HF tokens (floor
# 3.5e-7); measured injections <= 1e-13.
TOKEN_AUTH_THRESHOLD = 1e-8
TOKEN_AUTH_ARGMAX_CONF = 0.99
# Shadow mode: compute + log the check without rejecting. Flip to True once prod
# shadow logs confirm zero false positives.
TOKEN_AUTH_ENFORCE = False
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from reliquary.constants import TOKEN_AUTH_THRESHOLD, TOKEN_AUTH_ARGMAX_CONF, TOKEN_AUTH_ENFORCE; print(TOKEN_AUTH_THRESHOLD, TOKEN_AUTH_ARGMAX_CONF, TOKEN_AUTH_ENFORCE)"`
Expected: `1e-08 0.99 False`

- [ ] **Step 3: Commit**

```bash
git add reliquary/constants.py
git commit -m "feat(validator): add token-authenticity thresholds"
```

---

### Task 2: RejectReason

**Files:**
- Modify: `reliquary/protocol/submission.py` (RejectReason enum, after `BOXED_ANSWER_TAMPERED`)

- [ ] **Step 1: Add the enum value**

```python
    TOKEN_TAMPERED = "token_tampered"
```

- [ ] **Step 2: Verify**

Run: `python -c "from reliquary.protocol.submission import RejectReason; print(RejectReason.TOKEN_TAMPERED.value)"`
Expected: `token_tampered`

- [ ] **Step 3: Commit**

```bash
git add reliquary/protocol/submission.py
git commit -m "feat(protocol): add TOKEN_TAMPERED reject reason"
```

---

### Task 3: ProofResult fields

**Files:**
- Modify: `reliquary/validator/verifier.py` (ProofResult dataclass, after `completion_chosen_probs`, ~line 70)

- [ ] **Step 1: Add the fields**

```python
    # Token authenticity: argmax probability and argmax token id under T_PROTO,
    # aligned 1:1 with completion_chosen_probs (same surviving steps).
    completion_argmax_probs: list[float] = field(default_factory=list)
    completion_argmax_ids: list[int] = field(default_factory=list)
```

- [ ] **Step 2: Verify construction with defaults**

Run: `python -c "from reliquary.validator.verifier import ProofResult; p=ProofResult(all_passed=True,passed=1,checked=1); print(p.completion_argmax_probs, p.completion_argmax_ids)"`
Expected: `[] []`

- [ ] **Step 3: Commit**

```bash
git add reliquary/validator/verifier.py
git commit -m "feat(validator): carry argmax prob/id on ProofResult"
```

---

### Task 4: GPU stats function (chosen + argmax)

**Files:**
- Modify: `reliquary/validator/verifier.py` (`_gpu_completion_chosen_probs`, ~line 402; caller in `verify_commitment_proofs`, ~line 304)
- Test: `tests/unit/test_behavioural_validators.py`

- [ ] **Step 1: Write the failing test (CPU tensor, no GPU)**

Add to `tests/unit/test_behavioural_validators.py`:

```python
import torch
from reliquary.constants import T_PROTO


def test_gpu_completion_token_stats_returns_chosen_and_argmax():
    # 2 completion positions, vocab 4. Build logits so argmax is known.
    # tokens: prompt_length=1, completion at indices 1,2.
    seq_len = 3
    logits = torch.zeros(seq_len, 4)
    logits[0] = torch.tensor([0.0, 5.0, 0.0, 0.0])   # predicts token at idx1
    logits[1] = torch.tensor([0.0, 0.0, 0.0, 9.0])   # predicts token at idx2
    tokens = [0, 1, 3]  # idx1 token=1 (the argmax), idx2 token=3 (the argmax)

    chosen, amax_p, amax_id = verifier._gpu_completion_token_stats(
        logits, tokens, prompt_length=1, completion_length=2,
        seq_len=seq_len, device="cpu",
    )
    assert len(chosen) == len(amax_p) == len(amax_id) == 2
    # token equals argmax at both positions -> chosen prob == argmax prob
    assert amax_id == [1, 3]
    for c, a in zip(chosen, amax_p):
        assert abs(c - a) < 1e-6
        assert a > 0.9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/unit/test_behavioural_validators.py::test_gpu_completion_token_stats_returns_chosen_and_argmax -v`
Expected: FAIL — `AttributeError: module 'reliquary.validator.verifier' has no attribute '_gpu_completion_token_stats'`

- [ ] **Step 3: Rename and extend the function**

Replace `_gpu_completion_chosen_probs` with:

```python
def _gpu_completion_token_stats(
    logits_gpu: torch.Tensor,
    tokens: list[int],
    prompt_length: int,
    completion_length: int,
    seq_len: int,
    device: Any,
) -> tuple[list[float], list[float], list[int]]:
    """Per completion-producing position under T_PROTO, on GPU, vectorised:
    chosen-token prob, argmax prob, argmax token id. The three lists are
    aligned 1:1. Boundary positions (t == 0, t - 1 >= seq_len, t >= len(tokens))
    are skipped identically across all three.
    """
    if completion_length <= 0:
        return [], [], []
    t_start = prompt_length
    t_end = min(prompt_length + completion_length, len(tokens), seq_len + 1)
    valid_t = [t for t in range(t_start, t_end) if t > 0 and t - 1 < seq_len]
    if not valid_t:
        return [], [], []

    pos_tensor = torch.tensor(
        [t - 1 for t in valid_t], device=device, dtype=torch.long,
    )
    tok_tensor = torch.tensor(
        [tokens[t] for t in valid_t], device=device, dtype=torch.long,
    )
    scaled = logits_gpu[pos_tensor].float() / float(T_PROTO)
    probs = scaled.softmax(dim=-1)
    chosen = probs.gather(1, tok_tensor.unsqueeze(1)).squeeze(1)
    amax_probs, amax_ids = probs.max(dim=-1)
    return chosen.tolist(), amax_probs.tolist(), amax_ids.tolist()
```

- [ ] **Step 4: Update the caller in `verify_commitment_proofs`**

Replace the call (~line 304):

```python
    (
        completion_chosen_probs,
        completion_argmax_probs,
        completion_argmax_ids,
    ) = _gpu_completion_token_stats(
        logits_gpu, tokens, prompt_length, completion_length, seq_len, device,
    )
```

And add to the `ProofResult(...)` return (after `completion_chosen_probs=...`):

```python
        completion_argmax_probs=completion_argmax_probs,
        completion_argmax_ids=completion_argmax_ids,
```

- [ ] **Step 5: Run test to verify it passes**

Run: `pytest tests/unit/test_behavioural_validators.py::test_gpu_completion_token_stats_returns_chosen_and_argmax -v`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/verifier.py tests/unit/test_behavioural_validators.py
git commit -m "feat(validator): compute argmax stats alongside chosen probs"
```

---

### Task 5: The authenticity check

**Files:**
- Modify: `reliquary/validator/verifier.py` (add `evaluate_token_authenticity` after `evaluate_token_distribution`)
- Test: `tests/unit/test_behavioural_validators.py`

- [ ] **Step 1: Write the failing tests**

Add to `tests/unit/test_behavioural_validators.py`:

```python
def _proof_with_token_stats(chosen, argmax_probs, argmax_ids=None):
    return ProofResult(
        all_passed=True, passed=1, checked=1, has_sparse_outputs=True,
        completion_chosen_probs=chosen,
        completion_argmax_probs=argmax_probs,
        completion_argmax_ids=argmax_ids or [0] * len(chosen),
    )


def test_token_authenticity_honest_passes():
    proof = _proof_with_token_stats([0.99, 0.8, 1.0], [0.99, 0.8, 1.0])
    ok, m = verifier.evaluate_token_authenticity(proof)
    assert ok is True and m == {}


def test_token_authenticity_injection_fails():
    # one collapsed token while the model was near-certain of something else
    proof = _proof_with_token_stats(
        [1.0, 2.0e-20, 1.0], [1.0, 1.0, 1.0], argmax_ids=[5, 7, 9],
    )
    ok, m = verifier.evaluate_token_authenticity(proof)
    assert ok is False
    assert m["pos"] == 1 and m["argmax_id"] == 7


def test_token_authenticity_high_entropy_honest_passes():
    # low chosen prob but no confident argmax -> genuine uncertainty, accept
    proof = _proof_with_token_stats([1.0, 1.0e-9, 1.0], [1.0, 0.30, 1.0])
    ok, _ = verifier.evaluate_token_authenticity(proof)
    assert ok is True


def test_token_authenticity_empty_skips():
    ok, m = verifier.evaluate_token_authenticity(
        ProofResult(all_passed=True, passed=1, checked=1)
    )
    assert ok is True and m == {}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/unit/test_behavioural_validators.py -k token_authenticity -v`
Expected: FAIL — `AttributeError: ... has no attribute 'evaluate_token_authenticity'`

- [ ] **Step 3: Implement the check**

Add to `reliquary/validator/verifier.py`:

```python
def evaluate_token_authenticity(
    proof: "ProofResult",
    *,
    threshold: float | None = None,
    argmax_conf: float | None = None,
) -> tuple[bool, dict]:
    """Hard check: a completion token sampled at T_PROTO can never have
    chosen probability below ``threshold`` while the model's argmax sits at
    >= ``argmax_conf`` — that pattern is a post-hoc injection. Reads the
    aligned ``completion_chosen_probs`` / ``completion_argmax_probs`` from the
    GPU forward; no tokenizer needed. ``ok=True`` when no stats are available.
    """
    from reliquary.constants import TOKEN_AUTH_ARGMAX_CONF, TOKEN_AUTH_THRESHOLD

    if threshold is None:
        threshold = TOKEN_AUTH_THRESHOLD
    if argmax_conf is None:
        argmax_conf = TOKEN_AUTH_ARGMAX_CONF
    chosen = proof.completion_chosen_probs
    amax = proof.completion_argmax_probs
    if not chosen or not amax:
        return True, {}
    n = min(len(chosen), len(amax))
    for j in range(n):
        if chosen[j] < threshold and amax[j] >= argmax_conf:
            ids = proof.completion_argmax_ids
            return False, {
                "pos": j,
                "p_chosen": float(chosen[j]),
                "p_argmax": float(amax[j]),
                "argmax_id": (ids[j] if j < len(ids) else None),
            }
    return True, {}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/unit/test_behavioural_validators.py -k token_authenticity -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/verifier.py tests/unit/test_behavioural_validators.py
git commit -m "feat(validator): add evaluate_token_authenticity check"
```

---

### Task 6: Wire into the batcher (shadow/enforce)

**Files:**
- Modify: `reliquary/validator/batcher.py` (import block ~line 58; per-rollout loop after the boxed check, ~line 905)

- [ ] **Step 1: Add the import**

In the verifier import block (around line 58, next to `evaluate_boxed_answer_probability`):

```python
    evaluate_token_authenticity,
```

And import the enforce flag where constants are imported in this file:

```python
from reliquary.constants import TOKEN_AUTH_ENFORCE
```

(Add to the existing `from reliquary.constants import (...)` group if one exists; otherwise a new import line.)

- [ ] **Step 2: Add the check after the boxed-answer block**

Immediately after the `evaluate_boxed_answer_probability` block (after its `return reject(...)`, ~line 905), inside the same per-rollout loop:

```python
            auth_ok, auth_metrics = evaluate_token_authenticity(proof)
            if not auth_ok:
                logger.info(
                    "token_tampered hotkey=%s enforce=%s %s",
                    request.miner_hotkey, TOKEN_AUTH_ENFORCE, auth_metrics,
                )
                if TOKEN_AUTH_ENFORCE:
                    return reject(
                        RejectReason.TOKEN_TAMPERED,
                        "token_authenticity",
                        sketch_diff_max=sketch_diff_max,
                        lp_dev_max=lp_dev_max,
                        dist_q10_min=dist_q10_min,
                    )
```

- [ ] **Step 3: Run the full validator suite (non-regression + wiring)**

Run: `pytest tests/unit/test_behavioural_validators.py tests/unit/test_grpo_window_batcher.py -v`
Expected: PASS (all existing tests still green; shadow default means no new rejections).

- [ ] **Step 4: Commit**

```bash
git add reliquary/validator/batcher.py
git commit -m "feat(validator): wire token-authenticity check (shadow mode)"
```

---

### Task 7: Empirical acceptance on real data (GPU box)

**Files:**
- Create: `scripts/verify_token_authenticity.py`

Validates the end-to-end behaviour on the GPU box (`<gpu-box>`, venv `/root/vllmenv`) using the exact window checkpoint. Reuses `/root/winners_replay.json` (16 measured injected rollouts) and `/root/vllm_gen.jsonl` (800 honest vLLM completions).

- [ ] **Step 1: Write the acceptance script**

```python
"""Acceptance: build ProofResult stats via HF forward, run the real check.
Asserts: every measured-injected rollout is FLAGGED; every honest vLLM
completion PASSES. Run on the GPU box with the exact ckpt 306c4af8.
"""
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from reliquary.validator.verifier import ProofResult, evaluate_token_authenticity
from reliquary.constants import T_PROTO

CKPT = "/root/.cache/huggingface/hub/models--R0mAI--reliquary-sn-v23/snapshots/306c4af855889b3136765f7f7d589f4d7c133089"
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")
model = AutoModelForCausalLM.from_pretrained(CKPT, dtype=torch.bfloat16, device_map="cuda").eval()
dev = next(model.parameters()).device


def stats(prefix, comp):
    ids = torch.tensor([prefix + comp], device=dev)
    with torch.no_grad():
        logits = model(ids).logits[0]
    P = len(prefix)
    probs = (logits[P-1:P+len(comp)-1].float() / T_PROTO).softmax(dim=-1)
    chosen = probs.gather(1, torch.tensor(comp, device=dev).unsqueeze(1)).squeeze(1)
    amax_p, amax_id = probs.max(dim=-1)
    return chosen.tolist(), amax_p.tolist(), amax_id.tolist()


def check(prefix, comp):
    c, ap, ai = stats(prefix, comp)
    ok, _ = evaluate_token_authenticity(
        ProofResult(all_passed=True, passed=1, checked=1,
                    completion_chosen_probs=c, completion_argmax_probs=ap,
                    completion_argmax_ids=ai)
    )
    return ok

# injected rollouts must be flagged
flagged = total_inj = 0
for it in json.load(open("/root/winners_replay.json")):
    for r in it["rollouts"]:
        if r["reward"] != 0.0:
            continue
        pl = len(r["tokens"]) - r["completion_length"]
        total_inj += 1
        if not check(r["tokens"][:pl], r["tokens"][pl:]):
            flagged += 1
print(f"injected flagged: {flagged}/{total_inj}")

# honest completions must pass (sample 200 for speed)
fp = total_h = 0
for line in list(open("/root/vllm_gen.jsonl"))[:200]:
    r = json.loads(line)
    pl, cl = r["prompt_length"], r["completion_length"]
    total_h += 1
    if not check(r["tokens"][:pl], r["tokens"][pl:pl+cl]):
        fp += 1
print(f"honest false positives: {fp}/{total_h}")
assert flagged == total_inj, "some injected rollouts not flagged"
assert fp == 0, "false positive on honest completion"
print("ACCEPTANCE PASS")
```

- [ ] **Step 2: Run on the GPU box**

```bash
scp scripts/verify_token_authenticity.py <gpu-box>:/root/reliquary/scripts/
ssh <gpu-box> 'cd /root/reliquary && /root/vllmenv/bin/python scripts/verify_token_authenticity.py'
```
Expected: `injected flagged: 16/16`, `honest false positives: 0/200`, `ACCEPTANCE PASS`

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_token_authenticity.py
git commit -m "test(validator): empirical token-authenticity acceptance on real rollouts"
```

---

## Deployment (post-merge, separate from this plan)

1. Deploy with `TOKEN_AUTH_ENFORCE=False` (shadow). Watch logs for `token_tampered` lines: confirm the fabricating hotkeys are flagged and honest miners are not, across ≥1 day.
2. Flip `TOKEN_AUTH_ENFORCE=True` to enforce.
3. (Code env, separate work) Ensure the code-exec env reward is recomputed by the validator from submitted tokens before this check is relied on there.
