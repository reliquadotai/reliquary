# CoT / BFT — Faithful Implementation of the 0xgrizz Brief

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the CoT design from the 0xgrizz brief (2026-06-26) on `feat/cot-2b` exactly as specified, so the team can then prune what to keep.

**Architecture:** A thinking model is made trainable under the strict σ-gate by *Budget-Forced Termination* (BFT): the miner injects `</think>\n\nFinal Answer: \boxed{` at a thinking budget B and samples the answer; the validator carves the injected span out of the GRAIL authenticity checks (exact-match instead of prob-check) while a sampler-faithful recompute replays the miner's sampler; the trainer masks the forced span from the loss; a two-sided reward shaping (anti-under-thinking + overlong, gated behind boxed-authenticity) keeps the on-policy loop off the under-thinking attractor; checkpoint selection uses held-out forced pass@1.

**Tech Stack:** Python 3.11, PyTorch, HuggingFace transformers 5.x, pytest. Qwen3.5-2B. Existing GRAIL verifier + GRPO trainer.

## Global Constraints

- Branch: `feat/cot-2b` (do NOT branch off / merge to main).
- `GRAIL_PROOF_VERSION = "v7"` is already set — any wire-format change to the proof rides this version, do not introduce a new one without coordination.
- Keep inline comments to 1-2 sentences; long rationale goes in commit messages (user preference).
- All repo-bound text (comments, commits) in English.
- `SIGMA_MIN = 0.43`, `BOOTSTRAP_SIGMA_MIN = 0.33`, `MAX_TRUNCATED_PER_SUBMISSION = 1` — already reverted (S1), do not touch.
- `MAX_NEW_TOKENS_PROTOCOL_CAP = 32768` — hard generation backstop; BFT's thinking budget + answer budget must sum to ≤ this.
- `</think>` for Qwen3.5 is the atomic special-token id (the brief cites 248069); confirm against the live tokenizer at implementation time, never hard-code without a resolver.
- Sampler is protocol-fixed: `T_PROTO=0.6`, `TOP_P_PROTO=0.95`, `TOP_K_PROTO=20`, `PRESENCE_PENALTY_PROTO=1.5`.

---

## Roadmap — the prune checklist (6 phases, dependency order)

Each phase is independently testable. Phases 2-6 are scoped here and expanded to step-level TDD detail when reached (per the writing-plans Scope Check, multi-subsystem work is built as a sequence of plans).

| Phase | Subsystem | Brief | Independently shippable? | GPU-blocked? |
|---|---|---|---|---|
| **0** | σ revert + penalty removal | §5.1, §5.2 | ✅ **DONE (S1)** | no |
| **1** | **BFT generation** (miner) | §5.2 | testable; not deployable until Phase 2 | no |
| **2** | **BFT carve-out verification** (validator) | §5.4.2, §5.4.4 | with Phase 1 → end-to-end BFT | no |
| **3** | **Sampler-faithful GRAIL recompute** (validator) | §5.4.1 | ✅ yes — fixes the existing honest-false-reject landmine | **mechanism no / threshold recalibration YES** |
| **4** | **Trainer FORCE-mask + finite-guard** (trainer) | §5.4.3 | with Phase 1 → safe training on forced rollouts | no |
| **5** | **Two-sided reward shaping** (reward/advantage) | §5.3 | yes | no (params want a GPU sweep, §6.3) |
| **6** | **Early-stop** (checkpoint selection) | §5.5 | yes — ops/training | no |

**Sequencing notes:**
- Phase 3 (sampler-faithful recompute) is the brief's "long pole" and is independent of BFT — it also de-risks the *current* branch, where the presence-penalised sampler already ships against a sampler-blind validator. It can be built in parallel with Phase 1.
- Phase 2 depends on Phase 1 (needs the FORCE span to verify) and on Phase 3 (the carve-out's auth exemptions only make sense once the recompute is sampler-faithful).
- Phase 4 depends on Phase 1 (needs the `forced` flag + FORCE span boundaries).
- The two empirical values the brief leaves open — the recompute's 0%-FP thresholds (Phase 3) and the budget B / shaping params (Phases 1, 5) — are implemented **parameterised with a `# GPU-CALIBRATION TODO` marker**; the mechanism is complete, the number is tuned at run.

---

## Phase 1 — BFT generation (miner)

**Files:**
- Modify: `reliquary/constants.py` (add BFT constants near ROLLOUT GENERATION block)
- Modify: `reliquary/shared/modeling.py` (add `force_close_token_ids`, `has_think_close` helpers)
- Modify: `reliquary/miner/engine.py:501-576` (`_generate_m_rollouts` → two-phase generation)
- Test: `tests/unit/test_modeling_helpers.py` (helper tests)
- Test: `tests/unit/test_bft_generation.py` (new — generation logic)

**Interfaces:**
- Produces:
  - `constants.BFT_ENABLED: bool`, `BFT_THINKING_BUDGET: int`, `BFT_ANSWER_BUDGET: int`, `BFT_FORCE_TEMPLATE: str`
  - `modeling.think_close_token_ids(tokenizer) -> list[int]` — the atomic `</think>` id(s), resolver-based.
  - `modeling.force_close_token_ids(tokenizer) -> list[int]` — token ids of `BFT_FORCE_TEMPLATE` (`</think>` atomic id + encoded tail).
  - `modeling.has_think_close(tokens: list[int], think_close_ids: set[int]) -> bool`
  - Each rollout dict from `_generate_m_rollouts` gains `"forced": bool` and (when forced) `"force_span": tuple[int,int]` = (start, end) indices of the injected FORCE ids within `tokens`.

### Task 1: BFT constants

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_constants.py`, add:

```python
def test_bft_constants_present_and_within_cap():
    assert isinstance(C.BFT_ENABLED, bool)
    assert C.BFT_THINKING_BUDGET > 0
    assert C.BFT_ANSWER_BUDGET > 0
    # Forced thinking + forced answer must fit under the hard generation cap.
    assert C.BFT_THINKING_BUDGET + C.BFT_ANSWER_BUDGET <= C.MAX_NEW_TOKENS_PROTOCOL_CAP
    assert C.BFT_FORCE_TEMPLATE.startswith("</think>")
    assert "\\boxed{" in C.BFT_FORCE_TEMPLATE
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_constants.py::test_bft_constants_present_and_within_cap -q`
Expected: FAIL with `AttributeError: module 'reliquary.constants' has no attribute 'BFT_ENABLED'`

- [ ] **Step 3: Add the constants**

In `reliquary/constants.py`, after the `MAX_NEW_TOKENS_PROTOCOL_CAP` block:

```python
# ────────────────  BUDGET-FORCED TERMINATION (BFT)  ────────────────

# Make a thinking model trainable under the strict σ-gate: if a rollout has not
# emitted </think> by BFT_THINKING_BUDGET tokens, the miner appends the FORCE
# template and samples the answer in BFT_ANSWER_BUDGET more tokens, so the
# rollout terminates with a real (gradeable) answer instead of truncating to 0.
BFT_ENABLED = True
# Thinking budget B + answer budget sum to the hard cap. GPU-CALIBRATION TODO:
# the brief experimented at B=2048; production B is tuned to the model's natural
# finish-length so BFT only catches genuine non-finishers.
BFT_THINKING_BUDGET = 24576
BFT_ANSWER_BUDGET = 8192
# Injected verbatim at the budget. The </think> atomic token + a fixed answer
# preamble; the model samples the boxed answer after it.
BFT_FORCE_TEMPLATE = "</think>\n\nFinal Answer: \\boxed{"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_constants.py::test_bft_constants_present_and_within_cap -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reliquary/constants.py tests/unit/test_constants.py
git commit -m "feat(cot): add BFT thinking/answer budget + force-template constants"
```

### Task 2: `</think>` and FORCE token-id helpers + detector

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_modeling_helpers.py`, add:

```python
def test_has_think_close_detects_atomic_id():
    from reliquary.shared.modeling import has_think_close
    # 248069 = atomic </think> in the toy set
    assert has_think_close([1, 2, 248069, 3], {248069}) is True
    assert has_think_close([1, 2, 3], {248069}) is False
    assert has_think_close([], {248069}) is False

def test_think_close_and_force_ids_from_tokenizer():
    from reliquary.shared.modeling import (
        think_close_token_ids, force_close_token_ids,
    )

    class _Tok:
        # minimal stub: </think> is one atomic id, tail encodes to fixed ids
        def convert_tokens_to_ids(self, t):
            return 248069 if t == "</think>" else None
        def encode(self, text, add_special_tokens=False):
            assert add_special_tokens is False
            return [7, 8, 9]  # stand-in for "\n\nFinal Answer: \boxed{"

    assert think_close_token_ids(_Tok()) == [248069]
    # force = </think> atomic id followed by the encoded tail
    assert force_close_token_ids(_Tok()) == [248069, 7, 8, 9]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_modeling_helpers.py -q -k "think_close or force_ids"`
Expected: FAIL with `ImportError: cannot import name 'has_think_close'`

- [ ] **Step 3: Implement the helpers**

In `reliquary/shared/modeling.py`, append:

```python
def think_close_token_ids(tokenizer) -> list[int]:
    """Atomic ``</think>`` token id(s) for this tokenizer. Returns the single
    special-token id when one exists; raises if the tokenizer cannot resolve it
    (we never silently fall back to a split encoding, which the carve-out relies
    on being atomic)."""
    tid = tokenizer.convert_tokens_to_ids("</think>")
    if tid is None or (isinstance(tid, int) and tid < 0):
        raise ValueError("tokenizer has no atomic </think> token")
    return [int(tid)]


def force_close_token_ids(tokenizer) -> list[int]:
    """Token ids of ``BFT_FORCE_TEMPLATE``: the atomic </think> id followed by
    the encoded answer preamble (no special tokens added to the tail)."""
    from reliquary.constants import BFT_FORCE_TEMPLATE

    close_ids = think_close_token_ids(tokenizer)
    tail = BFT_FORCE_TEMPLATE[len("</think>"):]
    tail_ids = list(tokenizer.encode(tail, add_special_tokens=False))
    return close_ids + [int(t) for t in tail_ids]


def has_think_close(tokens: list[int], think_close_ids: set[int]) -> bool:
    """True iff any token is an atomic </think> id."""
    return any(int(t) in think_close_ids for t in tokens)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_modeling_helpers.py -q -k "think_close or force_ids"`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add reliquary/shared/modeling.py tests/unit/test_modeling_helpers.py
git commit -m "feat(cot): tokenizer helpers for </think> + FORCE token ids and detection"
```

### Task 3: Two-phase generation with FORCE injection

**Design:** generate phase 1 up to `BFT_THINKING_BUDGET`; rows that emitted `</think>` are done; rows that did not get the FORCE ids appended and a phase-2 `.generate()` of up to `BFT_ANSWER_BUDGET` tokens; each forced row records its `force_span`.

- [ ] **Step 1: Write the failing test**

In `tests/unit/test_bft_generation.py` (new):

```python
import torch
from reliquary.miner.engine import _bft_assemble_rollouts


class _FakeModel:
    """Phase-1 returns a fixed thinking tensor; phase-2 returns a fixed answer.
    Rows whose phase-1 lacks </think> (id 248069) are the BFT candidates."""
    device = "cpu"

    def __init__(self, phase1, phase2):
        self._phase1 = phase1
        self._phase2 = phase2
        self.calls = 0

    def generate(self, input_tensor, **kw):
        self.calls += 1
        return self._phase1 if self.calls == 1 else self._phase2


def test_bft_injects_force_only_for_unfinished_rows():
    prompt = [1, 1]
    think_close = {248069}
    force_ids = [248069, 7, 8]          # </think> + tail stub
    eos = {99}
    # row0 finished thinking (has 248069) then answered + EOS;
    # row1 never closed </think> within budget → must be forced.
    phase1 = torch.tensor([
        [1, 1, 5, 248069, 42, 99],      # finished
        [1, 1, 5, 6, 7, 8],             # still thinking at budget
    ])
    # phase-2 only runs on row1 (the unfinished one), with force appended:
    phase2_in_row = [1, 1, 5, 6, 7, 8, 248069, 7, 8]
    phase2 = torch.tensor([phase2_in_row + [55, 99]])   # boxed answer + EOS

    rollouts = _bft_assemble_rollouts(
        model=_FakeModel(phase1, phase2),
        prompt_tokens=prompt,
        think_close_ids=think_close,
        force_ids=force_ids,
        eos_ids=eos,
        answer_budget=8,
    )

    assert rollouts[0]["forced"] is False
    assert rollouts[1]["forced"] is True
    # forced row carries the injected span boundaries
    start, end = rollouts[1]["force_span"]
    assert rollouts[1]["tokens"][start:end] == force_ids
    # answer tokens after the force are kept
    assert rollouts[1]["tokens"][end:] == [55, 99]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_bft_generation.py -q`
Expected: FAIL with `ImportError: cannot import name '_bft_assemble_rollouts'`

- [ ] **Step 3: Implement `_bft_assemble_rollouts` and the phase-2 generation**

In `reliquary/miner/engine.py`, add a module-level helper (kept separate from `_generate_m_rollouts` so it is unit-testable without the full miner):

```python
def _bft_assemble_rollouts(
    *, model, prompt_tokens, think_close_ids, force_ids, eos_ids,
    answer_budget,
):
    """Given a completed phase-1 generation on the model, force-close the rows
    that never emitted </think>, run a phase-2 answer generation on just those
    rows, and return per-row rollout dicts with a ``forced`` flag and (for
    forced rows) the ``force_span`` boundaries within ``tokens``."""
    import torch
    from reliquary.shared.modeling import first_eos_index, has_think_close

    # NOTE: phase-1 generation is the caller's responsibility; here we receive
    # the model already primed to return phase-1 then phase-2 (see
    # _generate_m_rollouts). In production the caller passes the phase-1 tensor.
    phase1 = model.generate(None)
    plen = len(prompt_tokens)
    rows = [phase1[i].tolist() for i in range(phase1.shape[0])]

    finished, unfinished_idx, unfinished_primed = [], [], []
    for i, seq in enumerate(rows):
        gen = seq[plen:]
        if has_think_close(gen, set(think_close_ids)):
            fe = first_eos_index(gen, eos_ids)
            gen = gen[: fe + 1] if fe is not None else gen
            finished.append((i, {"tokens": prompt_tokens + gen,
                                 "prompt_length": plen, "forced": False}))
        else:
            primed = seq + list(force_ids)
            unfinished_idx.append(i)
            unfinished_primed.append(primed)

    forced = {}
    if unfinished_primed:
        width = max(len(p) for p in unfinished_primed)
        pad = (eos_ids and min(eos_ids)) or 0
        batch = torch.tensor([[pad] * (width - len(p)) + p
                              for p in unfinished_primed])
        ans = model.generate(batch, max_new_tokens=answer_budget)
        for k, i in enumerate(unfinished_idx):
            full = ans[k].tolist()
            primed = unfinished_primed[k]
            tail = full[len(batch[k]):]  # answer tokens after the primed input
            fe = first_eos_index(tail, eos_ids)
            tail = tail[: fe + 1] if fe is not None else tail
            tokens = primed + tail
            force_start = len(primed) - len(force_ids)
            forced[i] = {"tokens": tokens, "prompt_length": plen,
                         "forced": True,
                         "force_span": (force_start, len(primed))}

    out = [None] * phase1.shape[0]
    for i, r in finished:
        out[i] = r
    for i, r in forced.items():
        out[i] = r
    return out
```

> Note for the implementer: the test primes `_FakeModel` to return phase-1 on
> call 1 and phase-2 on call 2, so the helper calls `model.generate` twice. In
> `_generate_m_rollouts`, phase-1 is the existing batched `.generate()` call;
> refactor so the phase-1 tensor is passed through (e.g. wrap it in a tiny
> adapter exposing the same two-call contract, or split the helper signature to
> take `phase1_tensor` directly). Keep the production wiring change minimal and
> covered by Step 4 below.

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/unit/test_bft_generation.py -q`
Expected: PASS

- [ ] **Step 5: Wire into `_generate_m_rollouts` behind `BFT_ENABLED`**

In `reliquary/miner/engine.py:_generate_m_rollouts`, after the phase-1 `outputs = self.vllm_model.generate(...)` call, branch: when `BFT_ENABLED`, call `_bft_assemble_rollouts` with the phase-1 tensor and the resolved `think_close_ids` / `force_ids`; otherwise keep the current first-EOS-truncation path. Generate phase-1 with `max_new_tokens=min(self.max_new_tokens, BFT_THINKING_BUDGET)`. Add the `forced` key (`False`) to the non-BFT path too so downstream is uniform.

- [ ] **Step 6: Run the miner unit tests**

Run: `python -m pytest tests/unit/test_bft_generation.py tests/unit/test_modeling_helpers.py -q`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add reliquary/miner/engine.py tests/unit/test_bft_generation.py
git commit -m "feat(cot): BFT two-phase generation — force </think>+answer at budget"
```

### Task 4: Propagate `forced` / `force_span` into the submission

- [ ] **Step 1: Write the failing test** — assert `_build_rollout_submission` copies `forced` and `force_span` from the generation dict into `commit["rollout"]` metadata (so the validator carve-out and trainer mask can read them). Use the existing submission-building test fixtures in `tests/unit/test_submitter.py` as the pattern.

- [ ] **Step 2-4:** run-fail → implement the metadata copy in `reliquary/miner/engine.py:_build_rollout_submission` → run-pass.

- [ ] **Step 5: Commit**

```bash
git add reliquary/miner/engine.py tests/unit/test_submitter.py
git commit -m "feat(cot): carry forced/force_span rollout metadata into the commit"
```

---

## Phase 2 — BFT carve-out verification (validator)  *(detail at execution)*

**Files:** `reliquary/validator/verifier.py`, `reliquary/validator/batcher.py`, `tests/unit/test_grpo_window_batcher.py`, new `tests/unit/test_bft_carveout.py`.

**Scope (brief §5.4.2, §5.4.4):**
- Reject a forced rollout whose `force_span` is not **byte-exactly** `force_close_token_ids` (no attacker tokens smuggled into the carve).
- Anchor the carve on the **atomic** `</think>` id; reject a split-tokenised `</think>` (`[510, 26003, 29]`) so the carve cannot be slid.
- **Exempt the FORCE span positions** from the per-token authenticity / distribution checks (their probability is legitimately ~0); the answer tokens after the span are checked normally.
- **Answer-phase termination rule:** a forced rollout terminates on its answer EOS; it does not re-consume `MAX_TRUNCATED_PER_SUBMISSION`.
- Drift-tolerant trigger + bitwise cross-GPU surviving-index test to avoid honest false-rejects.

## Phase 3 — Sampler-faithful GRAIL recompute (validator)  *(detail at execution)*

**Files:** `reliquary/validator/verifier.py` (`_gpu_completion_token_stats`, `_gpu_challenge_logprobs`), new `tests/unit/test_recompute_sampler_faithful.py`.

**Scope (brief §5.4.1 — the long pole):**
- Before the chosen/argmax/logprob computations, apply, per position, the **same sampler the miner used**: subtract `PRESENCE_PENALTY_PROTO` from already-emitted completion tokens' logits, then apply `TOP_K_PROTO` / `TOP_P_PROTO` truncation, then `T_PROTO`.
- **GPU-CALIBRATION TODO:** re-calibrate the auth thresholds (currently `chosen < 1e-5 & argmax ≥ 0.99`) to **0% false-positive** on honest presence-penalised CoT prose — the raw honest floor (~3.5e-7) and prose argmax behaviour shift once the sampler is replayed. Mechanism lands here; the numbers are set from a GPU honest-rollout sweep.
- This phase also de-risks the *current* branch (presence-penalised sampler vs sampler-blind validator → honest false-rejects today).

## Phase 4 — Trainer FORCE-mask + finite-guard (trainer)  *(detail at execution)*

**Files:** the GRPO loss path (`reliquary/validator/` training step), new `tests/unit/test_force_mask_loss.py`.

**Scope (brief §5.4.3):**
- Mask the `force_span` positions out of the token-level loss (= DAPO Overlong Filtering, narrowed to the injected span). Train on thinking-before + answer-after only.
- Finite-guard the carve: `error_if_nonfinite`, bound `|logprob|`, pre-publish finite-check on the checkpoint (the carve otherwise amplifies the NaN/checkpoint-poison chain).

## Phase 5 — Two-sided reward shaping (reward/advantage)  *(detail at execution)*

**Files:** reward/advantage computation in the validator reward path; `reliquary/validator/boxed_integrity.py` (gating); new `tests/unit/test_reward_shaping.py`.

**Scope (brief §5.3) — faithful, two-sided:**
- **Anti-under-thinking:** a rollout that finished early (`len < SHAPE_LEN_FRAC * B`) AND is wrong → advantage gets `-SHAPE_PENALTY`. Forced / correct / tried-hard-wrong rollouts untouched.
- **Overlong (Rom's direction, kept per the brief's two-sided design):** penalise rambling/over-generation.
- Both **gated behind a truncation/boxed-authenticity check** (extend `boxed_integrity.py`) so neither side is gameable via EOS-suppression manufacture.
- Implement in the reward/advantage path, NOT the σ-gate. `SHAPE_PENALTY=0.5`, `SHAPE_LEN_FRAC=0.5` defaults (GPU-CALIBRATION TODO: sweep per §6.3).

## Phase 6 — Early-stop (checkpoint selection)  *(detail at execution)*

**Files:** checkpoint publish/selection path; new eval harness hook.

**Scope (brief §5.5):** select the published checkpoint by held-out **forced pass@1** (force every eval prompt at budget, `grade_forced` only, greedy), not last-step. The on-policy loop must not run open-ended.

---

## Self-Review

- **Spec coverage:** §5.1/§5.2 → Phase 0 (done); §5.2 BFT → Phase 1+2; §5.4.1 → Phase 3; §5.4.2/4 → Phase 2; §5.4.3 → Phase 4; §5.3 → Phase 5; §5.5 → Phase 6. All §5 items mapped.
- **Placeholder scan:** Phase 1 carries complete code/tests. Phases 2-6 are deliberately scope-level (multi-subsystem split per the skill); each is expanded to step-level TDD before it is executed — they are roadmap entries, not placeholder tasks.
- **Type consistency:** `forced: bool` / `force_span: tuple[int,int]` / `force_close_token_ids` / `think_close_token_ids` / `has_think_close` names are used consistently across Phases 1, 2, 4.
- **Open empirical values (flagged, not vague):** `BFT_THINKING_BUDGET`, recompute auth thresholds, `SHAPE_PENALTY`/`SHAPE_LEN_FRAC` — each parameterised with a GPU-CALIBRATION TODO.
