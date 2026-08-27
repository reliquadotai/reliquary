# Qwen3-4B-Base + DAPO Protocol Profile (v4) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new protocol profile `qwen3-4b-base-dapo-v4` that restarts training from the true base model `Qwen/Qwen3-4B-Base` with a maximally DAPO-faithful recipe: raw-completion DAPO prompt (no chat template, no `<think>`, no BFT), protocol sampling at T=1.0/top_p=1.0/top_k off (which makes `warp()` the identity and dissolves the ratio-space bug), clip-higher ε 0.2/0.28, KL β=0, π_old recomputed by the validator, and the OMI corpus restricted to the 32 canonical `train-*` shards.

**Architecture:** Everything is keyed off the new profile (`PROTOCOL_VERSION >= 4` derivations in `reliquary/constants.py`), so all changes are inert for the live v3 deployment until `RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-v4` is set. The only unconditional changes are two grader-extraction fixes that remove documented false negatives. No wire schema changes: `SamplingProfile` already carries per-profile sampling, BFT turns off via `bft=None`, and the runtime-fingerprint schema is left untouched (its `qwen35_*` flags report kernel importability, which stays consensus-neutral under qwen3).

**Tech Stack:** Python 3.11, PyTorch, transformers (Qwen3 = plain `AutoModelForCausalLM` path, already pinned by `tests/unit/test_modeling_helpers.py:58-62`), pytest.

## Global Constraints

- Branch from **`origin/main` (cff084c)**, NOT from the current checkout — HEAD (2a48c49, branch `feat/proof-path-qualification-carryover`) is 3 commits behind and lacks PR #167 (soft overlong punishment), which this plan builds on. Do not touch the uncommitted changes in the current working tree; use a fresh worktree.
- Never push to `main`; feature branch only, push only when explicitly asked (repo rule).
- All repo artifacts (code, comments, commit messages, this plan's outputs) in English.
- Model pin: `Qwen/Qwen3-4B-Base` @ `906bfd4b4dc7f14ee4320094d8b41684abff8539` (the exact revision measured in the 2026-08-03 headroom study).
- v4 sampling: `temperature=1.0`, `top_p=1.0`, `top_k=0`, `rollouts=8`, `do_sample=False`. `top_k=0` is the disable sentinel that is safe both in `warp()` (`if top_k and top_k > 0`) and in `ForcedSeedLogitsProcessor.__init__`'s `int(top_k)` coercion (`None` crashes there; do not use `None`).
- v4 trainer knobs: `LEARNING_RATE=1e-6` (inherited from the `>= 3` derivation), `KL_BETA=0.0`, `PPO_CLIP_EPSILON_LOW=0.2` / `PPO_CLIP_EPSILON_HIGH=0.28`, `RECOMPUTE_PI_OLD_FROM_VERIFY` default true, `OVERLONG_PENALTY_FACTOR=0.5` + `OVERLONG_PENALTY_CACHE_TOKENS=4096` (inherited, already live via PR #167), all `MASK_*_FROM_LOSS` false (already the main default).
- Reward stays `[0,1]` — group normalization is affine-invariant, so this is gradient-identical to DAPO's `{−1,+1}` (divergence audit §10). Do not rescale.
- New constants that gate miner-visible behavior must be derived from the profile (no env var), matching the convention enforced by `tests/unit/test_miner_facing_constants_are_declared.py`.
- Constants consumed by functions under test should be imported **lazily inside the function body** so tests can `monkeypatch.setattr(reliquary.constants, ...)` (existing convention, see `training.py:187,226,266`). Module-top imports require patching the *consuming* module instead — avoid adding new ones.
- Test command: `python -m pytest tests/unit/<file> -x -q` (no GPU needed for any test in this plan).

## Not in this plan (explicitly deferred)

- **Offline confirmatory DAPO run** (reliquary-experiments repo, separate effort). It is the recommended *deploy gate*: the code below can be built and merged first, but do not cut production over until one offline run shows the recipe converts Q3B's 0.361 pass-gap into actual gains.
- `SIGMA_MIN` relaxation toward DAPO's `0 < k < 8` filter, auction value-function δ review, and `B_BATCH` 8→16. All consensus-affecting; they form the post-cutover auction workstream and must be re-decided **before** miner supply ramps up (k=2 concentration risk).
- Row-level integer-answer filtering of OMI (DAPO §3.5). The manifest layer is footer-only by design; a row-level predicate needs a precomputed index artifact. Revisit after cutover.
- `G` (rollouts/prompt) 8→16: wire-affecting, separate decision.
- The deployment itself (env changes, miner coordination). See the checklist at the end.

---

### Task 0: Workspace setup

**Files:** none (git only)

- [ ] **Step 1: Fetch and branch from origin/main**

```bash
cd /home/ubuntu/Catalyst
git fetch origin
git worktree add ../Catalyst-v4 -b feat/qwen3-base-dapo-v4-profile origin/main
cd ../Catalyst-v4
```

All subsequent tasks run in `../Catalyst-v4`.

- [ ] **Step 2: Verify the baseline is green where we will work**

Run: `python -m pytest tests/unit/test_protocol_profiles.py tests/unit/test_training_overlong_penalty.py tests/unit/test_tokens.py tests/unit/test_openmathinstruct_environment.py tests/unit/test_virtual_parquet.py tests/unit/test_forced_sampling.py -q`
Expected: all pass (confirms PR #167 files are present — `test_training_overlong_penalty.py` only exists on main).

---

### Task 1: Grader extraction fixes (unconditional)

Two documented false-negative sources, measured in the headroom study (3/32 verdicts flipped in one cell): `_normalize_answer` does not strip inline LaTeX delimiters, and there is no `Answer:`-line extraction at all. These are pure acceptance-widening fixes: they only convert wrong-scored-correct answers to correct, never the reverse. They also make the production grader able to grade v4's `Answer:`-format completions. Production's semantic comparison (`_answers_equal`, sympy path) is kept — do **not** port the analysis scripts' weaker string-equality grading.

**Files:**
- Modify: `reliquary/environment/openmathinstruct.py` (`_normalize_answer` at ~74-100, `_compute_omi_reward` at ~371-395)
- Test: `tests/unit/test_openmathinstruct_environment.py`

**Interfaces:**
- Produces: `_compute_omi_reward(completion: str, problem: dict) -> float` (unchanged signature); module-level `_ANSWER_LINE_RE` regex. Task 5's prompt template relies on `Answer:` lines being gradable.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_openmathinstruct_environment.py`:

```python
from reliquary.environment.openmathinstruct import _compute_omi_reward, _normalize_answer


def test_normalize_strips_inline_latex_delimiters():
    assert _normalize_answer(r"\(\frac{14}{3}\)") == _normalize_answer(r"\frac{14}{3}")
    assert _normalize_answer(r"\[p - q\]") == _normalize_answer("p - q")


def test_answer_line_extraction_plain():
    problem = {"ground_truth": "42"}
    assert _compute_omi_reward("Step 1: compute.\nAnswer: 42", problem) == 1.0


def test_answer_line_extraction_markdown_emphasis():
    problem = {"ground_truth": "0"}
    assert _compute_omi_reward("reasoning...\n**Answer:** 0", problem) == 1.0


def test_answer_line_takes_last_occurrence():
    problem = {"ground_truth": "7"}
    text = "Answer: 3\nWait, recompute.\nAnswer: 7"
    assert _compute_omi_reward(text, problem) == 1.0


def test_boxed_still_wins_over_answer_line():
    problem = {"ground_truth": "5"}
    text = "Answer: 3\nActually \\boxed{5}"
    assert _compute_omi_reward(text, problem) == 1.0


def test_no_answer_still_zero():
    assert _compute_omi_reward("I cannot solve this.", {"ground_truth": "1"}) == 0.0
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `python -m pytest tests/unit/test_openmathinstruct_environment.py -q`
Expected: FAIL — `test_normalize_strips_inline_latex_delimiters`, `test_answer_line_*` fail; the boxed and no-answer tests may already pass.

- [ ] **Step 3: Implement**

In `reliquary/environment/openmathinstruct.py`, add after the `_MBOX_RE` definition (~line 71):

```python
# "Answer: X" line extraction, tolerating markdown emphasis around the cue
# ("**Answer:** 0"). v4 prompts instruct this format; post-trained models also
# emit it spontaneously. Boxed extraction still takes precedence.
_ANSWER_LINE_RE = re.compile(r"^[\s*_]*Answer[\s*_]*:[\s*_]*(.+?)\s*$", re.MULTILINE)
```

In `_normalize_answer`, insert immediately after `s = str(s)` (line ~77):

```python
    # Inline/display LaTeX delimiters: models emit "\(x\)" / "\[x\]" around
    # answers when not forced into \boxed{}; they never carry meaning.
    for delim in (r"\(", r"\)", r"\[", r"\]"):
        s = s.replace(delim, "")
```

In `_compute_omi_reward`, replace the extraction block (currently: `boxed = ...` / `if boxed is None:` tail-number fallback / `else:` boxed path) with:

```python
        boxed = _last_boxed_only_string(completion)
        if boxed is not None:
            candidate = _normalize_answer(_strip_boxed_wrapper(boxed))
        else:
            answer_lines = _ANSWER_LINE_RE.findall(completion)
            if answer_lines:
                candidate = _normalize_answer(answer_lines[-1])
            else:
                # Fallback: trailing number/fraction at end of text
                tail = completion.strip().split("\n")[-1].strip()
                m = re.match(r"^([\-\+]?\d+(?:\.\d+)?(?:/\d+)?)", tail)
                if m is None:
                    return 0.0
                candidate = _normalize_answer(m.group(1))
```

- [ ] **Step 4: Run the full file's tests**

Run: `python -m pytest tests/unit/test_openmathinstruct_environment.py -q`
Expected: PASS (including all pre-existing tests — the extraction change must not break boxed-path tests).

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/openmathinstruct.py tests/unit/test_openmathinstruct_environment.py
git commit -m "fix(grader): strip inline LaTeX delimiters and extract Answer: lines

Two measured false-negative sources (3/32 verdicts flipped in the
2026-08-03 headroom study). Also the grading prerequisite for the v4
raw-completion prompt format.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 2: v4 protocol profile + version-keyed constants

**Files:**
- Modify: `reliquary/protocol/profiles.py` (add `_SAMPLING_DAPO`, v4 entry in `_PROFILE_VALUES`)
- Modify: `reliquary/constants.py` (`_PROFILE_KL_BETA`, `RECOMPUTE_PI_OLD_FROM_VERIFY` default, new `RAW_COMPLETION_PROMPTS`, new `OMI_TRAIN_SHARDS_ONLY`)
- Test: `tests/unit/test_protocol_profiles.py`

**Interfaces:**
- Produces: profile id `"qwen3-4b-base-dapo-v4"`; constants `RAW_COMPLETION_PROMPTS: bool` and `OMI_TRAIN_SHARDS_ONLY: bool` (both `PROTOCOL_VERSION >= 4`). Tasks 4, 5, 6 consume these two constants; Task 7 consumes `PROTOCOL_VERSION` for the clip pair.

- [ ] **Step 1: Write the failing test**

Extend the subprocess knob-lock test in `tests/unit/test_protocol_profiles.py` (follow its existing convention exactly: it launches a subprocess with `RELIQUARY_PROTOCOL_PROFILE` set, prints resolved constants, and asserts against an `expected` dict — see lines ~179-243). Add a v4 case asserting:

| knob | expected v4 value |
|---|---|
| `PROTOCOL_VERSION` | `4` |
| `PROTOCOL_MODEL_ID` | `"Qwen/Qwen3-4B-Base"` |
| `PROTOCOL_MODEL_REVISION` | `"906bfd4b4dc7f14ee4320094d8b41684abff8539"` |
| `T_PROTO` / `TOP_P_PROTO` / `TOP_K_PROTO` | `1.0` / `1.0` / `0` |
| math cap / code cap | `16384` / `16384` |
| `BFT_ENABLED` | `False` |
| `BFT_THINKING_BUDGET` / `BFT_ANSWER_BUDGET` | `0` / `0` |
| `WINDOW_COLLECTION_SECONDS` | `300.0` |
| `FORCED_SEED_DOMAIN` | `"reliquary-forced-seed-v4"` |
| `LEARNING_RATE` | `1e-6` |
| `KL_BETA` | `0.0` |
| `MASK_MATH_FORCED_FROM_LOSS` | `False` |
| `OVERLONG_PENALTY_FACTOR` | `0.5` |
| `TRAIN_FORCED_REWARD_ZERO` | `True` |
| `RECOMPUTE_PI_OLD_FROM_VERIFY` | `True` |
| `RAW_COMPLETION_PROMPTS` | `True` |
| `OMI_TRAIN_SHARDS_ONLY` | `True` |

Also add to the same test (or the module's non-subprocess tests) an assertion that the **v3** profile still resolves `RAW_COMPLETION_PROMPTS=False`, `OMI_TRAIN_SHARDS_ONLY=False`, `RECOMPUTE_PI_OLD_FROM_VERIFY=False`, `KL_BETA=0.01` — the plan must not perturb the live profile.

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/unit/test_protocol_profiles.py -q`
Expected: FAIL with `ValueError: unknown protocol profile 'qwen3-4b-base-dapo-v4'` (fail-closed resolution in `resolve_protocol_profile`).

- [ ] **Step 3: Add the profile**

In `reliquary/protocol/profiles.py`, after `_SAMPLING` (~line 147):

```python
# DAPO/verl reference sampling (temperature 1.0, full support). top_k=0 is the
# disable sentinel accepted by both warp() and the miner's logits processor
# (None would crash int() coercion there). At these values warp() is the
# identity softmax, so the PPO ratio space and the sampling space coincide.
_SAMPLING_DAPO = SamplingProfile(
    rollouts=8,
    temperature=1.0,
    top_p=1.0,
    top_k=0,
    do_sample=False,
)
```

Append to `_PROFILE_VALUES` after the v3 entry:

```python
    ProtocolProfile(
        profile_id="qwen3-4b-base-dapo-v4",
        model_id="Qwen/Qwen3-4B-Base",
        model_revision="906bfd4b4dc7f14ee4320094d8b41684abff8539",
        protocol_version=4,
        collection_seconds=300,
        upload_grace_seconds=33,
        sampling=_SAMPLING_DAPO,
        environments={
            # No BFT: the base model emits no <think>; termination is trained
            # by the soft overlong punishment, not forced by a budget.
            "openmathinstruct": EnvironmentProfile(
                max_new_tokens=16384,
                bft=None,
            ),
            "opencodeinstruct": EnvironmentProfile(
                max_new_tokens=16384,
                bft=None,
            ),
        },
        throughput_tiebreak=ThroughputTiebreakProfile(
            token_cap=16384,
            bucket_tokens_per_round=50,
        ),
    ),
```

- [ ] **Step 4: Add the version-keyed constants**

In `reliquary/constants.py`:

(a) Replace the `_PROFILE_KL_BETA` line:

```python
# v4 (DAPO recipe) removes the KL term; v3 kept 0.01 against the rolling
# reference. β=0 leaves the circuit breakers as the collapse guard; the
# calibrated fallback is a pinned KL_BASE_MODEL + explicit beta, not the
# rolling reference (which anchors nothing over horizons > 4 windows).
_PROFILE_KL_BETA = (
    "0.0" if PROTOCOL_VERSION >= 4
    else "0.01" if PROTOCOL_VERSION >= 3
    else "0.04"
)
```

(b) Replace the `RECOMPUTE_PI_OLD_FROM_VERIFY` default:

```python
RECOMPUTE_PI_OLD_FROM_VERIFY = _os.environ.get(
    "RELIQUARY_RECOMPUTE_PI_OLD_FROM_VERIFY",
    "true" if PROTOCOL_VERSION >= 4 else "false",
).strip().lower() in ("1", "true", "yes", "on")
```

Note: with the rolling reference, `ref_model is verify_model is behavior_model`, and `training.py:1183-1184` aliases the forward (`if behavior_model is ref_model: behavior_lp = ref_lp`) — so recomputing π_old costs **zero** extra forwards in the default v4 configuration, and the KL forward we keep paying at β=0 is exactly the π_old forward. No skip logic needed.

(c) Add near the BFT/prompt constants block (~after line 200):

```python
# v4+ ships a raw-completion protocol: prompts are encoded without any chat
# template and carry the DAPO answer-line instruction instead of \boxed{}.
# One switch drives encode_prompt, render_canonical_prompt, and the OMI
# prompt template, so miner and validator can never disagree on prompt bytes.
RAW_COMPLETION_PROMPTS = PROTOCOL_VERSION >= 4

# v4+ restricts the OMI manifest to the canonical `train-*` shards. The
# train_1M/2M/5M subset shards duplicate 8,000,000 rows of `train` (36.4% of
# the index space, drawn 2-4x too often). Changing this changes len(env) and
# the prompt-index consensus, so it is only safe at a profile cutover.
OMI_TRAIN_SHARDS_ONLY = PROTOCOL_VERSION >= 4
```

If `tests/unit/test_miner_facing_constants_are_declared.py` maintains an explicit inventory of declared miner-facing constants, add `RAW_COMPLETION_PROMPTS` to it (read that test's mechanism first; follow its convention).

- [ ] **Step 5: Run the tests**

Run: `python -m pytest tests/unit/test_protocol_profiles.py tests/unit/test_constants.py tests/unit/test_miner_facing_constants_are_declared.py -q`
Expected: PASS. (`test_constants.py`'s BFT>0 assertions run under the default v2 profile, so they still hold; the import-time assert at `constants.py:187-193` passes for v4 since `16384 > 0 + 0`.)

- [ ] **Step 6: Commit**

```bash
git add reliquary/protocol/profiles.py reliquary/constants.py tests/unit/test_protocol_profiles.py tests/unit/test_miner_facing_constants_are_declared.py
git commit -m "feat(protocol): add qwen3-4b-base-dapo-v4 profile

Qwen/Qwen3-4B-Base pinned, DAPO sampling (T=1.0, full support), BFT off,
KL beta 0, pi_old recomputed by default, raw-completion prompt switch,
OMI train-shards-only switch. Inert unless the profile is selected.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 3: Pin warp() identity at v4 sampling values

No production code change — this test documents and locks the property the whole cutover leans on: at v4 sampling values the forced-seed warp is the identity softmax, so the importance ratio computed on raw log-probs is computed in the distribution the samples actually came from (dissolves divergence D1).

**Files:**
- Test: `tests/unit/test_forced_sampling.py`

**Interfaces:**
- Consumes: `warp(logits, t, top_k, top_p)` from `reliquary/environment/forced_sampling.py`.

- [ ] **Step 1: Write the test**

Append to `tests/unit/test_forced_sampling.py`:

```python
import torch

from reliquary.environment.forced_sampling import warp


def test_warp_is_identity_softmax_at_dapo_values():
    """v4 samples at T=1.0, top_p=1.0, top_k=0: warp must be plain softmax.

    This is the property that makes the PPO ratio space equal the sampling
    space (no r^(1/T) distortion, no support truncation). If warp() ever
    changes shape, this test must fail before the trainer silently drifts.
    """
    torch.manual_seed(0)
    logits = torch.randn(1000, dtype=torch.float64) * 5.0
    out = warp(logits, t=1.0, top_k=0, top_p=1.0)
    expected = torch.softmax(logits.float(), dim=-1)
    assert torch.allclose(out, expected, atol=1e-6)
    assert torch.all(out > 0)  # full support: nothing masked to zero


def test_warp_top_k_zero_and_minus_one_both_disable():
    logits = torch.randn(50)
    assert torch.allclose(
        warp(logits, t=1.0, top_k=0, top_p=1.0),
        warp(logits, t=1.0, top_k=-1, top_p=1.0),
    )
```

- [ ] **Step 2: Run**

Run: `python -m pytest tests/unit/test_forced_sampling.py -q`
Expected: PASS immediately (the property already holds; the test is a regression pin, not a change driver). If it fails, STOP — the D1-dissolution premise is wrong and the plan's sampling decision must be revisited.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_forced_sampling.py
git commit -m "test(forced-sampling): pin warp identity at v4 sampling values

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 4: Raw prompt encoding (encode_prompt + canonical renderer)

`encode_prompt` currently branches purely on `tokenizer.chat_template` — and Qwen3-4B-Base **does declare a chat template**, so without this task the v4 prompt would silently go through the chat path. The same branch is duplicated in `render_canonical_prompt` (feeds `prompt_content_sha256`); both must switch together or the content-cooldown hash and the encoded tokens disagree.

**Files:**
- Modify: `reliquary/protocol/tokens.py` (`encode_prompt`, ~lines 34-70)
- Modify: `reliquary/validator/prompt_content.py` (`render_canonical_prompt`, ~lines 37-56)
- Test: `tests/unit/test_tokens.py`, `tests/unit/test_prompt_content.py`

**Interfaces:**
- Consumes: `RAW_COMPLETION_PROMPTS` from Task 2.
- Produces: `encode_prompt(tokenizer, prompt_text) -> list[int]` returning `tokenizer.encode(prompt_text, add_special_tokens=False)` whenever `RAW_COMPLETION_PROMPTS` is true, regardless of the tokenizer's chat template; `render_canonical_prompt(prompt_text, tokenizer) -> str` returning `prompt_text` unchanged in that mode.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_tokens.py` (mirror the file's existing fake-tokenizer style — it already has fakes with a `chat_template` attribute, see its existing chat-path tests):

```python
def test_raw_completion_mode_bypasses_chat_template(monkeypatch):
    import reliquary.constants as C
    from reliquary.protocol.tokens import encode_prompt

    monkeypatch.setattr(C, "RAW_COMPLETION_PROMPTS", True)

    class _Tok:
        chat_template = "{% for m in messages %}...{% endfor %}"

        def apply_chat_template(self, *a, **k):
            raise AssertionError("chat template must not be used in raw mode")

        def encode(self, text, add_special_tokens=True):
            assert add_special_tokens is False
            return [101, 102, 103]

    assert encode_prompt(_Tok(), "Solve: 1+1") == [101, 102, 103]
```

Append to `tests/unit/test_prompt_content.py` the mirror test:

```python
def test_render_canonical_prompt_raw_mode(monkeypatch):
    import reliquary.constants as C
    from reliquary.validator.prompt_content import render_canonical_prompt

    monkeypatch.setattr(C, "RAW_COMPLETION_PROMPTS", True)

    class _Tok:
        chat_template = "{% for m in messages %}...{% endfor %}"

        def apply_chat_template(self, *a, **k):
            raise AssertionError("chat template must not be used in raw mode")

    assert render_canonical_prompt("Solve: 1+1", _Tok()) == "Solve: 1+1"
```

(Adjust `render_canonical_prompt`'s argument order to match the actual signature in the file — read it first; the test must call it the way production does.)

- [ ] **Step 2: Run to verify failures**

Run: `python -m pytest tests/unit/test_tokens.py tests/unit/test_prompt_content.py -q`
Expected: the two new tests FAIL with `AssertionError: chat template must not be used in raw mode`.

- [ ] **Step 3: Implement**

In `reliquary/protocol/tokens.py`, at the top of `encode_prompt`'s body (before the `chat_template = getattr(...)` line), add:

```python
    # v4+ raw-completion protocol: never apply a chat template, even when the
    # tokenizer declares one (Qwen3-4B-Base ships a template we must ignore).
    # Lazy import so tests can monkeypatch reliquary.constants.
    from reliquary.constants import RAW_COMPLETION_PROMPTS

    if RAW_COMPLETION_PROMPTS:
        return list(tokenizer.encode(prompt_text, add_special_tokens=False))
```

In `reliquary/validator/prompt_content.py`, add the mirror guard at the top of `render_canonical_prompt`'s body:

```python
    from reliquary.constants import RAW_COMPLETION_PROMPTS

    if RAW_COMPLETION_PROMPTS:
        return prompt_text
```

- [ ] **Step 4: Run the two files' full suites**

Run: `python -m pytest tests/unit/test_tokens.py tests/unit/test_prompt_content.py -q`
Expected: PASS (existing chat-path tests must still pass — the guard only fires when the constant is true, and it defaults false under the test-default v2 profile).

- [ ] **Step 5: Commit**

```bash
git add reliquary/protocol/tokens.py reliquary/validator/prompt_content.py tests/unit/test_tokens.py tests/unit/test_prompt_content.py
git commit -m "feat(protocol): raw-completion prompt encoding for v4 profiles

encode_prompt and render_canonical_prompt bypass the chat template when
RAW_COMPLETION_PROMPTS is set; Qwen3-4B-Base declares a template that
must be ignored. Both sites switch on one constant so tokens and the
prompt content hash cannot disagree.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 5: DAPO prompt template for OMI

Replace the `\boxed{}` suffix with the exact DAPO prompt measured in the headroom study (verbatim from `.r2_analysis/headroom/pilot.py:30-36`) when `RAW_COMPLETION_PROMPTS` is set. `str.format` is safe here: braces inside the substituted *value* (LaTeX in problems) are not re-processed; only the template's own braces matter.

**Files:**
- Modify: `reliquary/environment/openmathinstruct.py` (`_ANSWER_FORMAT_INSTRUCTION` block ~line 402-406, `get_problem` ~line 480)
- Test: `tests/unit/test_openmathinstruct_environment.py`

**Interfaces:**
- Consumes: `RAW_COMPLETION_PROMPTS` (Task 2); `Answer:`-line grading (Task 1).
- Produces: `get_problem(...)["prompt"]` rendered via `_render_prompt(question: str) -> str`.

- [ ] **Step 1: Write the failing test**

```python
def test_prompt_uses_dapo_template_in_raw_mode(monkeypatch):
    import reliquary.constants as C
    from reliquary.environment.openmathinstruct import _render_prompt

    monkeypatch.setattr(C, "RAW_COMPLETION_PROMPTS", True)
    p = _render_prompt("What is 2+2?")
    assert p.startswith("Solve the following math problem step by step.")
    assert "What is 2+2?" in p
    assert p.endswith('Remember to put your answer on its own line after "Answer:".')
    assert "\\boxed" not in p


def test_prompt_latex_braces_survive_template(monkeypatch):
    import reliquary.constants as C
    from reliquary.environment.openmathinstruct import _render_prompt

    monkeypatch.setattr(C, "RAW_COMPLETION_PROMPTS", True)
    q = r"Evaluate $\frac{1}{2} + \{x\}$."
    assert q in _render_prompt(q)


def test_prompt_keeps_boxed_suffix_in_legacy_mode(monkeypatch):
    import reliquary.constants as C
    from reliquary.environment.openmathinstruct import _render_prompt

    monkeypatch.setattr(C, "RAW_COMPLETION_PROMPTS", False)
    assert _render_prompt("Q").endswith("Put your final answer within \\boxed{}.")
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_openmathinstruct_environment.py -q`
Expected: FAIL — `ImportError: cannot import name '_render_prompt'`.

- [ ] **Step 3: Implement**

In `reliquary/environment/openmathinstruct.py`, next to `_ANSWER_FORMAT_INSTRUCTION`:

```python
# DAPO prompt (arXiv 2503.14476 style), verbatim from the 2026-08-03 headroom
# study cells. Used for v4+ raw-completion profiles: plain-text reasoning
# terminated by an "Answer:" line, no \boxed{}, no <think>. str.format only
# interprets braces in this template, never in the substituted problem text.
_DAPO_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. The last line of your\n"
    "response should be of the form Answer: $Answer (without quotes) where\n"
    "$Answer is the answer to the problem.\n\n"
    "{problem}\n\n"
    'Remember to put your answer on its own line after "Answer:".'
)


def _render_prompt(question: str) -> str:
    """Profile-appropriate prompt text. Shared by miner and validator via
    get_problem, so both encode identical bytes (GRAIL consensus)."""
    from reliquary.constants import RAW_COMPLETION_PROMPTS

    if RAW_COMPLETION_PROMPTS:
        return _DAPO_PROMPT_TEMPLATE.format(problem=question)
    return question + _ANSWER_FORMAT_INSTRUCTION
```

In `get_problem`, change `"prompt": question + _ANSWER_FORMAT_INSTRUCTION,` to `"prompt": _render_prompt(question),`.

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_openmathinstruct_environment.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/openmathinstruct.py tests/unit/test_openmathinstruct_environment.py
git commit -m "feat(omi): DAPO prompt template for raw-completion profiles

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 6: OMI corpus filter — canonical train shards only

Drop the `train_1M/2M/5M` subset shards (8,000,000 duplicate rows, 36.4% of index space). The filter must be applied in **both** listing paths — the remote `fs.ls` path in `_ensure_manifest` and the offline `_local_manifest_files` path — or the two silently disagree on `len(env)`, which is consensus-critical (`prompt_range.py:26-28`: "both sides MUST pass the same value"). The canonical shards are named `train-XXXXX-of-XXXXX.parquet` (hyphen); the subsets are `train_1M-...` etc. (underscore), so a `"train-"` basename prefix separates them exactly.

**Files:**
- Modify: `reliquary/environment/virtual_parquet.py` (constructor ~line 45, `_ensure_manifest` ~line 274-278, `_local_manifest_files` ~lines 143-168)
- Modify: `reliquary/environment/openmathinstruct.py` (dataset construction, ~line 422)
- Test: `tests/unit/test_virtual_parquet.py`

**Interfaces:**
- Consumes: `OMI_TRAIN_SHARDS_ONLY` (Task 2).
- Produces: `VirtualParquetDataset(repo, revision, *, columns=..., filename_prefix: str | None = None)` — when set, only parquet files whose **basename** starts with the prefix enter the manifest, in both listing paths, without perturbing sort order.

- [ ] **Step 1: Write the failing tests**

`tests/unit/test_virtual_parquet.py` already has the fixtures to reuse: `_make_parquet(path, values, rg_size)` writes a real tiny parquet, and `_LocalFS(files)` is an fsspec shim whose `ls` returns the given paths (the dataset constructor accepts `fs=`). Append:

```python
def _make_shard_layout(tmp_path):
    """Two canonical train- shards + two subset shards (the OMI dup pattern)."""
    _make_parquet(tmp_path / "train-00000-of-00002.parquet", [0, 1, 2], rg_size=2)
    _make_parquet(tmp_path / "train-00001-of-00002.parquet", [3, 4], rg_size=2)
    _make_parquet(tmp_path / "train_1M-00000-of-00001.parquet", [90, 91], rg_size=2)
    _make_parquet(tmp_path / "train_5M-00000-of-00001.parquet", [95], rg_size=2)
    return _LocalFS([
        tmp_path / "train-00000-of-00002.parquet",
        tmp_path / "train-00001-of-00002.parquet",
        tmp_path / "train_1M-00000-of-00001.parquet",
        tmp_path / "train_5M-00000-of-00001.parquet",
    ])


def test_filename_prefix_filters_remote_listing(tmp_path):
    fs = _make_shard_layout(tmp_path)
    ds = VirtualParquetDataset(
        "owner/repo", "rev", columns=["v"], fs=fs, filename_prefix="train-",
    )
    assert len(ds) == 5  # subset shards (underscore names) excluded
    assert [ds.get_row(i)["v"] for i in range(5)] == [0, 1, 2, 3, 4]


def test_filename_prefix_none_keeps_everything(tmp_path):
    fs = _make_shard_layout(tmp_path)
    ds = VirtualParquetDataset("owner/repo", "rev", columns=["v"], fs=fs)
    assert len(ds) == 8  # default unchanged: all four files in the manifest


def test_filename_prefix_filters_local_fallback(tmp_path, monkeypatch):
    import reliquary.environment.virtual_parquet as VP

    data_root = tmp_path / "data"
    data_root.mkdir()
    _make_parquet(data_root / "train-00000-of-00002.parquet", [0, 1], rg_size=2)
    _make_parquet(data_root / "train-00001-of-00002.parquet", [2, 3], rg_size=2)
    _make_parquet(data_root / "train_1M-00000-of-00001.parquet", [90], rg_size=2)

    class _FailingFS:
        def ls(self, base, detail=False):
            raise OSError("offline")

        def open(self, path):
            return open(path, "rb")

    # Route the offline fallback's snapshot_download to tmp_path. Adapt the
    # monkeypatch target to how _local_manifest_files actually imports it
    # (read virtual_parquet.py lines ~143-168 first: patch the name in the
    # module namespace it is resolved from).
    monkeypatch.setattr(VP, "snapshot_download", lambda **kwargs: str(tmp_path))
    ds = VP.VirtualParquetDataset(
        "owner/repo", "rev", columns=["v"],
        fs=_FailingFS(), filename_prefix="train-",
    )
    assert len(ds) == 4  # the train_1M shard never enters the manifest
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_virtual_parquet.py -q`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'filename_prefix'`.

- [ ] **Step 3: Implement**

In `VirtualParquetDataset.__init__`, add the keyword-only param and store it:

```python
        filename_prefix: str | None = None,
```
```python
        self._filename_prefix = filename_prefix
```

Add a small predicate method:

```python
    def _shard_included(self, path: str) -> bool:
        if self._filename_prefix is None:
            return True
        name = str(path).replace("\\", "/").rsplit("/", 1)[-1]
        return name.startswith(self._filename_prefix)
```

In `_ensure_manifest`, extend the listing comprehension:

```python
                files = sorted(
                    str(p)
                    for p in fs.ls(base, detail=False)
                    if str(p).endswith(".parquet") and self._shard_included(str(p))
                )
```

In `_local_manifest_files`, restrict both the download pattern and the glob:

```python
            allow_patterns=[
                f"{self._data_dir}/{self._filename_prefix or ''}*.parquet"
            ],
```
```python
        local_files = sorted(
            path
            for path in data_root.glob("*.parquet")
            if path.is_file() and self._shard_included(str(path))
        )
```

In `reliquary/environment/openmathinstruct.py` (~line 422):

```python
    from reliquary.constants import OMI_TRAIN_SHARDS_ONLY

    return VirtualParquetDataset(
        repo,
        revision,
        columns=["problem", "expected_answer"],
        # v4+: canonical corpus only. The train_1M/2M/5M shards are curated
        # subsets OF train; including them duplicates 8M rows (36.4% of the
        # index space). len(env) changes with this flag: cutover-only.
        filename_prefix="train-" if OMI_TRAIN_SHARDS_ONLY else None,
    )
```

(Match the actual construction-site shape — read the surrounding function first; if the import placement conflicts with the `_dataset_cache` keying, keep the cache key as `(repo, revision)` — the flag is constant for a process lifetime, so it cannot flip within one cache.)

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_virtual_parquet.py tests/unit/test_openmathinstruct_environment.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/environment/virtual_parquet.py reliquary/environment/openmathinstruct.py tests/unit/test_virtual_parquet.py
git commit -m "feat(omi): restrict manifest to canonical train-* shards on v4

Drops the train_1M/2M/5M subset shards (8M duplicate rows, 36.4% of the
index space, drawn 2-4x too often). Applied identically in the remote
listing and the local fallback so len(env) cannot fork. Profile-gated:
len(env) is prompt-range consensus.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 7: Clip-higher (ε_low 0.2 / ε_high 0.28, v4-keyed)

Replace the single symmetric `PPO_CLIP_EPSILON` with a low/high pair. For v2/v3 both are 0.2 (behavior identical); for v4+, high is 0.28 (DAPO §3.1). The values are interpretable on v4 precisely because Task 3's warp-identity holds. Three consuming sites in `training.py` (two clamp sites + the telemetry counters) and the config echo in `service.py`.

**Files:**
- Modify: `reliquary/constants.py` (~line 830)
- Modify: `reliquary/validator/training.py` (import ~line 27; `_rollout_loss` clamp; `_microbatch_grad` clamp; `_record_kl_stats` counters)
- Modify: `reliquary/validator/service.py` (~line 443 config echo)
- Test: create `tests/unit/test_ppo_clip_asymmetric.py`

**Interfaces:**
- Produces: constants `PPO_CLIP_EPSILON_LOW: float`, `PPO_CLIP_EPSILON_HIGH: float`. The name `PPO_CLIP_EPSILON` is **removed** — a final grep must show zero remaining references.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_ppo_clip_asymmetric.py`:

```python
"""Clip-higher: the surrogate must clamp with separate low/high epsilons.

PPO_CLIP_EPSILON_LOW/HIGH are module-top imports in training.py, so tests
patch the *consuming* module (reliquary.validator.training), not
reliquary.constants.
"""
import torch

import reliquary.validator.training as T


def _surrogate(ratio, advantage):
    """Reference implementation of the clipped surrogate used by both paths."""
    lo = 1 - T.PPO_CLIP_EPSILON_LOW
    hi = 1 + T.PPO_CLIP_EPSILON_HIGH
    return torch.min(ratio * advantage, torch.clamp(ratio, lo, hi) * advantage)


def test_constants_exist_and_default_symmetric_off_v4():
    # Under the default (v2) test profile both must be 0.2.
    assert T.PPO_CLIP_EPSILON_LOW == 0.2
    assert T.PPO_CLIP_EPSILON_HIGH == 0.2


def test_asymmetric_band_clamps_upside_at_high(monkeypatch):
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_LOW", 0.2)
    monkeypatch.setattr(T, "PPO_CLIP_EPSILON_HIGH", 0.28)
    adv = torch.tensor([1.0])
    # ratio above 1+eps_high: clipped to 1.28
    assert torch.isclose(_surrogate(torch.tensor([1.5]), adv), torch.tensor([1.28]))
    # ratio below 1-eps_low with negative advantage: clipped at 0.8
    adv_neg = torch.tensor([-1.0])
    assert torch.isclose(_surrogate(torch.tensor([0.5]), adv_neg), torch.tensor([-0.8]))


def test_microbatch_and_rollout_paths_share_the_band():
    """Source-level guard: both clamp sites must reference LOW and HIGH."""
    import inspect

    src_a = inspect.getsource(T._rollout_loss)
    src_b = inspect.getsource(T._microbatch_grad)
    for src in (src_a, src_b):
        assert "PPO_CLIP_EPSILON_LOW" in src
        assert "PPO_CLIP_EPSILON_HIGH" in src
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_ppo_clip_asymmetric.py -q`
Expected: FAIL — `AttributeError: module ... has no attribute 'PPO_CLIP_EPSILON_LOW'`.

- [ ] **Step 3: Implement the constants**

In `reliquary/constants.py`, replace `PPO_CLIP_EPSILON = 0.2` (and its comment):

```python
# PPO clip band. DAPO clip-higher (arXiv 2503.14476 §3.1): the upper clip is
# what allows low-probability (exploration) tokens to grow; v4 adopts the
# published 0.2/0.28. Pre-v4 profiles keep the symmetric 0.2 band. These are
# interpretable as true ratio bounds only because v4 sampling makes warp() the
# identity (see test_warp_is_identity_softmax_at_dapo_values).
PPO_CLIP_EPSILON_LOW = 0.2
PPO_CLIP_EPSILON_HIGH = 0.28 if PROTOCOL_VERSION >= 4 else 0.2
```

- [ ] **Step 4: Update the consumers**

In `reliquary/validator/training.py`:
- Import line ~27: replace `PPO_CLIP_EPSILON` with `PPO_CLIP_EPSILON_LOW, PPO_CLIP_EPSILON_HIGH`.
- `_rollout_loss` clamp:
```python
    surr2 = (
        torch.clamp(ratio, 1 - PPO_CLIP_EPSILON_LOW, 1 + PPO_CLIP_EPSILON_HIGH)
        * advantage
    )
```
- `_microbatch_grad` clamp:
```python
    clipped_surr = (
        torch.clamp(ratio, 1 - PPO_CLIP_EPSILON_LOW, 1 + PPO_CLIP_EPSILON_HIGH)
        * adv_cat
    )
```
- `_record_kl_stats` counters:
```python
        stats["ppo_ratio_below_clip_count"] += int(
            (ppo_ratio < 1.0 - PPO_CLIP_EPSILON_LOW).sum()
        )
        stats["ppo_ratio_above_clip_count"] += int(
            (ppo_ratio > 1.0 + PPO_CLIP_EPSILON_HIGH).sum()
        )
```

In `reliquary/validator/service.py` (~line 443), replace the echo:
```python
        "ppo_clip_epsilon_low": PPO_CLIP_EPSILON_LOW,
        "ppo_clip_epsilon_high": PPO_CLIP_EPSILON_HIGH,
```
(and fix its import).

- [ ] **Step 5: Sweep for stragglers**

Run: `grep -rn "PPO_CLIP_EPSILON\b" reliquary/ tests/ scripts/ | grep -v "PPO_CLIP_EPSILON_LOW\|PPO_CLIP_EPSILON_HIGH\|PPO_RATIO_OUTSIDE_CLIP"`
Expected: zero hits. Fix any found (tests referencing the old name get the LOW/HIGH pair; behavior at v2/v3 values is identical so assertions should not need value changes).

- [ ] **Step 6: Run the trainer test files**

Run: `python -m pytest tests/unit/test_ppo_clip_asymmetric.py tests/unit/test_train_step_microbatch.py tests/unit/test_training_rollout_loss.py tests/unit/test_training_stub.py -q`
Expected: PASS — in particular `test_train_step_microbatch.py`'s batched-vs-per-rollout grad-equivalence tests, which prove the two clamp sites stayed in sync.

- [ ] **Step 7: Commit**

```bash
git add reliquary/constants.py reliquary/validator/training.py reliquary/validator/service.py tests/unit/test_ppo_clip_asymmetric.py
git commit -m "feat(training): DAPO clip-higher, 0.2/0.28 on v4 profiles

Symmetric 0.2 preserved for pre-v4. Both surrogate paths and the ratio
telemetry share the band.

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 8: Hard-reject forced-span claims when BFT is disabled

On v4 no honest rollout is ever `forced=True` (miner's `bft_applicable` is false). But `forced`/`force_span` remain wire fields, and the validation predicates key off budgets that are now 0 — a crafted claim must fail closed at the single choke point rather than depending on degenerate arithmetic. `validate_force_span` (verifier) and `_force_span_valid` (admission) are the two duplicate predicates; both get the guard.

**Files:**
- Modify: `reliquary/validator/verifier.py` (`validate_force_span`, ~line 1255)
- Modify: `reliquary/validator/admission.py` (`_force_span_valid`, ~line 293)
- Test: create `tests/unit/test_force_span_bft_disabled.py`

**Interfaces:**
- Consumes: `BFT_ENABLED` from `reliquary/constants.py` (profile-derived).
- Produces: both predicates return failure for any `forced=True` rollout whenever `BFT_ENABLED` is false; non-forced rollouts are unaffected.

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_force_span_bft_disabled.py`. The fixture geometry is copied from `tests/unit/test_bft_carveout.py` (atomic `</think>` = 777, canonical FORCE ids `[777, 7, 8]`, prompt_length 2, thinking budget 2 — a span the predicates **accept** today). `_force_span_valid(tokens, meta, context)` only reads `context.think_close_ids` and `context.canonical_force_ids`, so a `SimpleNamespace` suffices; its `BFT_THINKING_BUDGET` is a module-top import in `admission.py`, so patch it to 2 to make the fixture geometry valid — the new guard must then be the *only* rejection cause.

```python
"""Forced-span claims must fail closed when the active profile has no BFT."""
from types import SimpleNamespace

import reliquary.constants as C
import reliquary.validator.admission as admission
from reliquary.validator.admission import _force_span_valid
from reliquary.validator.verifier import validate_force_span

# Same fixture geometry as test_bft_carveout.py: this exact claim is ACCEPTED
# when BFT is enabled, so the guard is the only thing rejecting it below.
_FORCE = [777, 7, 8]
_CLOSE = {777}
_TOKENS = [0, 1, 5, 6, 777, 7, 8, 55, 99]
_META = {"forced": True, "force_span": (4, 7), "prompt_length": 2}


def test_validate_force_span_rejects_forced_claim_when_bft_disabled(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)
    ok, exempt = validate_force_span(
        _TOKENS, _META, _FORCE, 2, thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is False
    assert exempt == set()


def test_validate_force_span_non_forced_unaffected(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)
    ok, exempt = validate_force_span(
        [0, 1, 5, 6], {"forced": False}, _FORCE, 2,
        thinking_budget=2, think_close_ids=_CLOSE,
    )
    assert ok is True
    assert exempt == set()


def test_admission_force_span_rejects_when_bft_disabled(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)
    monkeypatch.setattr(admission, "BFT_THINKING_BUDGET", 2)
    ctx = SimpleNamespace(think_close_ids=_CLOSE, canonical_force_ids=_FORCE)
    assert _force_span_valid(_TOKENS, _META, ctx) is False


def test_admission_non_forced_unaffected(monkeypatch):
    monkeypatch.setattr(C, "BFT_ENABLED", False)
    ctx = SimpleNamespace(think_close_ids=_CLOSE, canonical_force_ids=_FORCE)
    assert _force_span_valid([0, 1], {"forced": False}, ctx) is True
```

(The lazy in-function `from reliquary.constants import BFT_ENABLED` added in Step 3 is what makes the `monkeypatch.setattr(C, "BFT_ENABLED", False)` visible to both predicates.)

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_force_span_bft_disabled.py -q`
Expected: the forced-claim tests FAIL (today the predicates evaluate the degenerate budget arithmetic instead of failing closed).

- [ ] **Step 3: Implement**

At the top of `validate_force_span`'s forced branch (after the early `return (True, set())` for non-forced rollouts) in `reliquary/validator/verifier.py`:

```python
    from reliquary.constants import BFT_ENABLED

    if not BFT_ENABLED:
        # No profile-sanctioned force span exists: any forced claim is tampering.
        return False, set()
```

Mirror in `reliquary/validator/admission.py`'s `_force_span_valid`:

```python
    from reliquary.constants import BFT_ENABLED

    if not BFT_ENABLED:
        return False
```

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_force_span_bft_disabled.py -q && python -m pytest tests/unit -q -k "force_span or admission"`
Expected: PASS, including all pre-existing force-span tests (they run under the v2 default profile where `BFT_ENABLED` is true).

- [ ] **Step 5: Commit**

```bash
git add reliquary/validator/verifier.py reliquary/validator/admission.py tests/unit/test_force_span_bft_disabled.py
git commit -m "fix(validator): fail closed on forced-span claims when BFT is off

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 9: Full-precision AdamW default on v4

DAPO trains with full-precision AdamW; 8-bit optimizer-state quantization noise is non-trivial at LR 1e-6. v4 defaults to full precision (H100 80GB fits it: ~32GB fp32 optimizer state for 4B params + bf16 params/grads; v4 sequences are short). Pre-v4 keeps PagedAdamW8bit. Env-overridable both ways.

**Files:**
- Modify: `reliquary/constants.py` (new constant near the optimizer/LR block)
- Modify: `reliquary/validator/training.py` (`_build_optimizer`, ~lines 57-89)
- Test: `tests/unit/test_training_stub.py`

**Interfaces:**
- Produces: constant `OPTIMIZER_STATE_8BIT: bool` (env `RELIQUARY_OPTIMIZER_STATE_8BIT`, default `"0" if PROTOCOL_VERSION >= 4 else "1"`). `_build_optimizer` returns plain `torch.optim.AdamW` whenever it is false, even on CUDA.

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_training_stub.py` (it already covers `_build_optimizer` CPU/CUDA selection — follow its style):

```python
def test_build_optimizer_full_precision_when_8bit_disabled(monkeypatch):
    import torch

    import reliquary.constants as C
    from reliquary.validator.training import _build_optimizer

    monkeypatch.setattr(C, "OPTIMIZER_STATE_8BIT", False)
    params = [torch.nn.Parameter(torch.zeros(2))]
    opt = _build_optimizer(params)
    assert type(opt).__name__ == "AdamW"
```

- [ ] **Step 2: Run to verify failure**

Run: `python -m pytest tests/unit/test_training_stub.py -q`
Expected: FAIL — `AttributeError` on `OPTIMIZER_STATE_8BIT` (constant does not exist yet).

- [ ] **Step 3: Implement**

In `reliquary/constants.py`, next to the LR block:

```python
# Optimizer-state precision. 8-bit paged AdamW (bitsandbytes) saves VRAM but
# adds quantization noise that is non-trivial relative to 1e-6 updates; the
# DAPO-faithful v4 profile defaults to full precision (fits on H100 for 4B).
OPTIMIZER_STATE_8BIT = (
    _os.environ.get(
        "RELIQUARY_OPTIMIZER_STATE_8BIT",
        "0" if PROTOCOL_VERSION >= 4 else "1",
    )
    not in ("0", "false", "False")
)
```

In `_build_optimizer`, gate the bitsandbytes branch (lazy import for testability):

```python
    from reliquary.constants import OPTIMIZER_STATE_8BIT

    if OPTIMIZER_STATE_8BIT and torch.cuda.is_available() and cuda_parameters:
        ...existing bnb branch unchanged...
```

- [ ] **Step 4: Run**

Run: `python -m pytest tests/unit/test_training_stub.py -q`
Expected: PASS (existing selection tests run under v2 default where the flag is true — unchanged behavior).

- [ ] **Step 5: Add the knob to the profile lock test**

Add `OPTIMIZER_STATE_8BIT: False` to the v4 expected dict (and `True` for v3 if the test asserts v3 knobs) in `tests/unit/test_protocol_profiles.py`; run it.

- [ ] **Step 6: Commit**

```bash
git add reliquary/constants.py reliquary/validator/training.py tests/unit/test_training_stub.py tests/unit/test_protocol_profiles.py
git commit -m "feat(training): full-precision AdamW default on v4 profiles

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

### Task 10: Whole-profile coherence test + full suite

A single subprocess test that boots the constants under the v4 profile and asserts the cross-cutting invariants that individual tasks cannot see together.

**Files:**
- Test: `tests/unit/test_v4_profile_coherence.py`

- [ ] **Step 1: Write the test**

Follow `test_protocol_profiles.py`'s subprocess convention (env `RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-v4`, run a `python -c` script that imports `reliquary.constants` and prints a JSON of resolved values, assert in the parent). Assert:

```python
EXPECTED = {
    # identity & wire
    "profile_id": "qwen3-4b-base-dapo-v4",
    "protocol_version": 4,
    "forced_seed_domain": "reliquary-forced-seed-v4",
    "upload_grace": 33,
    # DAPO recipe coherence
    "t_proto": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "kl_beta": 0.0,
    "clip_low": 0.2,
    "clip_high": 0.28,
    "recompute_pi_old": True,
    "overlong_factor": 0.5,
    "overlong_cache": 4096,
    "optimizer_8bit": False,
    # BFT fully off
    "bft_enabled": False,
    "bft_thinking_budget": 0,
    "bft_answer_budget": 0,
    "mask_math_forced": False,
    "mask_code_truncated": False,
    "mask_budget_ended": False,
    # prompt & corpus switches
    "raw_completion_prompts": True,
    "omi_train_shards_only": True,
}
```

Also, in the same subprocess script, assert `reliquary.constants` **imports without raising** (the module has several import-time `assert`/`raise` guards — this is the test that catches a v4 value tripping one).

- [ ] **Step 2: Run it, then the full unit suite**

Run: `python -m pytest tests/unit/test_v4_profile_coherence.py -q && python -m pytest tests/unit -q`
Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/test_v4_profile_coherence.py
git commit -m "test: v4 profile cross-cutting coherence lock

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Deployment checklist (manual, NOT part of the code tasks)

Gate: the offline confirmatory DAPO run (reliquary-experiments) has shown learning on Qwen3-4B-Base. Coordinate the whole cutover with 0xgrizz — it is a coordinated miner + validator deploy.

1. **Validator env** (`.env` on ubuntu@209.20.157.231): `RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-base-dapo-v4`; `RELIQUARY_TRAINING_RUN_ID=<new id, e.g. qwen3-base-dapo-v4>` (resets prompt cooldown — a fresh model must re-see every prompt); keep drand tolerance at its current live value (0).
2. **Fresh checkpoint repo** via the `--hf-repo-id` CLI option (there is no validator env var for it). Precedent: the 2026-07-17 base reset. GOTCHA: `RELIQUARY_CHECKPOINT` rejects `repo@rev` syntax.
3. **No KL env needed**: β=0 is the profile default. If collapse telemetry appears (entropy slide, charabia), the calibrated response is `RELIQUARY_KL_BASE_MODEL=<v4 ck0 repo>@<sha>` + explicit `RELIQUARY_KL_BETA` (fixed mode hard-requires both, plus recompute-π_old, which is already the v4 default).
4. **Miners**: same image, new profile env. The generation contract is compared byte-for-byte; v3 submissions are rejected at the door (`generation_profile_id` mismatch), so flag-day cutover at a window boundary.
5. **Proof fleet**: `proof_capacity` hashes T/top_k/top_p/caps + `forced_sampling.py` bytes — qualification resets at cutover by design. Expect a re-qualification window (the carryover feature on the current WIP branch does not apply: the proof path *does* change).
6. **Watch on the first windows**: `train/kl_beta == 0.0`, `train/pi_old_recomputed == 1.0`, `ppo_ratio_outside_clip_ratio` (should be small but nonzero — it now measures pure staleness in the correct space), overlong metrics (in-ramp fraction), window fill rate (expect it to rise with the ~7.7x faster generator), wandb run `{hotkey[:8]}-{version}` in project `reliquary`.
7. **Rollback lever**: revert `RELIQUARY_PROTOCOL_PROFILE` + `--hf-repo-id` + `RELIQUARY_TRAINING_RUN_ID` to the v3 values. v3 checkpoints are untouched by the entire plan.
8. **Post-cutover workstream** (before miner supply ramps): σ-gate/auction-δ decision, then `B_BATCH` 8→16 with `MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW` raised in the same commit.

## Self-review notes

- Spec coverage: D1 (Task 2 sampling + Task 3 pin), D2 (Task 7), D4 (inherited from PR #167, locked in Task 10), D5 (Task 2 β=0 + deploy §3 fallback), D6 (Task 2 recompute default, zero-cost via ref aliasing), D11 partial (Task 1 grader; integer filtering deferred), D12 (Task 9), D13/BFT removal (Task 2 `bft=None` + Task 8 fail-closed + Tasks 4-5 prompt), corpus duplication (Task 6). D3/D7/D8 are consensus/structural — explicitly deferred with rationale.
- Known intentional deviations from "pure DAPO": batch scale (structural), no resampling half of dynamic sampling (structural), σ-gate stays 0.43 (deferred), per-env loss weighting kept (deliberate, documented in the divergence audit), reward [0,1] (affine-equivalent).
- Tasks 1-9 each carry their own tests and are independently revertible; only Tasks 4/5 (prompt bytes) and 2 (profile) are mutually load-bearing at deploy time — all inert until the profile env is set.

---

## Status addendum — 2026-08-12 hardening pass

Landed on `feat/qwen3-base-dapo-v4-profile` (fb283c7..ce7ca2c):

- Forced-seed diagnostics route dense at full support (top_k=0/-1); behavioural
  thresholds recalibrated for the v4 envelope on H100 (40 honest 16k OMI
  rollouts, 0 FP end-to-end): seed floors 0.70/0.65, MIN_EOS 0.001, median
  0.05, q10 2e-4 — all profile-versioned, v3 untouched.
- Grader Answer:-line + LaTeX-delimiter extraction merged. Measured at ck0
  with this profile's prompt: 46/48 completions box, 0 old-vs-new verdict
  flips — robustness against RL format drift, not a launch blocker.
- verl parity guards: dual clip c=10 (negative-advantage loss bound), drift
  backstop armed at 0.5 for the 16-window pi_old interval.
- Local OMI save_to_disk snapshots fail closed on v4+ (len(env) consensus).
- ck0 capacity data: real-prompt rollouts terminate naturally at 51-1392
  tokens (median ~500) — collection window and proof wall hold at cutover;
  re-check both if RL lengthens responses.

Accepted deviations from the DAPO reference (documented, not bugs):

1. Dynamic sampling filters degenerate groups but cannot oversample to refill
   the batch (decentralized supply); effective batch shrinks with the
   degenerate-group rate.
2. Group filter keys on shaped-reward std, not raw accuracy: all-wrong groups
   with different overlong penalties still train (verl would drop them).
3. Token-mean is per-env then env-weighted (w_e/N_e), not global.
4. AdamW weight decay 0.01 (torch default) vs verl 0.1; cosine LR over 10k
   windows vs constant-after-warmup (negligible over a run's first thousands).

Open deploy decisions: optimizer-state precision default (fp32 vs 8-bit) and
proof-wall sizing; cutover with FORCED_SEED_ENFORCE=false for the first
windows until cross-stack ratios are confirmed on live traffic.

### 2026-08-13 — DAPO zone-gate criterion

The v4 zone gate now uses DAPO's dynamic-sampling criterion: SIGMA_MIN drops
to 0.24 (BOOTSTRAP 0.22) on v4: for M=16 binary the k=1/k=15 extremes have
σ=0.2421, so 0.24 admits every k ∈ [1,15] while still filtering near-degenerate
CONTINUOUS (code) clusters a 0.0 floor would wrongly admit (the hard 0.43 floor
had rejected k ∈ {1,2,3,13,14,15} at M=16). Pragmatic binary-faithful port;
full metric=acc binarisation deferred. This resolves
the *criterion* half of the earlier "dynamic sampling absent" deviation; the
oversample/refill half is still not implemented (structural to decentralized
supply). Economic ranking is delegated to the auction value std·(1-mean)^δ,
whose (1-mean)^δ factor favours the rare-correct prompts. Accepted tradeoff
(owner decision 2026-08-13): this widens what is admitted and PAID and softens
the manufactured-variance surface the hard gate covered — watch the emission
share of low-σ groups and the reward-zone exploit signals after cutover. v3
economics unchanged (0.43/0.33).

### 2026-08-13 — length-curriculum start point

v4 profile starts at half of v3's window/cap: collection_seconds 300→150,
max_new_tokens 16384→8192 (both envs), throughput token_cap →8192. ck0
Qwen3-4B-Base terminates well short of 16384 (real OMI: median ~500, max ~1392
/ 40 rollouts), so a 16384-sized window burned wall-clock and seal-verify from
day one. 8192 sits above OVERLONG_PENALTY_CACHE_TOKENS=4096 so the soft-overlong
zone [4096,8192] stays ABOVE the natural length (clean zone [0,4096] covers the
whole ck0 distribution). Intended as the base of a length curriculum that ramps
cap+deadline up as the policy's reasoning lengthens; the cap-hit rate (already
observable via the `truncated` meta, uncensored) is the thermostat that says
when to raise. Not yet automated — this commit sets the starting constants only.
Watch: window under-fill at 150s (B_BATCH=16 × G=16 = 512 sequences must still
accumulate) and the code-env length distribution (unmeasured at ck0).
