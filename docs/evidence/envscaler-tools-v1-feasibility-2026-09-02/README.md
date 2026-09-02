# Can Reliquary use EnvScaler as a tool-calling environment? — 2026-09-02

The adapter is written and works. This measures whether it produces a
training signal, which is a different question and the one that decides.

Nothing is activated by this: `envscaler_tools_v1` is not registered in
`registry.py` and no profile declares it.

## Bound identities

- GPU: NVIDIA H100 PCIe 80 GB (sm90), vLLM 0.28.0,
  `VLLM_USE_FLASHINFER_SAMPLER=0` (no CUDA toolkit in the image).
- Models: `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`
  — the revision the profiles pin — and `Qwen/Qwen3-14B` as a ceiling.
- Sampling: T=1.0, top_p=1.0, top_k disabled, 16 rollouts, seed 7,
  512 max new tokens per turn, 12 turns.
- Prompting: the branch's own `CanonicalEpisodeRenderer`, not a bespoke
  format. A first pass used a hand-written renderer and is discarded.

## What the corpus is

51 worlds, 2,550 scenarios, drawn from EnvScaler's published RL split.
Each world is a Python class exposing tools; each scenario is a natural
language request plus boolean check functions over the terminal state.

| | |
|---|---:|
| tools per scenario | 18 median, 25 max |
| checks per scenario | 13 median |
| checks that *count* (false at reset) | 11 median |
| scenarios with <= 4 counting checks | 2.9% |
| scenarios with 0 counting checks | 0 |
| initial prompt with the TOOLS block | ~14 KB, ~3.5k tokens |

The difficulty distribution has no easy tail to carve out: the minimum is
2 counting checks, and only 75 of 2,550 scenarios sit at 4 or below.

## Determinism

The upstream classes are LLM-written and reach for wall-clock time, uuid4
and `datetime.now` — measured across the 191 published classes, 31% call
`time.time()`, 32% `datetime.now()`, 20% uuid, and only 36% are clean.
Left alone, miner and validator replaying one episode observe different
bytes and the honest miner is rejected.

Three frozen shims (`shims.py`, pinned by `tests/unit/test_envscaler_shims.py`)
recover every world: **500 of 500 replayed episodes are byte-identical.**
An earlier 188/300 divergence was two mistakes of mine — a missing
`deepcopy` of the config, and a forgotten `datetime` shim.

One further class calls `random.choice`, which no shim covers. It is
`env_42_sft`, outside the 51 RL worlds, so the corpus in use is unaffected
— but ingesting the SFT split later would need a fourth shim.

## Cost

At the measured median of 5 turns, a group of 16 rollouts re-prefills
roughly **280k tokens**, against about **10k** for a 16-rollout math or
code group — near **28x**. That ratio is the input `w_env` needs.

## Result — 48 tasks x 16 rollouts, three contracts

Two mistakes of mine were corrected before these runs and both had been
suppressing the numbers: a greedy `{.*}` in the harness that read 96%
invalid actions off a model emitting valid ones, and a `step()` that ended
the episode the first time a tool raised. A tool error is now an
observation with a budget of four, which is how the agent is supposed to
learn to correct.

| | 4B strict | 4B lenient | 14B strict |
|---|---:|---:|---:|
| pass@1 (binary) | 0.000 | 0.000 | 0.040 |
| groups in band | 0.0% | 0.0% | 2.1% |
| k=0 | 100.0% | 100.0% | 95.8% |
| k=16 | 0.0% | 0.0% | 2.1% |
| mean fractional reward | 0.014 | 0.065 | 0.210 |
| rollouts moving >=1 check | 6.9% | 25.3% | 35.0% |
| rollouts reaching the goal | 0.0% | 0.0% | 6.1% |
| groups with sigma_frac >= 0.24 | 0.0% | 0.0% | 2.1% |
| groups with any spread | 6.2% | 20.8% | 16.7% |
| episodes killed (budget 4) | 82.4% | 49.3% | 61.7% |
| turns unreadable | 65.4% | 29.8% | 42.0% |
| turns that are bare JSON | 34.6% | 27.2% | 58.0% |
| explicit termination | 13.3% | 38.0% | 8.5% |
| turns (median) | 4 | 5 | 4 |
| generated tokens (median) | 571 | 606 | 1122 |

`band` is the share of 16-rollout groups whose reward sigma clears
`SIGMA_MIN` (0.24) — at 16 rollouts, 1 to 15 successes. Outside it the
auction values the group at zero and the gate drops it, so it is the only
part of a corpus that produces gradient.

### The environment does not train our model

**Zero of 768 rollouts solved a task under either contract on
Qwen3-4B-Base**, and no group lands in the band under binary *or*
fractional reward. That is the answer to the question this document asks.

Qwen3-14B is markedly better at the task itself — mean fractional reward
0.210 against 0.065 — and still yields **one gradient-producing group in
48**. Paying 28x a math group's tokens for a 2.1% band is not a trade
worth making.

### The action contract is expensive, and free to fix

`AssistantAction.from_json` requires the whole completion to be one bare
JSON object after `strip()`. Relaxing that to *the first brace-balanced
object anywhere in the turn* — same action space, same determinism, both
sides running the same function — is worth, on the same generations:

| | strict | lenient |
|---|---:|---:|
| episodes killed by errors | 82.4% | 49.3% |
| turns unreadable | 65.4% | 29.8% |
| groups with any reward spread | 6.2% | 20.8% |
| rollouts moving at least one check | 6.9% | 25.3% |
| explicit termination | 13.3% | 38.0% |

Two observations sharpen this. At turn 1, **78.2% of rollouts emit a
valid, correctly named tool call while only 27.1% satisfy the contract** —
the model writes a sentence of reasoning, then the JSON, and production
discards both. And **no Qwen3-14B rollout, in 768, produced bare JSON on
every turn**: the contract is most hostile to the models most able to do
the work.

This is not an EnvScaler finding. `runner.py` is the single path for all
five episode environments on this branch, so the same tax is being paid by
`stateful_tools_v1`, `retrieval_tools_v1`, `workspace_tools_v1` and
`web_tools_v1`, whose 35-43% invalid-action rates were measured before
this was understood.

### Binary reward measures termination, not solving

On the 14B, `env_141_rl-task_3` has **all sixteen rollouts at fractional
1.0** — every counting check flipped — and ten binary successes. The six
others solved the task and did not emit the final marker. A group's binary
variance there is a formality, not a difficulty signal. The same shape
explains `stateful_tools_v1`'s apparent 62% success, 95% of which tracked
clean termination.

The `invalid_actions == 0` veto on success was dropped for the same
reason: recovering from a failed call is the behaviour being trained, so
errors are reported as checks rather than voiding the episode.

### Why the corpus is out of reach

The median scenario needs **11 checks to flip together**, and the
distribution has no easy tail: the minimum is 2, and 2.9% of scenarios sit
at 4 or below. Fractional credit does not rescue this, because the
fractional reward is nearly bimodal — a world is either manipulated
correctly or not touched at all. Median group sigma is 0.000 in all three
runs.

## What would change the verdict

1. **Generated tasks at 2-5 checks.** The corpus cannot supply them (75 of
   2,550), so they have to be produced — which is the offline generation
   pipeline already planned as the next step.
2. **A relaxed action contract.** Worth doing on its own merits for the
   four active episode environments, independent of EnvScaler.
3. **A materially stronger policy.** The 14B result is the evidence that
   the ceiling moves with model strength; it is not evidence that a 9B
   would clear the band.

## Recommendation

Keep the adapter, do not register it. It is 447 lines plus tests, it
replays deterministically, and it costs nothing while dormant — the same
posture as `cipher` in the logic roster. Re-measure when either the task
generator or a stronger policy exists.

## Reproducing

```bash
RELIQUARY_ENVSCALER_DATA=<data> VLLM_USE_FLASHINFER_SAMPLER=0 \
python scripts/eval_envscaler_band.py --model Qwen/Qwen3-4B-Base \
  --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --tasks 48 --max-turns 12 --max-action-tokens 512 \
  --contract strict --output band.json
```

## File integrity

```text
ec5cec5985387f515598d2f8cb804cd2665dd46b5dac331e1dcb5f98a18c63be  fix_14b_strict.json
9fefecc37114c6110840f5378d9db4659e93009905258a5d6cf99bde4f8fc7f2  fix_4b_lenient.json
c0cc72a5000a670d5fd160333a91c4c5f562db1aae22535ed86afdaeda7158f1  fix_4b_strict.json
```
