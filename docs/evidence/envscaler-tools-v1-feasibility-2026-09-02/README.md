# Can Reliquary use EnvScaler as a tool-calling environment? — 2026-09-02/03

The adapter works. This asks the separate, deciding question: does the
environment produce a training signal against the model we actually train?

Nothing is activated by this. `envscaler_tools_v1` is not in `registry.py`
and no profile declares it.

**Answer: not against `Qwen3-4B-Base`, but the reason is not the one a
first pass reported.** Under a faithful port, no rollout in 3,840 solves a
task and no group clears the gate — yet the best group reaches sigma 0.210
against a threshold of 0.24. The environment supplies gradient; it falls
just short of the bar that decides whether the gradient is kept.

## Bound identities

- H100 PCIe 80 GB, vLLM 0.28.0, torch 2.13.0+cu130,
  `VLLM_USE_FLASHINFER_SAMPLER=0` (no CUDA toolkit in the image).
- `Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539` — the
  revision the profiles pin — and `Qwen/Qwen3-14B` for the first-pass ceiling.
- T=1.0, top_p=1.0, top_k disabled, 16 rollouts, seed 7, 512 max new tokens
  per turn, 12 turns, 48 tasks per configuration.
- Prompting through the branch's own `CanonicalEpisodeRenderer`.

## What the corpus is

51 worlds, 2,550 scenarios, from EnvScaler's published RL split. Each world
is a Python class exposing tools; each scenario is a request plus boolean
check functions over the terminal state.

| | |
|---|---:|
| tools per scenario | 18 median, 25 max |
| checks per scenario | 13 median |
| checks false at reset | 11 median |
| scenarios with <= 4 such checks | 2.9% |
| initial prompt with the TOOLS block | ~14 KB, ~3.5k tokens |

The corpus also ships **140 SFT worlds against these 51 RL worlds**;
`rl_scen.json` covers only the 51. The RL half assumes a policy already
taught to call tools on the disjoint SFT half — a stage skipped here,
against a base model that has never emitted a tool call.

## Determinism

The upstream classes are LLM-written and reach for wall-clock time, uuid4
and `datetime.now`: across the 191 published classes, 31.4% call
`time.time()`, 31.9% `datetime.now()`, 20.4% uuid, 36.6% are clean. Left
alone, miner and validator replaying one episode observe different bytes
and the honest miner is rejected.

Three frozen shims (`shims.py`, pinned by `tests/unit/test_envscaler_shims.py`)
recover every world: **500 of 500 replayed episodes byte-identical.** An
earlier 188/300 divergence was two mistakes — a missing `deepcopy` of the
config, and a forgotten `datetime` shim.

One further class calls `random.choice`: `env_42_sft`, outside the 51 RL
worlds. Ingesting the SFT split later would need a fourth shim.

## What upstream actually does

From the published `base_env.py`, `rl_non_conv_env.py` and `parse_util.py`.
A first pass measured a port that diverged from all of this, and the
divergences all made the task harder.

| | upstream | first pass |
|---|---|---|
| unreadable turn | error observation, episode continues, uncapped | fatal after 4 |
| unknown tool | error observation, episode continues, uncapped | fatal after 4 |
| prose with no tool call | routed to `chat_with_user`, which **terminates and scores the state reached** | invalid action |
| tool raises | fatal | fatal after 4 |
| reward | `sum(checks)/len(checks)` — **continuous, over every check** | binary, conjunctive, over checks false at reset |

The reward was decisive. Binary-conjunctive over 11 checks is identically
zero for every rollout of a 4B, so sigma is exactly zero and the band is
empty **by construction rather than by measurement**. Upstream's reward is
non-zero for doing nothing: over 600 scenarios a no-op agent scores
**0.166 mean, 0.111 median**, zero in 31.5% — and breaking an invariant
costs reward.

## Result — faithful port

48 tasks x 16 rollouts per configuration. `--contract` is how a turn is
read (`strict` is production's `AssistantAction.from_json`, which demands
the whole completion be one bare JSON object; `lenient` takes the first
brace-balanced object anywhere). `--prose` is what an unreadable turn means
(`terminates` is upstream's routing; `retries` makes it recoverable, which
separates "the base model rambles" from "the environment cannot spread").

| | reliquary strict/term | reliquary lenient/term | reliquary lenient/retries | qwen chatml/retries |
|---|---:|---:|---:|---:|
| mean reward | 0.163 | 0.195 | 0.225 | 0.155 |
| best-of-16 reward | 0.170 | 0.219 | 0.252 | 0.155 |
| rollouts fully solving | 0.0% | 0.0% | 0.0% | 0.0% |
| **GROUPS IN BAND (sigma>=0.24)** | 0.0% | 0.0% | 0.0% | 0.0% |
| groups with any spread | 4.2% | 18.8% | 22.9% | 0.0% |
| median group sigma | 0.000 | 0.000 | 0.000 | 0.000 |
| max group sigma | 0.132 | 0.195 | 0.210 | 0.000 |
| groups sigma>=0.05 | 2.1% | 12.5% | 10.4% | 0.0% |
| groups sigma>=0.10 | 2.1% | 10.4% | 8.3% | 0.0% |
| groups sigma>=0.15 | 0.0% | 6.2% | 4.2% | 0.0% |
| groups sigma>=0.20 | 0.0% | 0.0% | 2.1% | 0.0% |
| ended by a raising tool | 5.5% | 12.5% | 27.6% | 0.0% |
| turns unreadable | 48.6% | 16.0% | 39.1% | 100.0% |
| explicit termination | 92.6% | 79.7% | 39.2% | 0.0% |
| over the 16k budget | 0.0% | 0.0% | 0.0% | 0.0% |
| turns (median) | 1 | 2 | 6 | 12 |
| generated tokens (median) | 193 | 294 | 636 | 5679 |

Three readings.

**The contract fix is worth a factor of five and is not enough.** Going
from production's strict reading to the lenient one raises mean reward from
0.163 to 0.225 against a 0.166 floor, and takes groups with any spread from
4.2% to 22.9%. The band stays at 0.0%.

**Qwen's own frame is worse, not better.** Rendering the episode as ChatML
with `<tool_call>` tags — the shape Qwen3 was post-trained on — collapses
to 100% unreadable turns and a mean reward of 0.155, *below* the do-nothing
floor. Inspection of the raw completions shows why: the base model emits
the right payload (`{"name": "get_user_by_id", "arguments": {...}}`) inside
invented tags (`<tool_use>`, `<dration>`), and otherwise drifts into
Chinese and repetition loops. `<tool_call>` is a post-training convention
and this model has had no post-training. Our own frame is the better one
here, which retires the hypothesis that the renderer is what costs us.

**No rollout ever solved a task** — 0 of 3,840 across every configuration,
including the 14B first pass.

## What sigma >= 0.24 demands of a continuous reward

Derived before the runs, and matched by them. The gate
(`difficulty_auction.py:113`, `SIGMA_MIN` in `constants.py:623`) is a raw
standard deviation over the group's rewards, with no normalisation for the
reward's scale. With a no-op floor of 0.166:

- as a near-binary mixture of "solves" and "does nothing", it needs
  **10% to 90% of rollouts to solve the task outright**;
- as partial progress alone, uniformly spread, it needs a spread of
  **0.83 of reward — about 11 of 13 checks — between rollouts of one group**.

Partial progress cannot reach it: five checks of spread between rollouts
gives sigma ~0.115, under half the bar. **The threshold implicitly assumes
a binary reward.** Any environment with graded credit is handicapped by it,
which is a fact about the gate, not about EnvScaler.

The observed maximum is 0.210, and the shape of that group is instructive.
`env_158_rl-task_5` is bimodal — its rollouts score either 0.077 or 0.615,
so some flip 8 of 13 checks and the rest flip one. That is the right shape.
It misses only because the upper mode is minority (about 3 of 16); balanced
8/8 the same two levels give sigma 0.269, which clears the gate.

What separates the 11 groups with spread from the 37 without is **not**
scenario length — both have a median of 13 checks. It is mean reward, 0.296
against 0.203: whether the model engages the task at all.

## First pass, superseded

`fix_*.json` hold the unfaithful-port runs. Their band figures measure our
contract and say nothing about EnvScaler, but two findings from them stand
because they concern `runner.py`, the single action path for all five
episode environments on this branch:

- **The strict contract is a branch-wide tax.** At turn 1, **78.2% of
  rollouts emit a valid, correctly named tool call while 27.1% satisfy the
  contract** — the model writes a sentence of reasoning, then the JSON, and
  production discards both. No Qwen3-14B rollout in 768 was bare JSON on
  every turn: the contract is most hostile to the models most able to do the
  work. This is plausibly part of the 35-43% invalid-action rates measured
  on the branch's own episode suite.
- **Binary reward measures termination, not solving.** One 14B group had all
  sixteen rollouts at the goal state and ten binary successes; the other six
  solved it and did not emit the final marker. The same shape explains
  `stateful_tools_v1`'s apparent 62% success.

## Cost

At the best configuration's median of 6 turns, a group of 16 rollouts
re-prefills roughly **340k tokens**, against about **10k** for a
16-rollout math or code group. That ratio is the input `w_env` needs.

## What would change the verdict

1. **Tasks a 4B solves outright between 10% and 90% of the time.** That is
   what the gate requires, stated as a specification. This corpus supplies
   none — hence the offline generator already planned.
2. **A gate that normalises for reward scale**, or an env-specific
   threshold. The best group here is at 0.210 against 0.24; the shape is
   right and the scale is not.
3. **The SFT stage.** 140 worlds exist for it and were not used.
4. **A relaxed action contract**, worth doing on its own merits for the four
   active episode environments whatever is decided about EnvScaler.

## Recommendation

Keep the adapter, do not register it — it replays deterministically and
costs nothing dormant, the posture `cipher` has in the logic roster.

Do not read this as "EnvScaler is too hard". Read it as: against a base
model with no tool-use post-training, on the RL half of a corpus whose SFT
half we skipped, the environment lands a factor of 1.15 short of a gate
calibrated for binary rewards.

## Reproducing

The corpus is **not vendored**. `RELIQUARY_ENVSCALER_DATA` must point at a
directory holding EnvScaler's `env_meta.json` (191 world classes) and
`rl_scen.json` (the RL split); the loader keeps the scenarios whose
`env_id` resolves and **addresses them by position**, so file order is part
of the corpus identity. Before any activation this should be a pinned
revision of a fork rather than a loose directory.

The exact bytes these measurements ran against:

```text
d2c0010f16ff77d6d55868ee386353b1d0aadace58beed1eed678e8f7c84c33d  env_meta.json
5977bda0b941a9111b290cbf5ffd6d70678a36ddc499b8f153826fd22999337e  rl_scen.json
```

```bash
RELIQUARY_ENVSCALER_DATA=<data> VLLM_USE_FLASHINFER_SAMPLER=0 \
python scripts/eval_envscaler_band.py --model Qwen/Qwen3-4B-Base \
  --revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --tasks 48 --max-turns 12 --max-action-tokens 512 \
  --contract lenient --prose retries --output band.json
```

## File integrity

```text
ec5cec5985387f515598d2f8cb804cd2665dd46b5dac331e1dcb5f98a18c63be  fix_14b_strict.json
9fefecc37114c6110840f5378d9db4659e93009905258a5d6cf99bde4f8fc7f2  fix_4b_lenient.json
c0cc72a5000a670d5fd160333a91c4c5f562db1aae22535ed86afdaeda7158f1  fix_4b_strict.json
a5d2f95b3a9ff29bd6042ce96060be2102fba2cb857aa5fa69e3d2e42906e46c  qw_retries.json
95e0abf2189f1df59ce94f2e46e41bf147cf968fee440cc451cb7fcace218a8b  qw_terminates.json
f7764aca17b64f6b18f4d5966601f2366531a181db95b5aea624b25364f762eb  up_lenient_retries.json
d676590469f382b62ee49db05444f97bee79bed5d2d5b203ce28e49c68dfa414  up_lenient_terminates.json
be87531620b061c25cef590a541954345399fdf288d863f9522b4446dfc1fd50  up_strict_retries.json
25e365629695da1a8915765321bc0393020bfa117c74ad71c931302c81d4a3f2  up_strict_terminates.json
```
