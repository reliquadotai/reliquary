# Episode limits per environment (2026-11-14)

Goal: an agentic eval order lets the client choose its episode limits (turns, context, wall
time) within bounds set per environment. The default is each benchmark's own agent settings. The
quote scales with the chosen limits, and the report records the limits used and whether they were
the reference ones.

Status: branches `feat/episode-limits-per-env` in Catalyst and in the platform (reliquary-platform
#219), with PRs open.
Nothing is merged or deployed.

## 1. Why

On 2026-11-13 the DeepSWE and SWE-bench Pro test orders scored 0. 9 of 10 episodes hit our fixed
48-turn cap and 1 hit the 60k context. Those benchmarks' reference agents run 250 turns or have no
cap, so a 48-turn cap measures our cap rather than the model.

Other harnesses work the same way: each task has its own defaults and the user can override them.

| Tool | How the user overrides limits |
|---|---|
| Inspect AI | `message_limit`, `token_limit`, `time_limit` per task or per eval |
| verifiers | a `max_turns` env argument (default None) and token and timeout caps |
| HUD | `max_steps` |
| Harbor | per-task `[agent].timeout_sec`, plus `timeout_multiplier` and `override_timeout_sec` |

## 2. Reference settings

Gathered 2026-11-14 from each benchmark's repo, paper or leaderboard.

| Set | Reference agent settings | Source | Our default |
|---|---|---|---|
| SWE-bench Verified (k100, full k493) | mini-SWE-agent `benchmarks/swebench.yaml`: `step_limit: 250`, `cost_limit: 3`, 60 s per command, the model's context. On `LimitsExceeded` it submits nothing (scored 0). | github.com/SWE-agent/mini-swe-agent | 250 turns, 131,072 tokens (profile `swebench-bash-only`) |
| SWE-bench Pro | Scale public leaderboard: "uncapped cost and a turn limit of 250", SWE-agent or mini-SWE-agent (* rows). The paper used 50 turns and $2. Pro V2's locked protocol has no step cap and a 50-minute budget. | labs.scale.com/leaderboard/swe_bench_pro_public, arXiv 2509.16941, scaleapi/SWE-bench_Pro-os v2/README | 250 turns, 131,072, 4 h wall |
| DeepSWE v1.1 (Datacurve) | mini-SWE-agent with no step or cost cap and the model's native context; running out of context or time is a failure. The agent timeout is 9,000 s in the paper and 10,800 s in the repo's `task.toml`. | arXiv 2607.07946, github.com/datacurve-ai/deep-swe | 250 turns (our ceiling), 131,072, 4 h wall |
| Prime's `deep_swe` env (same tasks) | `max_turns` None (unlimited), task timeouts ignored by default | PrimeIntellect-ai/prime-envs, verifiers | (comparison only) |
| Terminal-Bench 2.1 | Terminus 2: no turn cap, the model's context with summarisation, each task's own agent timeout (360 to 12,000 s) | harbor-framework/harbor, terminal-bench-2-1 | 250 turns, 131,072, each task's timeout (profile `terminal-bench-terminus-2`) |
| NL2Repo-Bench | no published reference agent configuration found | — | 250, 131,072, 4 h (`official: null`) |
| TMax (`tmax_sft`) | our own set, no external leaderboard | — | 100 turns, 65,536, 2 h |
| AutomationBench (Zapier) | eval script `--max-steps 50` (the env class itself defaults to 25); a step may make parallel calls | github.com/zapier/AutomationBench | not orderable yet (see §7) |

Where a reference has no turn cap, we use 250, the ceiling of the corpus wire
(`protocol.corpus_submission.MAX_PROFILE_TRAJECTORY_TURNS`). Where it uses the model's own
context, we use 131,072. A 128k audit fits on one 80 GB card for Qwen3.5-9B (measured on
2026-11-13). The catalog text calls these values the reference "as the subnet runs it" and spells
out every deviation.

## 3. Scoring: model limits score 0, infra limits are redone

This follows #507 (`swe-verified-full`):

- **Model limits score 0:** a run that hits its turn cap (`max_turns`), its context
  (`context_length`) or the box's call or transcript budget (`budget_exhausted` on `max_calls`
  or `transcript_bytes`) scores 0. That is how mini-SWE-agent and the DeepSWE paper count them.
  The report counts these runs under `limit_cuts`.
- **Infra limits are redone:** a run that hits the wall budget (`expired`), the CPU budget, a
  failed box, a lost miner or a lost gateway is replayed on the same public draw and never
  scored. That is why wall time can be chosen but is never something the model is scored on: a
  slow miner or gateway must not cost a model points.

There is one known deviation. SWE-agent auto-submits the current diff when it hits a limit, and
some of Scale's Pro leaderboard rows are SWE-agent runs. We score a limit cut as 0, like
mini-SWE-agent and like the leaderboard's mini-SWE-agent rows.

## 4. Catalyst: limits stated directly (`episode.limit_cut_fails`)

Before this change, an eval episode reached 250 turns and 131,072 tokens only through a named
profile (`EPISODE_PROFILES`). A profile also pins its harness, its per-command timeout and a
minimum wall budget. Adding one profile per benchmark would repeat those pins for every set. A
profile is still the right tool when a leaderboard pins a harness and a command timeout (SWE
bash-only, Terminus 2), so the profiles stay.

The new field is `episode.limit_cut_fails: true`, written only when true so every other manifest
hashes as before. It applies to eval jobs only, and never together with a profile, since a profile
already implies it. With the field set:

- The ceilings are the wire's: `max_turns` up to `EPISODE_TURNS_CEILING` (250) and
  `max_total_tokens` up to `EPISODE_TOKENS_CEILING` (131,072).
- `sandbox.budgets.cpu_s` may reach the wall bound (14,400 s; every gateway runs with
  `RELIQUARY_SANDBOX_EPISODE_MAX_CPU_S=14400`).
- `_check_limit_budgets` enforces the same rules as for a profile:
  - the episode runs on the signed sandbox;
  - `max_calls >= max_turns`, so a model making one call per turn never hits the call budget
    first;
  - `cpu_s >= wall_s`, so the wall budget (infra) cuts first.
- Scoring goes through `corpus.job.limit_cuts_fail(episode)`, which is true for a scoring profile
  or for the field. `scores_model_closes`, `model_closed` (miner, signed intake, sessions) and
  `eval.grading.episode_limit_cut_fails` all read it. The model-limit and infra-limit rules above
  hold unchanged.
- The report's `provenance.episode` (`eval.job_grading.episode_protocol`) records `max_turns`,
  `max_tokens_per_turn` and `max_total_tokens` as before. For any episode that scores limit cuts
  it adds `limit_cut_fails`, `per_call_timeout_s`, `max_calls` and `wall_s`, and `profile` when
  there is one.

Unchanged:

- Without the field and without a profile, the ceilings stay at 64 turns and 60,000 tokens.
- RL and corpus jobs refuse the field.
- The corpus wire already takes 250 turns and 131,072 tokens.
- The admin's existing check refuses an episode longer than the model's `max_position_embeddings`.

**This needs a miner release.** Miners and validators parse the job with `corpus.job`. A build
without this change refuses a manifest carrying `limit_cut_fails` (unknown field), so the platform
sends the field only once `EVAL_EPISODE_DIRECT_LIMITS=on` (§5). The profile path (SWE, TB 2.1)
needs no new release beyond the one that runs the #507 and #518 profiles.

Miners must also be able to serve 131,072-token episodes. The engine is sized from the job
(`max_model_len = max_total_tokens`), so a miner whose GPU cannot hold that context fails to start
the job's engine and does not take it. Miners should check this before turning on eval jobs.

## 5. Platform: one table, order fields, quote, report

`worker/evaluation-episodes.ts` `EPISODE_LIMITS` has one entry per agentic set:

- `defaults`: the reference settings from §2;
- `official`: the reference as a customer reads it, or null;
- the bounds for each field:
  - turns 10 to 250;
  - context 32,768 to 131,072;
  - `wall_seconds` 3,600 to 14,400, or null where it is not the order's to choose;
- `profile`: the reliquary profile, when the set plays one;
- `pace`: an optional gateway-sizing override.

**Order fields per agentic set** (or at the order level for a one-set order): `max_turns`,
`max_episode_tokens` (the context bound) and `wall_seconds`. A field left out takes the set's
default. A value out of bounds gets a 422 `episode_protocol_mismatch` that names the set (on a
multi-set order), the range and the default. `wall_seconds` is refused where it is fixed (SWE
sets: the profile's 4 h; TB 2.1: each task's own timeout). `profile` is still accepted where
offered, for back-compat. On the SWE sets it is now the default.

**The job:**

| Set | What the job carries |
|---|---|
| profile sets (SWE, TB 2.1) | the profile template, with the chosen turns and context (reliquary bounds them by the profile) |
| other sets (DeepSWE, SWE-bench Pro, NL2Repo, TMax), `EVAL_EPISODE_DIRECT_LIMITS=on` | the template plus `limit_cut_fails: true`, `max_calls: 512` (at least the turns), and `wall_s = cpu_s =` the chosen wall |
| other sets, flag off | their template's 48 turns and 60,000 tokens, as before (`legacy`, `official_settings: false`) |

**Quote:**

- Reserved tokens are already rows × min(max_new_tokens × max_turns, max_episode_tokens), so they
  scale with the chosen limits.
- Gateway hours now scale with turns: episode seconds = max(`GATEWAY_EPISODE_SECONDS`, turns × set
  pace). The default pace is the SWE leaderboard estimate, 2,700 s expected and 7,200 s long per
  250 turns. The worst case takes the longest episode at the chosen wall budget.
- TB 2.1 keeps its old pricing (900 s / 2,700 s), because each task's own timeout bounds a run
  there.
- The 24 h run cap is unchanged. Past it, the quote is refused (`exceeds_generation_deadline`)
  with the hours needed, and suggests fewer problems or samples, splitting the order, or lower
  limits.

**Report:**

- The quote, the order view, and the delivered summary and webhook all record, per agentic set,
  `max_turns`, `max_episode_tokens`, `wall_seconds`, `official_settings` and `official`.
- `official_settings` is true only when all three limits equal the set's reference. It is always
  false when the set has no reference (NL2Repo) and on legacy sets.
- The catalog's `subnet_episode.limits` shows each field's default, min and max, the `official`
  text and `limit_cut_fails`. Its `max_turns` and `max_episode_tokens` fields keep meaning "at
  most".
- The set descriptions (D1 `eval_sets.description`) still say "up to 48 turns". A prepared update
  is kept with the eval-catalog runbooks; it is applied at deploy.

## 6. Deploy order

1. Merge the Catalyst PR, then cut and announce a miner and validator release. Wait until the
   miners that serve eval jobs run it.
2. Deploy the platform with `EVAL_EPISODE_DIRECT_LIMITS` off. At this point:
   - SWE and TB 2.1 orders play their profile at 250 turns and 131,072 tokens by default (the
     #507 and #518 code);
   - the other sets keep 48 turns and 60,000 tokens;
   - catalog, quote and report show the new fields.
3. Set `EVAL_EPISODE_DIRECT_LIMITS=on`. DeepSWE, SWE-bench Pro, NL2Repo and TMax then default to
   their reference limits.
4. Apply the catalog description update.

## 7. Not done

- **AutomationBench:**
  - It is a suite episode (replayed by the validator), and suite jobs stay at 64 turns and 60,000
    tokens.
  - The package's `max_turns` is 100: one call per turn, against upstream's 50 steps that may each
    make parallel calls.
  - Before it is offered, either the suite ceilings rise for eval jobs or the default is 50 steps
    mapped to 64 turns.
- **SWE-bench Pro V2 protocol** (50-minute budget, no step cap, partial patch graded): we do not
  score on wall time, which is infra here.
- **Per-command timeouts** stay as the templates set them: 60 s on SWE (the leaderboard's) and 600
  s elsewhere. Scale's Pro config uses 450 s, and the DeepSWE and Pro V2 minis probably use 30 s
  (not stated upstream).
