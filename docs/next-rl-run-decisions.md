# Next RL run: decisions to make

Branch: `feat/next-rl-run` = main + 0xgrizz's service-policy stack (#317-#328) +
signed-episode sandboxes (`feat/signed-episode-validator`). Draft PR #333 (CI only,
**never merge before the run launch**: a merge to main is a weight-only deploy).

Phases:
- 0, base branch: **done** (07-10, a9a10f1d, 0 conflicts).
- 1, adapt the service-policy stack to RL: decisions below, then code.
- 2, multi-turn RL: 16 sandbox sessions per group, observations masked in the loss.
- 3, async windows: measure the tolerable staleness first.
- 4, infra and a short real run.

Status: `DECIDED`, `OPEN` or `KEEP` (unchanged).

Still open: J only (left open on purpose), plus the "even for math?" point of H.

## State of the run's inputs (07-10)

SFT distillation (Qwen3.8-27B teacher, corpus jobs on the subnet), before the RL:
- Done: code v1, math OMI, IF, logic, science (cap 0 since 07-10, not retired).
- Running: code v2 (~84 %), SWE agentic (~60 %; pay held by an undecided submission).
- Hard math: env `reliquary_hard_math_v1` merged (environments #28, reliquary #335).
  nvidia/Nemotron-Math-v2 AoPS subset, 23,227 train problems, disjoint from DAPO.
  Left to launch: corpus image with the wheel (with 0xgrizz), miners on main >= #335
  with the wheel, a 200-problem pilot, then the job (n=4, 32k, cap 0.10). Keep only
  EOS-terminated traces at export (a 32k-truncated trace is read whole).
- DAPO-Math-17k stays reserved for the RL.

RL envs:
- Single-turn, ready: DAPO, code (OpenCodeInstruct), IF, logic, science, telecom.
- Terminal (TMax): validation ends ~07-10 20:00 UTC; SFT/RL split coded on
  environments `feat/reliquary-terminal-tmax` (`tmax_sft` / `tmax_rl`). An SFT terminal
  job costs 3-5 days of Catalyst work (grade the final box state, not a patch); the
  TMax paper saw no gain from SFT before RL. Proposed: all TMax to RL.
- SWE and terminal RL halves need phase 2 (multi-turn).
- Competitive code: NOT ready (07-10 audit): branch unpushed, dataset not built, no
  stdin mode in the gVisor worker, nothing registered, band not measured. ~6-8 days.
- This branch must merge main to pick up hard math and the new Teutonic profile
  digest (#335 changed it).

Proposed order: code phase 1, fix the env list (start single-turn and add multi-turn
envs live, which G allows), qualify on Teutonic (band per env, cooldowns, forced
seed), phase 4 short run; phase 2 in parallel, phase 3 after.

---

## A. How long an observation stays valid: DECIDED (07-10, revised)

Problem: the context hash includes the checkpoint, so observations, epochs, the panel
and the cooldown reset at every checkpoint adoption (about every window in RL).

Decision:
- Observations are a **run-wide log**, keyed by `(order, dataset, prompt)`, never
  reset. Each entry keeps `checkpoint`, `window` and a timestamp as attributes, not as
  part of the key.
- **The validator enforces no validity or exclusion rule** (no K, no blocking of
  "always solved"). **The miner chooses** which prompts to try, from the published data
  (see D).
- Only validator-side effects:
  - exploration is paid for the first scan of a prompt in the run only (see B);
  - the training cooldown of in-zone prompts stays (see E).
- A grading error or an incomplete group is not an observation.

## B. Paying out-of-zone groups (exploration): DECIDED (07-10)

- **Price: 15 % of a training group.**
- **Paid: the first miner to send a prompt never scanned before in the run.** A prompt
  already scanned (by anyone, at any checkpoint) earns nothing more as exploration.
- Cap: exploration ≤ 10 % of the window pool. **Unused budget goes back to training.**
  Nothing is burned and no fixed reserve is taken.
- Not paid: unusable groups (grading error, incomplete).
- To code: replace #327's fixed reserve b + divisor d + burned remainder with this model.

## C. Verifying exploration groups: DECIDED (07-10)

- **Training groups (in-zone):** full pipeline unchanged. Every group is proven,
  because its tokens enter the gradient, the trainer needs π_old from the proof, and
  forced-seed checks rely on it.
- **Exploration groups (out-of-zone, paid 15 %):** sampled TOPLOC audit, not trained.
  - **q = 15 % base** (same as the corpus). Extra proof load is about 0.67 × q ≈ +10 %.
  - **100 % audit for the first 100 exploration groups of a new hotkey.**
  - Audit draw by drand **after** submission (not predictable). Payment is held until
    the window's draw is known.
  - **Sanction** on a failed audit: forfeit all exploration earnings of the period, and
    no exploration for **24 h**.
  - A separate exploration proof budget, lower priority than training (never starves
    it).
  - Published observations carry `audited` / `pending` / `unproven`.
  - Always on, for free: validator-recomputed grading, prompt fidelity, length bounds.

## D. What is published to miners: DECIDED (07-10)

Static public files on R2, not an API that miners poll (0xgrizz's idea). The
validator's API is never hit by observation readers.

- **What:** for every verified group (training or exploration): env, prompt index,
  checkpoint, window, timestamp, **the 16 rewards** and the derived verdict (in-zone /
  16/16 / 0/16), the chosen candidate group (2×M seeds), and the status (trained /
  exploration paid / already scanned).
  - Rescans that are not proven are also published, marked **unproven**.
  - No hotkey, no tokens.
- **How:**
  - **Immutable segments** `observations/run-<id>/seg-NNNNNN.jsonl.gz`, one per
    **flush every 60 s**, cached indefinitely.
  - **Index** `observations/run-<id>/index.json`: segment list (seq range, window,
    checkpoint, sha256, URL), rewritten at each flush, short cache (~15 s), **signed
    with the validator hotkey**.
  - A miner reads the index every 15-30 s and downloads only new segments; at start it
    downloads the whole history.
  - Cost: about 3,000 PUTs a day; reads are cached and R2 egress is free.
- **Miner side:** a reference client keeps a local table prompt → observations. A
  default policy (skip 16/16, retry 0/16 after a few checkpoints) is overridable: each
  miner decides.
- Grizz's `/service-observations` becomes internal/admin. The run-wide log (A) feeds
  the segment writer.

## E. Cooldown from the in-zone rate: DECIDED (07-10), manual first

- **Start manual:** before the run we measure the in-zone rate ourselves (qualification
  sweep on the starting model), set a static cooldown per env, and have **a simple way
  to change it live** (an operator command or a registry field that takes effect
  without a restart; no redeploy).
- **An analyser recommends:** the validator computes the in-zone rate continuously from
  the run-wide log (A), counting **first scans only**, which are always reported since
  the first scan is paid in both lanes. It also tracks Q (groups actually consumed by
  the trainer, smoothed) and computes #324's formula
  `cooldown = N × p / Q × margin` (bounds, EMA, hysteresis). The result is shown on
  the dashboard/CLI as a **recommendation only**. A human applies it.
- The analyser never closes admission. With insufficient data it just says so.
- Later: an `active` mode that applies the recommendation by itself, switchable once
  its values have been validated over the run.

## F. Seeds 2×M (#325): KEEP

The miner chooses between 2 candidate groups of 16. This is intended: over-generation
pays off (the user is to supply the paper). Never allow choosing individual rollouts.

## G. Several envs per task, one checkpoint lineage: DECIDED (07-10)

- **One task = one checkpoint lineage = several envs**, like today's production RL task
  (per-env quota per pick, one optimizer step per pick, per-env pricing).
- The service contract lists the envs: dataset, policy (2×M seeds, exploration,
  cooldown) and quota for each.
- **Envs can be added or removed, and their quotas changed, live during the run**
  (same mechanism as E: an operator command, no redeploy).
- Everything decided in A-E applies **per env**: observation log, exploration, audit,
  cooldown recommendation.
- #322's per-task checkpoint isolation is kept. It separates **runs**, not envs.
- **Exploration cap per env: DECIDED (07-10).** 10 % of each env's share, not 10 % of
  the whole window, so one very out-of-zone env cannot drain the others.

## H. Truncation and `\boxed` (#328): DECIDED (07-10), one point to revisit

Today one truncated rollout rejects the whole group, on both lanes, and `\boxed` is
forced everywhere.

Decision:
- **Truncation:** go back to the current tolerance (`robust_utility_admits`). One
  truncated rollout no longer rejects the whole group.
- **`\boxed`:** forced **only for math envs**, never for the others.
- **To revisit:** whether `\boxed` should be forced even for math (user: "à voir même").
  Check what the math graders already require before coding the rule.

## I. Forced seed hard-on (#328): KEEP + qualify

Qualify on GPU with Teutonic before the run (agreement measured 0.897 on 09-20; floors
0.80/0.70).

## J. Production robustness (#327/#328): OPEN (left open on purpose, 07-10)

- A bad service archive makes the whole subnet abstain from setting weights. Abstain
  for that task only.
- `runtime.active()` fsyncs on every `/state`. Remove the fsync from reads.
- Declaring a service task breaks validators that have not been upgraded. Write an
  explicit deployment order.
- Possible gap between scheduler and batcher targets on post-PASS policy limits:
  verify.

## K. Offline curation and mapping (#320/#321): KEEP + guard

Keep it as an offline tool. Add a guard: never curate prompts from held-out eval sets
(BFCL, tau2, IFBench, GPQA, AIME, LCB, MMLU-Pro, Terminal-Bench, SWE-bench Verified).
Fix:
- circular verification for uploaded completions;
- "generation verified" being nearly unreachable under partial audit;
- a row with several groups selected as soon as one group matches;
- a crash on a blank line.

## Phase 1 implementation rulings: CONFIRMED by the user (10-08)

Rules the phase 1 plan had to settle, each confirmed:

1. **Sanction period.** The "period" forfeited on a failed audit is the window being
   settled. A window already paid cannot be clawed back; the 24 h exploration ban
   covers what comes after.
2. **Audit timing.** The proof plane only holds the current checkpoint, so an
   exploration group is audited inside its own window. A group drawn for audit but
   not audited before the window closes is **not paid and not sanctioned**.
3. **Adding an env live.** Only an env **already declared in the profile at launch**
   can be switched on. An env's "quota" is its emission share; 16 groups per env per
   pick stays a protocol constant. Multi-turn envs (SWE, terminal) must therefore be
   declared at launch and activated when ready.
4. **`\boxed`.** Math: a missing box makes the rollout "uncertain" and does not
   reject the group. Science: a missing box is a plain 0.
5. **Full pool: proportional split.** A training group counts 1 share, an exploration
   group 0.15 share. Price of a share = `P / max(T, total shares)` per env, where T
   is the env's training slots in the window.
   - Window not full: everyone is paid full price and the rest is burned, as today.
   - Window full with exploration: **everyone is scaled by the same factor**,
     explorers included (at most about -9 %, since exploration is capped at 10 % of
     the pool). Nobody is paid first.
6. **Cooldown delay.** A cooldown change applies from the next window, which can be
   up to 6 h later under fill-closed.

Needed from the user before the run (not for coding): a public R2 bucket for the
observation files, an admin token for the internal endpoint, and the GPU
qualification of forced seeds on Teutonic.

## L. Multi-turn episodes: phase 2

The service stack refuses episodes. Spec to write: one RL group = 16 signed sandbox
sessions bound to one precommit; reward per episode from the final record; in-zone
over the episodes; TOPLOC on the model spans; observations masked in the trainer
loss; payment and cooldown per episode group.

## M. Async windows: phase 3

`PIPELINED_WINDOWS` is refused, and only the current checkpoint is accepted. First
measure the fraction of tokens outside the clip at a lag of 1/4/8/16 steps. Then
design bounded staleness (MiMo: 4) and possibly partial rollouts.

## Also pending (outside phase 1)

- SWE prod hotfix: the agent's diff can modify untracked/ignored files (train and
  polyglot).
- Make `reliquary-sandbox` public, or vendor attest/observation/client/bridge.
- Image registry for the tmax base.
- Terminal uid separation (17/64 MiMo rows served).
- Per-CPU capacity in the fleet.
- Early release of sessions that never opened a box (needs a heartbeat that lists
  open sessions).
- The validator serves its own hotkey to miners (removes `--validator-hotkey`).
- Catalyst caplog tests silenced by bittensor (pre-existing).
- The fill-closed profile must accept Teutonic 9B.
