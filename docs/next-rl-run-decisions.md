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

## C. Proof cost of exploration: OPEN

Today every exploration group is fully proven. At 32k tokens on Teutonic that is
about 20-30 s of H100 per group. Cheap no-signal groups can starve the proof plane.

Recommendation: sampled TOPLOC audit, as in the corpus, instead of a full proof. A
separate exploration proof budget that can never starve training.

## D. What is published to miners: OPEN

Today `/service-observations` exists, resets at every checkpoint, and no miner client
reads it.

Recommendation: per prompt, publish the verdict (in-zone / always solved / always
failed), the checkpoint and the age, immediately. Write a miner client that skips
excluded prompts.

## E. In-zone rate for the cooldown (#324): OPEN

Today the panel is an operator-supplied JSON file, unsigned. A bad panel closes
admission.

Recommendation:
- The validator draws a random prompt sample itself (drand) and measures the rate
  from observations (exploration + training).
- A bad panel falls back to the static cooldown and never closes admission.
- Guard Q (consumption) against miners who under-supply on purpose.

## F. Seeds 2×M (#325): KEEP

The miner chooses between 2 candidate groups of 16. This is intended: over-generation
pays off (the user is to supply the paper). Never allow choosing individual rollouts.

## G. One env per task with its own checkpoint lineage (#328): OPEN, big change

This makes a mixed-env Teutonic run impossible (each env would train its own model).

Recommendation: several envs per task, with one shared checkpoint lineage.

## H. Truncation and `\boxed` (#328): OPEN

Today one truncated rollout rejects the whole group, on both lanes, and `\boxed` is
forced everywhere.

Recommendation: go back to the current tolerance (`robust_utility_admits`), and force
`\boxed` only for math.

## I. Forced seed hard-on (#328): KEEP + qualify

Qualify on GPU with Teutonic before the run (agreement measured 0.897 on 09-20; floors
0.80/0.70).

## J. Production robustness (#327/#328): OPEN

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
