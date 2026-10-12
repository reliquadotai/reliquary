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
- Scan rules in force:
  - A prompt with a training observation is scanned: exploration on it earns nothing, whoever saw it first.
  - A proven training group counts as a scan whatever its pay; only the groups of an aborted window do not.
  - Exploration on a prompt trained in the same window is unpaid (published `exploration_unpaid`, reason `trained`).
  - That case carries no sanction: no ban, no forfeit, no probation lost.

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
    Graded (user decision 10-09): only a DETERMINISTIC forgery (proof/TOPLOC mismatch, token
    authenticity, forged termination, forced-seed hard CDF mismatch) bans for 24 h. A
    STATISTICAL-only failure (threshold checks) forfeits the same earnings, does not ban, and
    puts the hotkey back at 100 % audit until 20 new audits pass; it bans only when more than
    3 of its last 20 NON-FORCED (base-rate drawn) audits failed statistically (to recalibrate at
    qualification from the measured honest false-reject rate). An unclassified stage counts as
    statistical. Amended 10-09: the class is that of the STRONGEST failure (an audit that fails a
    statistical gate is re-proven with the statistical gates off; any failure there is
    deterministic), and any failure, a deterministic one included, restarts the 20-pass
    re-probation (a hotkey back from a ban is audited at 100 %). Missing episode replay spans
    are no verdict (validator_lost).
  - A separate exploration proof budget, lower priority than training (never starves
    it).
  - Published observations carry `audited` / `pending` / `unproven`.
  - Always on, for free: validator-recomputed grading, prompt fidelity, length bounds.
  - Audit rules in force:
    - Probation counts passed audits: a hotkey stays at 100 % until 100 of its exploration groups have passed.
    - A hotkey's not-drawn groups are paid only if its own drawn groups up to that round were audited; another hotkey's groups are never affected.
    - Audits of hotkeys past probation run before audits of hotkeys in probation.
    - Queued audits drain at seal, before the window is finalized; a group drawn but not audited is unpaid, never sanctioned.

## D. What is published to miners: DECIDED (07-10)

Static public files on R2, not an API that miners poll (0xgrizz's idea). The
validator's API is never hit by observation readers.

- **What:** for every verified group (training or exploration): env, prompt index,
  checkpoint, window, timestamp, **the 16 rewards** and the derived verdict (in-zone /
  16/16 / 0/16), the chosen seed indices (M of the 2×M pool seeds, in the order of the rewards), and the status (trained /
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

## F. Seeds 2×M (#325): KEEP, corrected by the user on 10-08

- For each (env, prompt, window) the pool is 2×M public seeds (32 when M=16), the same
  for every miner. There are no fixed candidate groups.
- A miner builds its group from ANY M distinct seeds of the pool. Cherry-picking is
  intended: miners over-generate and keep the subset they like (the user is to supply
  the paper).
- On the wire a group is the pool digest plus the chosen seed indices in increasing
  order; rollout i is the i-th chosen seed, so one subset has one encoding.
- A seed's forced draw depends on (pool, seed, token position) only, never on its rank in
  the group nor on the hotkey: the same seed gives every miner the same draw.
- The group id is the digest of that selection. Public observation events list the chosen
  seeds next to the per-rollout rewards, so miners learn which seed gave which reward.
- Prompt dedup and cooldown rules are unchanged.

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

   **Amended (phase 2, multi-turn RL).** A signed-sandbox episode environment served through the
   verifiers bridge can also be added to a running run that launched with one: a schedule request carries
   its entry and its public catalogue (package, version, manifest digest, image lock, RL split; its share is
   the schedule's) and names the operator's conformance report. The validator activates it only when the
   package is on its owner's allowlist, it has the env installed at that identity without moving any active
   environment's install, at least one sandbox machine serves the package with enough of its locked images
   and enough servable tasks, and the report is a passing gate of exactly that env, run under the same image
   lock. It takes effect at the next window. A new version is a new environment name, a new distribution
   name and a new sandbox environment name, whose tasks start never scanned; the old one is switched off by
   the schedule. Its turn and token limits stay within the launch environments'. Single-turn environments
   are still declared at launch.
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
- Per-CPU capacity in the fleet: done in phase 2. Sandbox machines reserve each session's agent box in CPU and memory and run grading boxes in a grading pool; RL sessions are placed only on machines that report these resources.
- Early release of sessions that never opened a box (needs a heartbeat that lists
  open sessions).
- The validator serves its own hotkey to miners (removes `--validator-hotkey`).
- Catalyst caplog tests silenced by bittensor (pre-existing).
- The fill-closed profile must accept Teutonic 9B.

## Operating multi-turn RL capacity (phase 2)

How the validator shares the sandbox fleet between RL miners, and what the operator does. The constants
named here live in `reliquary/sandbox/rl_capacity.py` unless another module is given. They are starting
values, to be tuned at qualification (see the last list).

### Sandbox machines

- **Resources.** RL machines set `episode_cpu_millis_total`, `episode_memory_bytes_total`,
  `episode_cpu_overcommit` (measured at qualification) and the grading pool
  (`episode_grading_cpu_millis_total`, `episode_grading_memory_bytes_total`). A machine without them, or
  without an `episode_present_window_s` of 1 to 120 s, gets no RL session (the validator logs one ALERT per
  machine). Set each machine's session slots (`episode_capacity`) to what its CPU and memory reservations
  hold, and `episode_present_window_s` to 120.
- **One issuer per machine.** A machine that serves RL sessions is not also placed on by a corpus validator:
  corpus and RL fleets are separate machine sets.
- **Per-task images.** For an environment with one image per task, the validator accepts a precommit only for
  a task some serving machine could take (its image held, its box and memory available). Pre-pull the images
  of the tasks you mean to serve, by digest (an image loaded from an archive has no registry digest and never
  counts as present), and keep that set dense: miners do not see it and draw tasks blindly, each refused
  precommit costing them one of their precommits per minute. A machine listing many thousands of images
  sends a large capacity report; watch its size when the set grows.
- **409 or 503.** A precommit is refused `task_unservable` (409, final) only when the validator's view of the
  fleet is fresh and complete and no serving machine, or set of machines, can host the task's group. Any view
  involving staleness (no fresh serving machine, a placement it cannot judge, a group reachable only through
  stale or unreadable machines, a task resolve that timed out) is answered 503 (`fleet_unavailable`,
  `environment_unserved`), which miners retry.
- **Precommit cost.** Each new precommit costs the validator one dry-run database read (in a thread), one
  task resolve (cached; a cold resolve is bounded by the source's I/O timeout) and a lookup per machine.
  Out-of-range task indices are refused by the dry run before any resolve. This path runs outside the
  pre-authentication limit: its concurrency is bounded only by the registered hotkeys' precommit rate
  (30 a minute each). Size the validator host for that rate times the hotkeys expected.
- **Homogeneous fleet.** The quota counts the whole fleet. A miner's sessions on one environment are capped
  at its allocation times that environment's share of the fleet (`quota_env`, 429, retryable, never held
  against the miner); this cap applies to established hotkeys only. Prefer machines that serve every episode
  environment with all of its images, and keep each per-task image on machines that can host a whole group
  (M sessions) at once: otherwise its tasks are not offered.

### Starting the validator

- **Pinned sandbox.** The validator refuses to start signed-episode RL, with an ALERT, when the installed
  reliquary-sandbox is not the commit this build pins (`SANDBOX_COMMIT` in `reliquary/sandbox/__init__.py`): an
  older one cannot verify gateway-signed refusals, which would then be charged to miners. Sync the validator's
  environment from the pinned dependency before starting it.
- **Degraded start.** At start the validator waits up to `BOOT_FLEET_WAIT_S` (120 s, in
  `reliquary/validator/rl_sandbox_wiring.py`) for a machine serving each episode environment. An environment
  still unserved is started degraded: refused retryably (`environment_unserved`) and logged as an ALERT; the
  validator serves everything else and serves it as soon as a machine reports it. Each unserved environment
  can add up to 120 s to the start, and each environment's reservation tier is seeded from a sample of its
  tasks (`TIER_SEED_TASKS`, at most `TIER_SEED_TIMEOUT_S` = 60 s per environment). Nothing else (HTTP server,
  single-turn windows, weights) runs before this ends; it shows as `startup_stage=episode_services`.
- **Added environments at start.** After a restart, the validator builds the environments added to the run
  and waits at most `BOOT_ADDED_BUILD_TIMEOUT_S` (900 s, in `reliquary/validator/service.py`) for those
  builds, in total. A slow dataset download can therefore delay a start by up to 15 minutes. An active added
  environment that is not built by then is switched off (see "Automatic switch-off").
- **Fail-closed.** If every active environment is an added one and none of them can be built or served, the
  validator refuses to start (at a window boundary, it retries the window). Keep at least one launch
  environment active, or expect a restart loop while the only active environment cannot be built.

### Tokens

- **Token validity.** A session token is valid for its open window plus the episode's wall time, the window
  lengthened to at least the task's prepare timeout + 120 s (the gateway's preparation slack and the token's
  transit). With the default prepare timeout of 600 s, an operator who sets `RELIQUARY_SANDBOX_OPEN_WINDOW_S`
  below 720 s gets tokens longer than that window, corpus tokens included: set it to at least prepare + 120.
- **Activation uses the environment's own preparation.** When an environment is added, the machines are
  checked for the token the issuer will give its tasks: the largest prepare timeout among 256 tasks spread
  evenly over its RL split (`ACTIVATION_PREPARE_SAMPLE` in `reliquary/validator/catalogue_activation.py`; a
  task that declares none counts the default). A sampled task whose prepare timeout cannot be read now makes
  the request wait. A task outside the sample that declares a longer preparation is never placed on a machine
  whose `max_token_validity_s` does not cover its token: when no serving machine covers it, its precommits are
  refused `task_unservable` (409, final) and miners lose those precommits without seeing why. Keep the
  machines' `max_token_validity_s` well above the longest preparation the environment declares.
- **Present window.** RL tokens must be presented within 120 s (`episode_present_window_s`): a machine
  unreachable for longer makes its miners' tokens unusable, and those opens count as the miners' waste. Keep
  RL machines reachable, and before maintenance take a machine out of the directory's active machines (it then
  gets no new session) and let its sessions end.
- **Directory outages.** A failed open whose gateway receipt cannot be checked (the machine directory is not
  readable) is refused retryably, and the miner resends it for about a minute. An outage longer than that
  leaves the session live until its token expires: the receipt is never credited and the session lapses.

### Quota

- **Sharing.** RL sessions are shared per hotkey by weighted water-filling: a hotkey's weight is the useful
  session time it produced in the last 24 h times a yield multiplier, and it gets what it currently asks for up
  to its weighted share, the rest going to those that want more. New hotkeys (fewer than `YIELD_MIN_DECIDED`
  decided sessions) share a guaranteed `NEWCOMER_SHARE` (a tenth) of the fleet and, with
  `NEWCOMER_TAKES_SPARE`, spare capacity; hotkeys at or below the yield floor share a hard cap of
  `LOW_YIELD_SHARE` (3 %). Registering more hotkeys buys nothing beyond the newcomer pool. Failures of the
  machines or of the validator, and refusals the gateway signed as its own, never count against a miner.
- **Fleet bound.** A new group starts only while the committed sessions plus one group (M) fit the active
  environments' fleet, so a synchronized turnover of every group cannot overcommit it. What a hotkey held is
  remembered as demand for `DEMAND_MEMORY_S` (600 s), decaying linearly, so a key does not lose its share
  between two groups. A restart empties this memory.
- **Tiny fleets.** The newcomer and low-yield pools are each at least one seed pool (P = 2 M sessions). On a fleet of a
  few hundred sessions they take a large slice before established hotkeys; below one group, low-yield hotkeys
  could take the whole fleet.
- **Pools while groups wait.** A group that is asked for and not started counts its whole seed pool (2 M) of
  demand. A newcomer waiting on its two groups therefore shows 64 sessions instead of 32, so the newcomer
  pool's allocation, and the room kept for it from established hotkeys, double while its groups wait (bounded
  by the pool plus spare). The newcomer pool can look reserved but idle for up to `DEMAND_TTL_S` (30 s) per
  burst of asks. A hotkey that just graduated keeps a grace period during which it waits for the newcomers'
  room; that wait doubles in the same way, so graduates start later when many genuine newcomers wait.
- **Rounding.** The number of groups a miner is told is its potential divided by the pool size, rounded up: a
  potential just above one group tells 2 groups. The second group can start only once the first shrinks;
  until then its opens are refused `quota_live` (retryable), about one try every 10 s. Harmless, but it shows
  in the refusal counts.
- **The reference miner** runs as many groups as `POST /rl/sessions/allocation` (signed by its own hotkey)
  says, never more than its own `--groups-in-flight` ceiling and never fewer than one. It keeps at most
  `RL_OPENS_IN_FLIGHT` (16) opens in flight per process and `RL_OPENS_SIDE_BY_SIDE` (4) per group
  (`reliquary/miner/rl_episode_client.py`), and pauses an environment for `ENV_PAUSE_S` (60 s,
  `reliquary/miner/episode_mining.py`) after a `quota_env` refusal.
- **Every miner: play from M.** Once a group has M graded, usable episodes, a `quota_live` or `quota_pool`
  refusal of a further seed means there is no room beyond M: play from those M and stop asking for seeds.
  Waiting for more keeps the group's graded sessions counted, so groups at M wait on each other until the window
  ends (the reference miner does this: `RL_OVERGENERATION_REFUSALS`).
- **Transcript checks.** At most `MAX_CHECKS_RUNNING` transcript checks run at once for all miners, and a
  hotkey takes at most one less, so one operator's groups wait on its own checks. Size it from the measured
  check time at qualification.

### Watching and growing the fleet

- The fleet is fixed and grown by hand: watch `GET /rl/saturation` (bearer token
  `RELIQUARY_RL_SATURATION_TOKEN`, or the service observations' token when unset) and the ALERT "rl sandbox
  fleet saturated"; a machine added to the directory is used at the next directory refresh and poll, without
  a validator restart.
- There is an ALERT per environment (or image set) that loses every serving machine, one when a hotkey's
  capacity verdicts pile up, and one when transcript checks are refused for being busy.

### Validator restarts

- An open whose answer a restart lost is not charged to its miner when it was issued in the minute before the
  validator's last activity, no machine report listed it live, and the miner resends it after the restart. A
  validator outage longer than the miner's give-up time (a few minutes of unreachable resends) loses these
  opens.

### Grading pool

- A grading the pool could not start in time is aborted by the machine; when other miners' gradings held the
  pool it counts toward no abort cap. Size the pool at qualification for several miners' bursts at once so
  this stays rare (the signed report's `grading_pool.timeouts`): a miner holding exactly half of a small pool
  is not charged for its aborts, but two hotkeys of one miner can still make each other's aborts look like
  another miner's.

### Adding an environment to a running run

1. **Gateways first.** A new environment, or a new version (a new environment, distribution, import module and
   sandbox environment name), is first served by the gateways: add its sandbox environment name by a rolling
   restart, with exactly the options of its catalogue entry, and pre-pull its images by digest. Submit `add`
   only once that rollout and the pre-pull are done: activation counts the machines that serve it now, and a
   fresh, complete fleet view in which no machine serves it yet refuses the request.
2. **Install it on the validator.** Before installing, check that the wheel's top-level import module is its
   distribution name with `-` replaced by `_` and is not the module of an installed environment (the wheel's
   `*.dist-info/top_level.txt`, or `python -m zipfile -l <wheel>`): two distributions shipping one module
   overwrite each other's files. Install it with `pip install --no-deps` and run `pip check`; if `pip check`
   fails, do not add the environment (never install its missing dependencies into the validator's
   environment: uninstall it). New files under a module this process already loaded at another identity need
   a validator restart.
3. **Allow it.** List its `name==version` in `episode-packages.allow` in the service folder (owner-reviewed
   packages only: its code runs in the validator process; the validator does not even import a package that is
   not listed).
4. **Gate it.** Run the sandbox's conformance suite as a gate, with the env's served options and RL split. The
   report records the image lock it ran under; the validator accepts it only when that lock is the catalogue's
   `image_lock_sha256` (a report without one, from an older sandbox, is refused).
5. **Describe it** with `python -m reliquary.environment.bridged_episode describe` (see
   `docs/runbooks/rl-bridged-episode-envs.md`), then add it **inactive** with
   `python -m reliquary.services.schedule add NAME --entry ... --attestation ...`. The validator builds it as
   soon as it sees the request, and checks it again at the window boundary. Request ids beginning `auto-` are
   reserved for the validator and refused (the CLI never writes them).
6. **Waiting.** A check the validator cannot settle now (a stale or incomplete fleet view, a transient
   download or network error, a build still running) leaves the request pending with status `waiting` and the
   reason (`show`). It waits at most `RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S` (default 3600 s, counted from its
   first wait and kept across restarts), then is refused; resubmit it once the cause is fixed.
7. **Miners.** Miners mine a bridged environment from its public catalogue: they install exactly the pinned
   env package (`pip install --no-deps <wheel>`, the identity the order names) and the sandbox bridge. The env
   wheels and the bridge must therefore be published before launch: miners cannot install a private
   `reliquary-sandbox` (see "Make `reliquary-sandbox` public, or vendor ..." under "Also pending"). A miner
   binds its prompts when it starts, so for an environment added during the run it must wait until the
   announcement names it, install the package, add the name to `--episode-envs` and **restart**; nothing
   restarts it automatically.
8. **Switch it on.** Once miners see it announced, `python -m reliquary.services.schedule set --active ...
   --share ...`. It takes effect at the next window.

The SWE catalogue entry has not yet been produced from the real package (describing it needs a host with
`swebench` installed). Produce and review it there (its identity, its rows, and the lock coverage of a sample
of its task images) before qualification or any production run that serves SWE.

### Automatic switch-off

- One added environment never stops the windows or the start. The validator checks an `add` request with a
  dry run before committing it, so a refusal changes nothing. When an added environment that was already
  committed then fails here (its build or installation fails at a window boundary, or at a restart it is not
  built within the boot wait or the services refuse it), it is unloaded and an ALERT is logged. If it was
  active, the validator switches it off itself: it writes the next schedule revision N with the same active
  environments less that one, their shares renormalised in the proportions they had, under the request id
  `auto-deactivate-r<N>`.
- The validator then rebuilds the environment in the background, after 60 s and doubling after each failed
  attempt, up to once an hour (`ADDED_RETRY_MIN_S`, `ADDED_RETRY_MAX_S`). It does not switch it back on: once
  its ALERT clears and it is served again, the operator runs `set --active` again.
- A pending operator request written for revision N, which the switch-off took, can never apply: it is
  refused with a message saying so. Resubmit it against revision N+1.

### Constants to tune at qualification

- Miner: `RL_OPENS_IN_FLIGHT` (16 opens in flight per process) and `RL_OPENS_SIDE_BY_SIDE` (4 per group);
  `ENV_PAUSE_S` (60 s pause of an environment after `quota_env`).
- Quota: `DEMAND_MEMORY_S` (600 s; 120 s is the alternative if the memory holds idle room too long),
  `NEWCOMER_SHARE`, `LOW_YIELD_SHARE`, the graduation grace (`GRACE_SESSIONS`) and `NEWCOMER_TAKES_SPARE`
  (True: newcomers may use spare capacity, the operator grows the fleet by hand).
- Additions: `RELIQUARY_SCHEDULE_ADD_WAIT_MAX_S` (3600 s), `BOOT_ADDED_BUILD_TIMEOUT_S` (900 s),
  `ADDED_RETRY_MAX_S` (1 h).
- Checks: `MAX_CHECKS_RUNNING`, from the measured transcript check time.
- Machines: `episode_cpu_overcommit` and the grading pool's size.
