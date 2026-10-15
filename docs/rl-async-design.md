# Async RL with leased prompts and a Gumbel forced draw: design

Status: DRAFT for review (2026-10-15). Supersedes the window redesign discussion in
`docs/rl-windows-redesign-notes.md`. Evidence: `docs/rl-off-policy-tests.md` and the
campaign reports under `~/offpolicy-lab/results/` (fairness, staleness, red team,
literature, weight-sync cadence, 2B mini-training of 2026-10-15).

Tags: **DECIDED** (user agreed), **PROPOSED** (recommended, awaiting the user),
**PENDING** (waits on a measurement).

## 1. Decisions in one page

1. **Fully asynchronous** (DECIDED). No windows, picks, fill-closed rounds or trainer pacing.
   Miners generate continuously on whatever recent checkpoint they hold; the trainer takes a
   step whenever its buffer holds a batch.
2. **One verification: the forced draw** (DECIDED). Every token is the validator's own
   Gumbel-max pick for a public noise stream. A rollout is accepted when its match rate
   against the latest checkpoint is above a floor. No TOPLOC, no trust in miner logprobs, no
   old checkpoints on the validator.
3. **Gumbel-max draw**, placed by the vLLM runner's own token count (DECIDED), with an fp32
   LM head in the reference miner and on the validator (DECIDED).
4. **Leases** (DECIDED). A miner reserves prompts before generating. The draw's nonce depends
   on (validator secret, prompt, checkpoint) alone, so a lease can neither be pre-generated
   nor re-rolled. Every lease ends in a submission or an expiry; there is no release. The
   public seed pool (choose M of 2xM) is removed.
5. **Quota measured, not declared** (DECIDED). Concurrent leases per hotkey follow measured
   throughput and latency (Little's law), with slow start and halving on expiry.
6. **Validator grades everything; trains on what is in zone** (DECIDED). Out-of-zone groups
   are graded, stored, audited by sampling, and paid 25 % of an in-zone group on a prompt's
   first scan.
7. **Prompt lifecycle** (DECIDED): never scanned = leasable; in zone = trained, then cooldown;
   too easy = blacklisted for the run; too hard (e.g. 0/16) = cooldown, then leasable again.
   All of it readable by miners.
8. **Loss: substitution** (DECIDED). At each completion position the trainer scores the
   validator's pick v* (the miner's token where they agree). Drift is exactly zero for any
   miner hardware.
9. **Training forward = verification** (PROPOSED). With the trainer holding exactly the
   published weights, one forward gives v*, the match rate and the loss.
10. **One checkpoint per optimizer step, shipped as sparse deltas** (PROPOSED, PENDING
    measurement), full anchors every N steps, full backups to Hugging Face in the background
    (DECIDED). Miners download through presigned R2 URLs (DECIDED).
11. **Pay per matched token** of trained groups, with a per-rollout floor so that quantized
    generators do not out-earn honest bf16 ones. Per-environment prices steer supply to the
    trainer's target mix.

## 2. Evidence behind the decisions

Rows 1-10 on the RL base `ReliquaryForge/teutonic-i-attention-graft-healed-v1` (9B, T=1
full vocab), vLLM 0.30.0 miners, HF proof path validators.

| Question | Result |
|---|---|
| Honest match, inverse-CDF (today), bf16 heads | 0.944-0.950 on 3090/5090/A6000/L40, IF env ~0.87 |
| Honest match, **Gumbel**, bf16 heads | **0.986** on all cards (spread 0.0006), IF 0.984, rollout min >= 0.95, 0 groups rejected |
| Honest match, Gumbel, fp32 heads both sides | 0.994 |
| Validator hardware | changes pooled match by < 0.0005 |
| Desync bug (counter-based processor) | 21-47 % of honest rollouts on preemption; runner-keyed position: 0 bad over 38M+ positions, 2,000+ resumes; production port passes async on/off/multiprocess |
| Why inverse-CDF disagrees | CDF geometry (low-p tokens mid-vocab), no token-specific bias; gone under Gumbel |
| Value of disagreement positions (branching, 2,450 positions) | not higher than entropy-matched controls (abs Q gap 0.096 vs 0.103), signed gap +0.004 |
| Drift (constant advantage), staleness 1/4/8 | current loss 59k / 418k / 900k (cosine 1.0 with "SFT toward the sampler"), substitution 0 |
| Staleness under Gumbel | match 0.987 / 0.983 / 0.982 / 0.970-0.978 / 0.949-0.970 at lag 0 / 1 / 2 / 4 / 8 |
| Quantized miners vs bf16 validator, Gumbel | fp8 KV 0.985 (undetectable), fp8 weights 0.963-0.971, W4A16/NVFP4 ~0.855 |
| Trusting declared logprobs | not needed; red team: small gain, residual coordinated-Sybil risk |
| **2B mini-training** (Qwen3.5-2B, GSM8K, 120 steps, 2 seeds; sampler with noisy weights at match 0.90 + lag up to 3) | no correction: sampler-trainer KL 0.022 -> 0.15 (doubles every ~40 steps), entropy 0.40 -> 0.22; substitution and the paper's MIS+SC with the true q: KL flat at 0.022, entropy 0.34, same evals within noise. Uncorrected run scores higher at 120 steps (sharpening on an easy task). 400-step runs and an lr 5e-6 control in progress |
| Weight-sync practice (literature) | labs sync every 1-16 optimizer steps and tolerate 4-8 policy versions with an off-policy-aware loss; 0.6-1.1 % of bf16 weights change per step (with fp32 masters), lossless deltas 60-100x smaller; Grail (SN81) ships deltas + an anchor every 50 on R2 |

## 3. Flow

```
miner                           validator (leases, grading, trainer)
-----                           ------------------------------------
GET /rl/prompts/state  ------>  blacklist, cooldowns, scan history
GET /rl/quota  -------------->  quota(hotkey)
POST /rl/lease [P1..Pn]  ---->  exclusive leases, nonce_i = H(S_k, P_i, k)
                 <-----------   {lease_id, nonce, k, expires_at}
generate 16 rollouts per P
  (Gumbel draw from nonce, runner-keyed, fp32 head)
POST /rl/submit  ------------>  grade all rollouts (CPU)
                                in zone  -> buffer
                                out zone -> store; sampled audit forward vs latest
                                trainer: forward on a batch = v*, match, loss
                                  rollouts under the floor masked, groups judged, step
                                publish delta k+1 (+ anchor every N, HF backup every M)
GET /rl/checkpoint  --------->  index (signed) + presigned R2 URLs
                                reveal S_k when k retires
```

## 4. Components

### 4.1 Forced draw (protocol)

- **Gumbel-max:** pick = argmax_v (log p_v + G_v), G_v = -log(-log U_v), U_v from a
  counter-based PRNG keyed by (nonce, rollout index, position, v). splitmix64 + float64 G is
  bit-identical on Ampere, Ada, Hopper, Blackwell. It is an exact sample of p; diversity is
  unchanged.
- **Position:** `num_tokens_no_spec - num_prompt_tokens (+ base_offset)` from the vLLM input
  batch, params keyed by request id (`feat/forced-seed-runner-fix`, validated). vLLM pinned
  to 0.30.x; speculative decoding refused.
- **fp32 head:** logits kept in fp32 at the LM head output on miner and validator (match 0.986
  -> 0.994).
- **Verification statistic:** per-rollout match on stochastic positions (max p < 0.99) and
  pooled per-hotkey statistics.
- Protocol version bump: Gumbel and inverse-CDF are incompatible; validators first.

### 4.2 Leases

- **Request:** `POST /rl/lease {env, prompt_ids[], checkpoint k}`; batched requests allowed.
- **Exclusivity:** one open lease per (prompt, k). A prompt submitted at k is closed for k
  (the draw would be identical anyway).
- **Fairness:** requests are collected in short intervals (~1 s); conflicts inside an interval
  are resolved at random, so network proximity buys nothing. A refused request returns only
  "taken"; open leases are never listed.
- **Nonce:** `H(S_k, prompt, k)` with S_k a validator secret committed when checkpoint k is
  published and revealed when k is retired, so anyone can audit that nonces were
  deterministic. Same prompt + same checkpoint = same draw for every lease: no re-roll, no
  pre-generation.
- **Mandatory submission:** a lease ends only by a submission (any outcome) or by expiry.
  Miners select prompts before seeing results, never after.
- **Expiry:** fixed per environment: twice the time a 3090 needs for 16 rollouts at max length,
  measured at qualification (PROPOSED). An expired lease's draw goes to the next lessee unchanged (accepted; throwaway
  keys have tiny quotas and expiry earns nothing).

### 4.3 Quota

- Target concurrency = measured throughput (groups/min) x measured lease-to-submit latency x
  1.5, per hotkey, over a sliding window.
- New hotkeys start at 4 concurrent leases (PROPOSED). While a hotkey uses its full quota and
  delivers on time, the quota doubles (slow start). Every expiry halves it. Out-of-zone
  submissions count as delivered work.
- Published per hotkey with the reasons. Registration burn applies.

### 4.4 Submission and grading

- **Grading:** every group is graded by the validator (rewards never come from the miner).
- **In zone** (sigma >= 0.24, at most one truncated rollout, current truncation rule and
  `unproven` lane): goes to the buffer; verification happens in the training forward (4.7).
- **Out of zone:** stored with grades. Audit regime from `next-rl-run-decisions.md` §C:
  100 % of a hotkey's first 100 out-of-zone groups (probation), then a random fraction drawn
  after submission (15 %, user 2026-10-15), graded sanctions (deterministic forgery bans, statistical
  failure re-opens probation). An audit is a separate forward against the latest checkpoint
  with the same floor as training (PROPOSED: no old checkpoints kept). Upload carries tokens
  only, compressed.
- Group integrity: all 16 rollout indices required; the draw is bound to the rollout index.

### 4.5 Prompt lifecycle and miner data

- **Never scanned:** leasable by anyone.
- **In zone:** trained and paid in full; the prompt enters the training cooldown, counted
  in checkpoints.
- **Out of zone, too easy:** blacklisted for the rest of the run, never leased again.
- **Out of zone, too hard** (e.g. 0/16): same cooldown as in-zone prompts, then leasable again.
- Binary rewards: out of zone means 16/16 (blacklist) or 0/16 (cooldown). Continuous rewards
  with too little variance: mean >= 0.5 blacklist, else cooldown. Groups out of zone for
  truncation: cooldown (user, 2026-10-15).
- **Pay:** an out-of-zone first scan earns 25 % of an in-zone group, after audit (replaces
  the 15 % of the decisions doc §B). Later rescans of a too-hard prompt are unpaid when out
  of zone.
- **Data for miners:** every prompt's state (never scanned, cooldown until k, blacklisted)
  and its scan history (16 rewards per scan, checkpoint, verdict; no hotkey, no tokens) live
  in the lease DB. Exposed as `GET /rl/prompts/state` (current state, incremental) and as
  immutable segments in the R2 bucket (decisions doc §D), which double as the public
  difficulty map.

### 4.6 Staleness and floor

- Miners generate on any checkpoint they hold; the lease records k; verification is always
  against the latest checkpoint. Staleness costs match, hence pay, and is self-priced.
- **Floor:** 0.95 per rollout with bf16 heads (about 0.96 with fp32 heads, to calibrate at
  qualification). Keeps every honest rollout at lag <= 2, rejects all W4A16. fp8 weights sit
  around 0.95: a pricing question, not a floor question.
- **Rollout-level rule:** a rollout under the floor is excluded and unpaid; the group is
  judged on its pooled match (one bad rollout must not kill the group: x16 compounding).
- **No quality lag cap** (user, 2026-10-15): the floor alone rejects rollouts that are too stale.
- **Leases only on checkpoints whose secret S_k is unrevealed** (DECIDED). Once S_k is public
  anyone can compute every nonce of k, pre-generate and lease only the prompts that land in
  zone. S_k is revealed late (e.g. 16 checkpoints after k), so the rule never binds an honest
  miner. Publish match by declared lag.

### 4.7 Buffer, trainer and loss

- Accepted in-zone groups enter a buffer tagged with env, checkpoint and arrival time. The
  trainer steps as soon as it can fill a batch at the target env mix; a short env does not
  block, its price rises. Within an env, freshest first. Groups that wait too long fail the floor
  against the newer checkpoint and are dropped unpaid.
- **One forward, three outputs** (PROPOSED). The trainer holds exactly the published
  checkpoint k. Its forward over a batch gives, per completion position, the logits, hence
  v* = Gumbel pick with the lease's noise, hence:
  1. the match rate per rollout (verification);
  2. the substitution loss `-A log p_theta(v*)`;
  3. telemetry (match, substitution rate, entropy, drift estimate).
  Rollouts under the floor and groups that fail are masked out of the loss before the
  backward. The separate proof forward remains only for out-of-zone audits.
  - Needs: the trainer forward's logits (packed, FSDP2, bf16 compute, fp32 head) must give
    the same picks as the batch-1 proof path; measured cross-hardware spread is < 0.0005,
    to re-measure on the trainer.
  - A group is accepted, and paid, when it is trained, not when it is submitted.
- **Loss details:** advantages as today (group-normalised, token-level normalisation,
  degenerate groups dropped, soft overlong penalty, no KL). With one step per checkpoint the
  PPO ratio is 1 and the clip is inert; with N > 1 steps per checkpoint the ratio is taken
  against the published checkpoint, which the validator knows exactly.
- Loss switchable by config (substitution / current) as a guard.

### 4.8 Checkpoint publication

- **Cadence** (PROPOSED, PENDING measurement): one checkpoint per optimizer step, shipped as
  a lossless sparse delta (new bf16 values at the changed positions) plus the SHA-256 of the
  rebuilt checkpoint. A full anchor every N steps (e.g. 50). Fallback: 2-4 steps per
  checkpoint if deltas are too large or miners cannot keep up.
- **Lagging miners:** a cumulative delta from their version to the latest (up to a size cap), else the
  latest anchor plus deltas. A hash mismatch means re-download the anchor.
- Training is untouched: the trainer keeps its fp32 master weights (both the single-GPU and
  FSDP2 paths already do) and never reads deltas.
- **Transport** (DECIDED): bucket `reliquary-public` on R2, separate from the private buckets,
  written with a token scoped to it alone. Miners call `GET /rl/checkpoint` and receive a
  hotkey-signed index plus short-lived presigned URLs; they download from R2 directly
  (egress free). A custom public domain can be added later if the bucket's zone moves to the
  same Cloudflare account.
- **Backup** (DECIDED): a full checkpoint pushed to Hugging Face every M steps (about every
  1-2 h), in the background, never blocking the trainer; a rolling window of recent
  checkpoints plus milestone checkpoints kept forever, so the HF quota never fills (it
  stopped the subnet on 2026-09-16).
- **Measured 2026-10-15** (9B, lab trainer with fp32 masters, lr 2e-6, no warmup; checkpoints
  k0/k8/k16): 8 steps change 9.6-10.2 % of bf16 elements, compressed delta 1.76-1.84 GB
  (17.9 GB full); 16 steps change only 13.5 % (union), 2.32 GB; rebuild bit-exact. Per-step
  delta not measured directly (no consecutive checkpoints), expected ~0.3-0.6 GB. Changes
  concentrate in the DeltaNet `conv1d` weights (25-36 % per 8 steps). Decode is slow in the
  first implementation (68-109 s for 8 steps): to optimise before wiring.
- Anchors carry the HF config and tokenizer files so a new miner needs nothing else.

### 4.9 Payment and prices

- Pay flat per matched token of trained groups (DECIDED, user 2026-10-15). Rollouts below the
  floor are never paid.
- Out-of-zone first scans: 25 % of an in-zone group, after audit (4.5). Otherwise unpaid;
  out-of-zone work counts for quota either way.
- Per-rollout floor prevents quantized formats from out-earning bf16 (measured: a 3090 earns
  +65 % paid tokens/s with W4A16 under pay-per-match without a floor).
- Per-env price controller (async version): rise when the env's share of trained tokens is
  below target or the trainer waited on it; descend when its buffer overflows. `/rl/pay` and
  quota published.

### 4.9b Miner formats and pay (PENDING the fp8 replication)

- Miners pick the format that maximises paid tokens per second. Match on the 9B under Gumbel:
  bf16 0.986, fp8 KV 0.985, fp8 weights 0.963-0.971, W4A16/NVFP4 ~0.855.
- fp8 weights lose ~2 % of pay under pay-per-matched-token and generate ~30-60 % faster
  (estimate, to measure on 2-3 cards): fp8 is the rational choice for every miner.
- A floor cannot separate fp8 from honest bf16: honest bf16 at lag 4 matches 0.970-0.978, the
  same as fp8 at lag 0. A pay curve steep enough to cancel a 40 % speed gain (about 30 % of
  pay for 2 points of match) would also punish honest staleness.
- So the decision rests on the training value of fp8 data (2B runs with a real vLLM fp8
  sampler, 3 seeds, in progress):
  - **fp8 trains as well:** fp8 weights become an official format; the network gets
    30-60 % more tokens for the same emission.
  - **fp8 trains worse:** price the measured loss of value into pay per rollout match, and
    ship the fp32 head (honest 0.994) to widen the gap between bf16 and fp8.
- W4A16 and below stay under the floor in both cases.

### 4.10 Multi-turn and sandboxed envs

- Leases, mandatory submission, quota and lifecycle apply unchanged. Episode verification
  (signed sandboxes, per-turn forced draw) stays; the per-turn draw moves to Gumbel and the
  runner position. Sandbox time counts in latency, not in the match.

## 5. What is removed

- Fill-closed windows, picks, `fifo-ingress/v1` selection, the trainer cursor and pick
  pacing, `candidate_shortfall`, window-level price snaps.
- The public seed pool, precommits and `no_reveal_circuit`.
- TOPLOC / GRAIL proofs on RL rollouts, miner logprob claims as a trust input.
- The separate proof forward for in-zone groups (if 4.7 is confirmed).
- The counter-based `ForcedSeedVLLMProcessor` and the inverse-CDF draw (kept only for tests
  until removed).

## 5c. Considered and set aside

- **Oversampling with selection** (PODS-style down-sampling, Reinforce-Ada-style adaptive
  sampling to the first opposite outcome). Selection on outcomes is safe only under a fixed
  validator rule (a miner's free choice of 16 among 32 is an untraceable steering channel);
  the rule-based version works but adds protocol and grading complexity for an unmeasured
  gain. Set aside (user, 2026-10-15): leases impose 16 draws, all submitted. Reopen if the
  in-zone rate is too low in the run (many 16/16 groups). The user plans to revisit it
  together with variance-weighted pay; even then, the 16 trained rollouts must be chosen by
  a fixed validator rule, because equal-variance subsets leave the miner a free choice
  (length, style, steering).

## 6. Open decisions

1. Checkpoint cadence: 1 step per checkpoint with deltas vs 2-4 steps (after the delta
   measurement).
2. Training forward as verification (4.7): confirm after measuring trainer-forward picks
   against the proof path.
3. Floor value with fp32 heads (qualification).
4. fp8 weights as an official format (4.9b, fp8 replication).
5. Lease expiry and starting quota numbers (qualification).
6. Learning rate on the 9B (2B: 5e-6 learns faster with substitution; uncorrected collapses).

## 7. Work plan

| Phase | Content | Depends on |
|---|---|---|
| P0 | Merge the runner-keyed position fix (`feat/forced-seed-runner-fix`) | user go |
| P1 | Gumbel draw (miner runner, verifier, episode turns), fp32 head, protocol version; GPU validation on two card types | P0 |
| P2 | Lease service: endpoints, exclusivity, interval fairness, nonce commit/reveal, expiry, quota, prompt lifecycle and `/rl/prompts/state` (built: `feat/rl-leases`, lifecycle to add) | - |
| P2b | Checkpoint publication: deltas, anchors, signed index, presigned URLs, HF backup rotation | delta measurement |
| P3 | Submission path: grade-all, buffer intake, out-of-zone store and sampled audit | P1, P2 |
| P4 | Async trainer: env-mix batches, one-forward verification + substitution loss, telemetry, loss switch | P1, P3 |
| P5 | Payment per matched token, floor, 25 % first-scan pay, async price controller, `/rl/pay` | P3, P4 |
| P6 | Reference miner: lease client, prompt-state reader, quota-aware scheduler, background delta download, Gumbel generation | P1, P2, P2b |
| P7 | Qualification on GPUs: end-to-end on a multi-card testbed (honest, fp8, int4, stale, Sybil and squat attempts) | all |
| P8 | Deployment: validators (accept both protocols) -> miners -> switch; delete the window path at the new run's launch | P7 |

## 8. Risks

- **vLLM internals:** the runner-keyed position and Gumbel depend on vLLM 0.30 internals;
  pinned, refused otherwise. Upgrades need re-validation.
- **Validator load:** grading all groups plus audits; with 4.7 the trainer forward replaces
  the in-zone proof forward, so the extra GPU cost is the audits only.
- **Dynamics over hundreds of steps:** the 2B runs cover 400 steps; the 9B run's telemetry
  (match, substitution rate, entropy) and the loss switch are the guard.
- **Lease service as a single point of failure:** needs persistence and restart safety
  (open leases survive a restart).
- **Delta chain integrity:** a corrupted delta would put miners on wrong weights; the signed
  index and per-checkpoint SHA-256 catch it, with the anchor as recovery.
