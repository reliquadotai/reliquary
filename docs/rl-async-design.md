# Async RL with leased prompts and a Gumbel forced draw: design

Status: DRAFT for review (2026-10-15). Supersedes the window redesign discussion in
`docs/rl-windows-redesign-notes.md`. Evidence: `docs/rl-off-policy-tests.md` and the
campaign reports under `~/offpolicy-lab/results/` (fairness, staleness, red team,
literature, H100 runs of 2026-10-14/15).

## 1. Decisions in one page

1. **Fully asynchronous.** No windows, picks, fill-closed rounds or trainer pacing. Miners
   generate continuously on whatever recent checkpoint they hold; the trainer takes a step
   whenever its buffer holds a batch.
2. **One verification: the forced draw.** Every token is the validator's own Gumbel-max pick
   for a public noise stream. A rollout is accepted when its match rate against the latest
   checkpoint is above a floor. No TOPLOC, no trust in miner logprobs, no old checkpoints on
   the validator.
3. **Gumbel-max draw**, placed by the vLLM runner's own token count (not a logits-processor
   counter). Optionally an fp32 LM head on both sides.
4. **Leases.** A miner reserves prompts before generating. The draw's nonce is revealed only
   with the lease and depends on (validator secret, prompt, checkpoint) alone, so a lease
   can neither be pre-generated nor re-rolled. Every lease must end in a submission, whatever
   its outcome; there is no release.
5. **Quota measured, not declared.** Concurrent leases per hotkey follow its measured
   throughput and latency (Little's law), with slow start and halving on expiry.
6. **Validator grades everything; trains on what is in zone.** In-zone groups are fully
   verified, trained and paid. Out-of-zone groups are graded, stored, verified by random
   sampling, not paid, and count toward quota.
7. **Loss: substitution.** At each completion position the trainer scores the validator's
   pick v* (equal to the miner's token where they agree). Drift is exactly zero for any
   miner hardware; one optimizer step per batch.
8. **Pay per matched token** of trained groups, with a per-rollout floor so that quantized
   generators do not out-earn honest bf16 ones. Per-environment prices steer supply to the
   trainer's target mix.

## 2. Evidence behind the decisions

All on the RL base `ReliquaryForge/teutonic-i-attention-graft-healed-v1` (9B, T=1 full
vocab), vLLM 0.30.0 miners, HF proof path validators.

| Question | Result |
|---|---|
| Honest match, inverse-CDF (today), bf16 heads | 0.944-0.950 on 3090/5090/A6000/L40, IF env ~0.87 |
| Honest match, **Gumbel**, bf16 heads | **0.986** on all cards (spread 0.0006), IF 0.984, rollout min >= 0.95, 0 groups rejected |
| Honest match, Gumbel, fp32 heads both sides | 0.994 |
| Validator hardware | changes pooled match by < 0.0005 |
| Desync bug (counter-based processor) | 21-47 % of honest rollouts on preemption; runner-keyed position: 0 bad over 38M+ positions, 2,000+ resumes; production port passes async on/off/multiprocess |
| Why inverse-CDF disagrees | CDF geometry (low-p tokens mid-vocab), no token-specific bias; gone under Gumbel |
| Value of disagreement positions (branching, 2,450 positions) | not higher than entropy-matched controls (abs Q gap 0.096 vs 0.103) |
| Drift (constant advantage), staleness 4 | current loss 418k (7x noise floor, cosine 1.0 with "SFT toward the sampler"), substitution ~0 |
| Staleness cost, inverse-CDF match | 0.973 / 0.962 / 0.962 / 0.945 / 0.917 at 0 / 1 / 2 / 4 / 8 steps |
| Quantized miners vs bf16 validator, Gumbel | fp8 KV 0.985 (undetectable), fp8 weights 0.963-0.971, W4A16/NVFP4 ~0.855 (AUC 1.0) |
| Trusting declared logprobs | not needed; red team: small gain, residual coordinated-Sybil risk |

## 3. Flow

```
miner                         validator (lease + proof)                trainer
-----                         -------------------------                -------
GET /rl/quota  ------------>  quota(hotkey)
POST /rl/lease [P1..Pn]  --->  exclusive leases, nonce_i = H(S_k, P_i, k)
                <-----------  {lease_id, nonce, k, expires_at}
generate 16 rollouts per P
  (Gumbel draw from nonce, runner-keyed)
POST /rl/submit  ----------->  grade all rollouts (CPU)
                               in zone  -> forced-draw check vs latest (GPU)
                               out zone -> sampled check (p = audit rate)
                               store group + verdict      ----------> buffer (in-zone, accepted)
                                                                       step when a batch fills
                                                                       publish checkpoint k+1
                                                           <---------- reveal S_k after k closes
```

## 4. Components

### 4.1 Forced draw (protocol)

- **Gumbel-max:** pick = argmax_v (log p_v + G_v), G_v = -log(-log U_v), U_v from a
  counter-based PRNG keyed by (nonce, rollout index, position, v). The lab's splitmix64 +
  float64 G is bit-identical on Ampere, Ada, Hopper, Blackwell (`lab/gumbel.py`). It is an
  exact sample of p; diversity is unchanged.
- **Position:** `num_tokens_no_spec - num_prompt_tokens (+ base_offset)` from the vLLM input
  batch, params keyed by request id (`feat/forced-seed-runner-fix`, validated). vLLM pinned
  to 0.30.x; speculative decoding refused.
- **fp32 head:** logits kept in fp32 at the LM head output on miner and validator (match 0.986
  -> 0.994; negligible generation cost). Decide at qualification.
- **Verification:** the validator recomputes logits (HF bf16 proof path, batch 1) on the
  latest checkpoint and its own Gumbel pick per position. Per-rollout match on stochastic
  positions (max p < 0.99) and pooled per-hotkey statistics.
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
  There is no release, so miners cannot select on the realised outcome; they select prompts
  before seeing results.
- **Expiry:** fixed per environment and generous (max length at a slow reference speed, plus
  margin). Not derived from the miner, so not manipulable.

### 4.3 Quota

- Target concurrency = measured throughput (groups/min) x measured lease-to-submit latency x
  1.5, per hotkey, over a sliding window.
- New hotkeys start at a small quota (e.g. 4). While a hotkey uses its full quota and
  delivers on time, the quota doubles (slow start: 4 -> ~1,600 in ~9 rounds). Every expiry
  halves it. Out-of-zone submissions count as delivered work.
- Published per hotkey (with the reasons) so miners can see their state.
- Anti-squat: blocking prompts requires generating and submitting them; new keys can hold
  only a handful of leases; registration burn applies.

### 4.4 Submission and verification

- **Grading:** every group is graded by the validator (rewards never come from the miner).
- **In zone** (sigma >= 0.24, at most one truncated rollout, current env rules): full
  forced-draw check against the latest checkpoint, per-rollout floor (initial candidate 0.95
  under Gumbel, to calibrate, see 4.5), group accepted -> buffer.
- **Out of zone:** stored with grades. Reuse the DECIDED exploration audit regime
  (`next-rl-run-decisions.md` §C): 100 % audit of a hotkey's first 100 out-of-zone groups
  (probation), then a random fraction drawn after submission (user: 20 %; decisions doc:
  15 %), graded sanctions (deterministic forgery bans, statistical failure re-opens
  probation). Upload carries tokens only (no proofs), compressed.
- Group integrity: all 16 rollout indices required; the draw is bound to the rollout index.

### 4.5 Staleness

- Miners generate on any checkpoint they hold; the lease records k; verification is always
  against the latest checkpoint. Staleness costs match, hence pay, and is self-priced.
- **Measured under Gumbel (replay of stored runs, 2026-10-15):** honest match 0.987 / 0.983 /
  0.982 / 0.970-0.978 / 0.949-0.970 at lag 0 / 1 / 2 / 4 / 8; worst honest rollout 0.974 /
  0.963 / 0.964 / 0.946-0.956 / 0.883-0.942. W4A16 at lag 0: 0.855, best rollout ~0.91.
  Gumbel halves the staleness cost of inverse-CDF.
- **Floor:** 0.95 per rollout with bf16 heads (about 0.96 with fp32 heads, estimated). Keeps
  every honest rollout at lag <= 2, rejects all W4A16. fp8 weights sit around 0.95: a pricing
  question, not a floor question.
- **Rollout-level rule:** a rollout under the floor is excluded and unpaid; the group is
  judged on its pooled match (one bad rollout must not kill the group: x16 compounding).
- **Max lag S = 4** checkpoints (leases on older k are refused). Lag 8 fails every group
  early in training, and a lease cannot know the training phase.
- Publish match by declared lag; revisit S when honest p1 at lag S nears the floor.

### 4.6 Buffer and trainer

- Accepted in-zone groups enter a buffer tagged with env, checkpoint and arrival time.
- The trainer takes a step as soon as it can fill a batch at the target env mix; if an env is
  short, the step proceeds with what is available (no blocking) and that env's price rises.
  Within an env, freshest first.
- If supply exceeds trainer capacity, groups older than the max lag are dropped unpaid and
  prices fall; only trained groups are paid.
- One optimizer step per batch (decided). Loss: substitution (4.7). Checkpoints published
  continuously; miners download in the background.

### 4.7 Loss

- Per completion position: score `log p_theta(v*)` where v* is the validator's Gumbel pick at
  that position (equal to the miner token on ~98.6 % of stochastic positions). Advantages as
  today (group-normalised, token-level normalisation, degenerate groups dropped, soft
  overlong penalty, no KL).
- With one step per batch, the PPO ratio is 1 and the clip is inert; it can stay for safety.
- v* comes from the proof forward (stored per position), or is recomputed from the trainer's
  own logits with the same noise (equivalent when the trainer runs the latest checkpoint).
- Telemetry per step: match rate, substitution rate, logit-space drift estimate, entropy,
  grad norm. Switchable loss (substitution / mask / current) by config.

### 4.8 Payment and prices

- Pay per matched token of trained groups. Tokens below the floor are never paid.
- Out-of-zone groups: the DECIDED exploration rule stays (§B of the decisions doc: 15 % of a
  training group for the first scan of a prompt in the run, capped, unused budget back to
  training); otherwise unpaid. They count for quota either way.
- Per-rollout floor prevents quantized formats from out-earning bf16 (measured: a 3090 earns
  +65 % paid tokens/s with W4A16 under pay-per-match without a floor).
- Per-env price controller (async version): rise when the env's share of trained tokens is
  below target or the trainer waited on it; descend when its buffer overflows. `/rl/pay` and
  quota published.

### 4.9 Data products

- All groups (in and out of zone) stored with grades, checkpoint and miner: pass rate per
  prompt per checkpoint, SFT-grade all-correct groups, miner selection quality.
- The run-wide observation log and its public R2 segments already exist as DECIDED
  (decisions doc §A, §D): every verified group's rewards per prompt and checkpoint are
  published, which is the difficulty map. Leases add the lease/scan state on top.

### 4.10 Multi-turn and sandboxed envs

- Leases, mandatory submission and quota apply unchanged. Episode verification (signed
  sandboxes, per-turn forced draw) stays; the per-turn draw moves to Gumbel and the runner
  position. Sandbox time counts in latency, not in the match.

## 5. What is removed

- Fill-closed windows, picks, `fifo-ingress/v1` selection, the trainer cursor and pick
  pacing, `candidate_shortfall`, window-level price snaps.
- TOPLOC / GRAIL proofs on RL rollouts, miner logprob claims as a trust input.
- The counter-based `ForcedSeedVLLMProcessor` (kept only for tests until removed).

## 5b. Proposed resolutions of conflicts with earlier decisions (pending user OK)

From `docs/rl-async-implementation-map.md` §2:
- C1 exploration pay: keep a first-scan bonus (15 %), paid only after audit.
- C2 public seed pool (choose M of 2xM): REMOVED. The lease imposes 16 draws, all submitted.
- C4 audits: an out-of-zone audit is checked against the lease's checkpoint, so the proof
  path keeps the last 1-2 checkpoints for audits only.
- C6 cooldown: keep the in-zone training cooldown, counted in checkpoints.
- C7 steps: several optimizer steps per published checkpoint (no 18 GB publish per step);
  the PPO clip stays meaningful; v* is the latest published checkpoint's pick.
- C8 truncation: keep the current robust rule and `unproven` lane.
- C12 expired leases hand the same draw to the next lessee: accepted (throwaway keys have
  tiny quotas, expiry earns nothing); monitored.

## 6. Open decisions

1. fp32 LM head: in the reference miner and the proof path by default (user, 10-15);
   remaining question is whether the floor is calibrated on fp32 or bf16 miners.
2. Per-rollout floor value under Gumbel, after the staleness measurement; floor relative to
   lag or absolute; max lag S.
3. Out-of-zone audit rate and penalty.
4. Out-of-zone audit rate after probation (15 % vs 20 %).
5. Exact lease expiry per env and quota parameters (start, growth, headroom).
6. Pay schedule details: flat per matched token vs a match-rate multiplier.
7. Official quantized formats (validator-computed q + score centering) later, or never.

## 7. Work plan

| Phase | Content | Depends on |
|---|---|---|
| P0 | Merge the runner-keyed position fix (`feat/forced-seed-runner-fix`) | user go |
| P1 | Gumbel draw in reliquary (miner runner + validator verifier + episode turns), fp32 head option, protocol version; staleness-under-Gumbel measurement; floor calibration | P0 |
| P2 | Lease service: endpoints, exclusivity, interval fairness, nonce commit/reveal, expiry, quota controller | — |
| P3 | Submission path: grade-all, in-zone full check, out-of-zone sampled audit, storage of all groups | P1, P2 |
| P4 | Async buffer and continuous trainer (env mix selection, max lag, checkpoint publishing cadence) | P3 |
| P5 | Substitution loss + drift telemetry + loss switch | P1 |
| P6 | Payment per matched token, floor, async price controller, `/rl/pay` + quota endpoints | P3, P4 |
| P7 | Reference miner: lease client, quota-aware scheduler, background checkpoint download, Gumbel generation | P1, P2 |
| P8 | Qualification on GPUs: end-to-end on a multi-card testbed (honest, fp8, int4, stale, Sybil and squat attempts) | all |
| P9 | Deployment order: validators (accept both protocols) -> miners -> switch | P8 |

## 8. Risks

- **vLLM internals:** the runner-keyed position and Gumbel depend on vLLM 0.30 internals;
  pinned, refused otherwise. Upgrades need re-validation.
- **Staleness vs floor under Gumbel** (4.5) not yet measured.
- **Validator load:** grading all groups plus audits; proof forwards only for in-zone and
  audited groups.
- **Dynamics over hundreds of steps:** only the live run shows them; the drift telemetry and
  the loss switch are the guard.
- **Lease service as a single point of failure:** needs persistence and restart safety
  (open leases survive a restart).
