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
    generators do not out-earn honest bf16 ones. Per-environment prices steer each env's supply to
    its target share of the trainer's speed, read from the training queue (DECIDED 2026-10-17).

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
| **2B long run and lr 5e-6** (same noisy sampler, match 0.90, lag up to 3) | lr 2e-6, 400-step run: no correction collapses at step ~130 (eval 0.78 -> 0.005 by step 200, KL 0.024 -> 9.6); substitution flat KL 0.022 and still improving at step 260 (0.83/0.795). lr 5e-6: no correction collapses to 0 on both seeds; substitution 0.83/0.81; paper MIS+SC (true q) 0.86/0.84 (2 seeds, on H100), reference without mismatch 0.87/0.86. Substitution costs ~0.03 at 4.2 % substituted positions; production rate ~1.4 % |
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
  counter-based PRNG keyed by (nonce, rollout index, position, v). splitmix64 + a float32 G
  built from IEEE +, -, *, / only (its own logf, no FMA) is bit-identical on every device,
  CPU included (`forced_sampling` module doc). It is an exact sample of p; diversity is
  unchanged.
- **Position:** `num_tokens_no_spec - num_prompt_tokens (+ base_offset)` from the vLLM input
  batch, params keyed by request id (`feat/forced-seed-runner-fix`, validated). vLLM pinned
  to 0.30.x; speculative decoding refused.
- **fp32 head:** logits kept in fp32 at the LM head output on miner and validator (match 0.986
  -> 0.994): bf16 operands, fp32 products and accumulation (`torch.mm(..., out_dtype=float32)`,
  what vLLM's `head_dtype=float32` runs). Measured 2026-10-16 on H100: ~1 % of miner decode
  throughput; on the validator 79 -> 5.8 ms per 2048 rows against the former fp32 SGEMM (needs a
  torch with `out_dtype`; 2.7 falls back to the SGEMM). The 0.994 was measured with the SGEMM
  validator: re-measure at qualification (1 different pick in 381k between the two).
- **Verification statistic:** per-rollout match on stochastic positions (max p < 0.99) and
  pooled per-hotkey statistics.
- Protocol version bump: Gumbel and inverse-CDF are incompatible; validators first.

### 4.2 Leases

- **Request:** `POST /rl/lease {env, prompt_ids[], checkpoint k}`; batched requests allowed.
- **Exclusivity:** one open lease per prompt, whatever its k (no parallel leases of one prompt
  over the unrevealed checkpoints). A prompt submitted at k is closed for k (the draw would be
  identical anyway). A request is cut to the hotkey's room (quota - open leases) before any lookup.
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
  keys have tiny quotas and expiry earns nothing). A submission is judged at its arrival: once it
  arrived on time the sweep cannot expire its lease while it waits for or goes through grading.

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
  with the same floor and match as training (PROPOSED: no old checkpoints kept); a failure of a
  group leased more than 2 checkpoints back (`RL_SANCTION_MAX_LAG`, the trainer's rule) is
  inconclusive, unsanctioned. An audit that abstains (too few drawn positions) or whose forward
  keeps failing is no probation pass and re-opens probation. Upload carries tokens only,
  compressed; `/rl/submit` names its hotkey in a header, checked registered before the body is
  read.
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
  the 15 % of the decisions doc §B), when its rollouts carry at least `RL_FIRST_SCAN_MIN_TOKENS`
  (placeholder 1024) model tokens. Later rescans of a too-hard prompt are unpaid when out
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
  judged on its pooled match (one noisy rollout must not kill the group: x16 compounding). The
  floor tolerates honest noise, never forgery: one rollout under `RL_FORGERY_MATCH_FLOOR`
  (placeholder 0.5) rejects the group as evidence.
- **No quality lag cap** (user, 2026-10-15): the floor alone rejects rollouts that are too stale.
- **Leases only on checkpoints whose secret S_k is unrevealed** (DECIDED). Once S_k is public
  anyone can compute every nonce of k, pre-generate and lease only the prompts that land in
  zone. S_k is revealed late (e.g. 16 checkpoints after k), so the rule never binds an honest
  miner. Publish match by declared lag.

### 4.7 Buffer, trainer and loss

- Accepted in-zone groups enter a buffer tagged with env, checkpoint and arrival time. The
  trainer takes `batch_groups` groups of any env in arrival order and waits until the batch is
  full (DECIDED 2026-10-16; prime-rl, AReaL, PipelineRL, verl, SkyRL, open-instruct all do this).
  The batch mix is whatever arrives: each group has its own baseline, so composition adds noise,
  not bias. The loss is the global token mean of the step.
- A global cap on buffered groups (`rl_max_pending_groups`, placeholder 256 = 4 batches; open
  leases are not counted, the quota and expiry bound them, and one hotkey's open leases take at
  most 1/4 of the cap) holds miners back when they outrun the trainer (refusal `full`), instead of
  dropping their work: AReaL/SkyRL throttle generation the same way. Groups past `max_lag` are still dropped
  unpaid, as a safety.
- Streaming micro-batches as groups arrive (RollPacker) was looked at: it helps synchronous
  systems where the trainer idles during generation; here the buffer already keeps the trainer
  busy, so it would only shave one step of lag when supply is short. Not done.
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
  - A group is accepted, and paid, when it is trained, not when it is submitted. It is reported
    payable before its step's checkpoint is published: pay is for trained work, and if the
    trainer dies in between the step's weights are lost but the miners did the work (deliberate).
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
  latest anchor plus deltas. A hash mismatch means re-download the anchor. Pruned objects leave
  the index first and the bucket one presigned-URL lifetime (15 min) later.
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
- Per-env price controller (DECIDED 2026-10-16, measured from the queue 2026-10-17): one ratio per
  env, r = its supply / (its target share x the trainer's speed). Everything is read from the
  training queue, nothing is reported by the trainer: the speed is the completion tokens it took
  over the seconds it did not wait (sums decayed per tick); an env's supply is the completion
  tokens of its groups queued per second; its target share (`env_split`, the operator's only
  knob) is renormalised over the envs that supplied. Per tick (300 s): r under 0.90 rises x1.05,
  over 1.10 (or refused by the global cap) descends 2 %, else holds. Two global guards from the
  queue: while the cap refuses leases no price rises, while the trainer waits (over 5 % of the
  tick) no price descends. The trainer neither took nor waited for 3 ticks (down): hold.
  `/rl/pay` publishes ratio, supply, demand and price per env.
  Simulation (scratchpad `queue_sim.py`, 72 h, 7 scenarios, miners reallocating on prices, groups
  taking several steps to generate): without the guards, miners moving between envs keep some
  env short at every tick, it rises x1.05 while the long ones descend 2 %, and every price
  ratchets to the cap with the queue full. With them, prices settle, the mix is within a few
  points of target, and the spend falls to 18-67 % of the budget when GPUs are plentiful (the rest
  burns). The cap is then the lag knob: trainer wait 7 / 2 / 1 / <1 % and queue wait p50 0.6 / 0.9
  / 1.3 / 2.2 steps at cap 128 / 160 / 192 / 256 (192 by default). The generation time, not the
  queue, is most of the lag (p50 ~5 steps issue -> train).

### 4.9b Miner formats and pay (DECIDED 2026-10-16: no official format)

- No format is official or forbidden: miners pick whatever maximises paid tokens per second,
  and the match floor is the only rule. Match on the 9B under Gumbel: bf16 0.986, fp8 KV
  0.985, fp8 weights 0.963-0.971, W4A16/NVFP4 ~0.855 (under the floor).
- Pay per matched token already prices a format's mismatch; the substitution loss keeps the
  trainer's gradient unbiased whatever the miner ran.
- fp8 data on the 2B (real vLLM fp8 sampler) trained as well as bf16 within seed noise; the
  3-seed comparison against the papers' correction was not finished and is not needed for
  this decision.

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
4. Lease expiry and starting quota numbers (qualification).
5. Learning rate on the 9B (2B: 5e-6 learns faster with substitution; uncorrected collapses).

## 7. Work plan

| Phase | Content | Depends on |
|---|---|---|
| P0 | Merge the runner-keyed position fix (`feat/forced-seed-runner-fix`) | user go |
| P1 | Gumbel draw (miner runner, verifier, episode turns), fp32 head, protocol version; GPU validation on two card types | P0 |
| P2 | Lease service: endpoints, exclusivity, interval fairness, nonce commit/reveal, expiry, quota, prompt lifecycle and `/rl/prompts/state` (built: `feat/rl-leases`, lifecycle to add) | - |
| P2b | Checkpoint publication: deltas, anchors, signed index, presigned URLs, HF backup rotation | delta measurement |
| P3 | Submission path: grade-all, buffer intake, out-of-zone store and sampled audit | P1, P2 |
| P4 | Async trainer: arrival-order full batches, one-forward verification + substitution loss, telemetry, loss switch | P1, P3 |
| P5 | Payment per matched token, floor, 25 % first-scan pay, async price controller, `/rl/pay` | P3, P4 |
| P6 | Reference miner: lease client, prompt-state reader, quota-aware scheduler, background delta download, Gumbel generation | P1, P2, P2b |
| P7 | Qualification on GPUs: end-to-end on a multi-card testbed (honest, fp8, int4, stale, Sybil and squat attempts) | all |
| P8 | Deployment: validators (accept both protocols) -> miners -> switch; delete the window path at the new run's launch | P7 |

## 8. Risks

- **vLLM internals:** the runner-keyed position and Gumbel depend on vLLM 0.30 internals;
  pinned, refused otherwise. Upgrades need re-validation.
- **Cost of the Gumbel draw (measured 2026-10-16, H100):** one noise value per vocabulary
  token (~248k) per position. The first implementation ([rows, vocab] int64/float64
  temporaries in ~12 torch ops) cost 2.4 ms at 64 rows and 9.5 ms at 256 rows, against a
  9B vLLM decode step of ~9.8 / ~20.5 ms (25-45 % of the step; float64 was only ~20 % of
  it, the unfused int64 temporaries the rest). The fused Triton pass (`gumbel_kernel`, one
  read of the logits, noise in registers) takes 0.15 / 0.46 ms (0.20 / 0.56 ms for the whole
  `pick_rows` call; ~2-3 % of the step, cheaper
  than vLLM's own softmax + exponential sampling at 0.27 / 0.83 ms). Miners need Triton
  (vLLM ships it); without it `pick_rows` falls back to the torch reference (same bits,
  ~3x slower than the old path).
- **Validator load:** grading all groups plus audits; with 4.7 the trainer forward replaces
  the in-zone proof forward, so the extra GPU cost is the audits only.
- **Dynamics over hundreds of steps:** the 2B runs cover 400 steps; the 9B run's telemetry
  (match, substitution rate, entropy) and the loss switch are the guard.
- **Lease service as a single point of failure:** needs persistence and restart safety
  (open leases survive a restart).
- **Delta chain integrity:** a corrupted delta would put miners on wrong weights; the signed
  index and per-checkpoint SHA-256 catch it, with the anchor as recovery.
