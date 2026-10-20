# Async RL: GPU qualification plan

Plan of 2026-10-16, for the launch the week of 2026-10-19. Everything in
`docs/rl-async-design.md` is coded on `feat/rl-async`; these tests decide the remaining
numbers (design §6) and confirm the risky assumptions (design §8). Each test names what it
decides and its pass criterion. Results go to `~/offpolicy-lab/results/qualif-2026-10/`.

## Hardware

| Role | Need | Why |
|---|---|---|
| Trainer | 1 box, >= 4 x 80 GB (H100/A100), NVLink preferred | 9B FSDP2 with fp32 masters and Adam (~150 GB) |
| Validator GPU (audits, proof path) | 1 x H100 (`noble-fox-be`) | audits and the proof-path reference |
| Miners | 3 card types: one 24-32 GB GeForce (4090/5090/3090), one 48 GB (L40S/A6000), one H100 | honest match per card, fp8 gain, sync time |
| Validator controller | CPU box (local VPS is enough for the test) | leases, buffer, ledger, prices |

The 9B in bf16 needs >= 24 GB: on a 24 GB card, check the KV cache left at `max_tokens`.

## Tests

### Q1 Bit-exact noise on every card (minutes per card)
`scripts/gumbel_check.py` on each card type and on the validator/trainer boxes. Pass: the
three hashes (bits, G, picks) equal the H100 reference. Decides nothing, gates everything.
Reference: bits `d34e0008d56bc338`, G `d1c696b1fd17f4b4`, picks `0b427caf7280d994`. Measured 2026-10-16,
identical on H100, A100-SXM4, RTX 4090 (torch 2.13 cu130) and on CPU (torch 2.11, 2.13).

### Q2 Honest match: miner vs proof path vs trainer forward (the key test, ~half a day)
Same leases, same prompts (math, code, logic, 64 x 16 each, max length of the run):
- miners: vLLM 0.30 + fused Gumbel + fp32 head, on each card; also a bf16-head miner and an
  fp8-weights miner on one card;
- verifier A: proof path (HF, batch 1, `fp32_head_matmul`);
- verifier B: the FSDP2 trainer forward as `train-async` runs it (packed micro-batches,
  sharded), picks read from the `DrawGate` tallies without an optimizer step.
Measure per rollout: match on stochastic positions, its distribution, the group pooled match.
- Pass for design §1.9 (training forward = verification): B agrees with A on >= 99.9 % of
  picks and the per-rollout match distributions overlap. If not: a separate verification
  forward on the trainer box (cost to measure in Q4).
- Decides: rollout floor and group floor (honest p0.1 on the worst card minus a margin; fp8
  and bf16-head miners reported, they are miners' choices), the forgery tolerance.

### Q3 Miner throughput and sync (~2 h per card)
- vLLM decode tokens/s at batch 64/256: plain sampling vs forced draw (fused Gumbel + fp32
  head); bf16 vs fp8 weights. Pass: forced draw costs <= 5 %.
- Checkpoint sync: download a real per-step delta from R2 (presigned) + decode + in-place
  weight load into the running engine; time per card. Decides the checkpoint cadence
  (design §6.1) together with Q4.

### Q4 Trainer on the 9B (~half a day)
`reliquary train-async` on the trainer box against a local validator, fed by Q5's miners
or by a replayed buffer:
- step time (forward + backward + optimizer), the speed it reports, publication time;
- per-step delta size (compressed) and `DeltaPublisher` time; anchor size and time;
- HF backup in background does not stall steps.
Decides: `batch_groups`, steps per checkpoint, anchor interval.

### Q5 End-to-end with honest and adversarial miners (~1 day, overnight run)
Validator controller + trainer + 3 honest miners (one per card) for >= 100 steps, then add:
stale (holds an old checkpoint), fp8, W4A16 (must fall under the floor), a forger (random
tokens on some positions), a squatter (takes leases, never submits), a Sybil set (10 hotkeys
on one GPU), a resubmitter.
Check: leases, expiries and quota (slow start, expiry cut, paid-only credit), lifecycle states and
`/rl/prompts/state`, audits and bans, `full` refusals under the global cap, `/rl/pay` and a
settled period, price ticks (supply/demand ratio per env, read from the queue), substitution rate, KL, entropy,
eval on held-out problems every ~20 steps.
Decides: lease expiry and starting quota, `rl_max_pending_groups`, price band; first look at
the 9B learning rate (2e-6 vs 5e-6 on two short branches if time allows).

### Q6 Multi-turn episode env (~2 h)
One sandboxed env through leases and the episode engine on the 9B: episode engine restart
time per checkpoint, verification of a full episode, pay.

## Order

Day 1: Q1 on every card, Q2, Q3 in parallel on the miner cards.
Day 2: Q4, then Q5 starts (overnight).
Day 3: Q5 results, Q6, parameters set in the registry entry; fixes.
Day 4: buffer for reruns; deployment prep (torch with `mm(out_dtype)` in the validator and
proof-worker images; signer, proof worker, controller in that order).

## Outputs to fill

| Parameter | Placeholder | Set by |
|---|---|---|
| `RL_ROLLOUT_MATCH_FLOOR` | 0.90 | Q2 |
| `RL_GROUP_MATCH_FLOOR` | 0.90 | Q2 |
| `RL_FORGERY_MATCH_FLOOR` | 0.5 | Q2 (honest p0.001 on the worst card) |
| `RL_GAP_TOLERANCE` | 8.0 (honest gaps grow with lag) | Q5: logged `async/gap_*` at the real lag |
| `RL_SUBSTITUTE_GAP` | 3.0 (re-grade threshold) | Q5: re-grade rate vs honest moved rewards |
| `RL_TERMINAL_GAP` | 1.0 (final EOS) | Q5: `async/rollouts_bad_termination` at real lag |
| `RL_AUDIT_FORGERY_MISMATCH_TOLERANCE` / `_POSITIONS` | 1 hard mismatch per 4096 near-deterministic positions | Q5: honest long code rollouts at lag 0-2 |
| `RL_AUDIT_MAX_LAG` | 4 | Q5: audit lag and dropped audits under the real audit load |
| `RL_FIRST_SCAN_MIN_TOKENS` | 1024 | Q5 |
| `HOTKEY_MAX_OPEN` | 64 open leases | Q5 (Sybil, squatter) |
| `MAX_GRADING_PER_HOTKEY`, submit reads | 2; 8 total, 2 per hotkey | Q5 |
| lease expiry reference speed | 20 tok/s | Q3, Q5 |
| starting quota | 4 | Q5 |
| `EXPIRY_BACKOFF` | x0.9 per expiry (at least one lease) | Q5 |
| `rl_max_pending_groups` (buffered groups only) | 192 (simulated: the lag knob) | Q5 |
| price band low/high | 0.90 / 1.10 (rise 1.05, decay -2 %/tick) | price simulation, Q5 |
| `batch_groups` | 64 | Q4 |
| steps per checkpoint | 1 | Q3, Q4 |
| learning rate | 2e-6 | Q5 |
| per-env cooldown (math, code) | global default | operator, before launch |

## Math budget probe (2026-10-19)

Question: does a rollout cut at the math cap fail, or would it have been right with more budget? Model: the SFT
starting candidate (teutonic-sft-full-v2-e1 final), 100 DAPO-Math-17k and 100 hard-math RL prompts, 4 rollouts each,
RL sampling (T=1, top_p=1, no top_k), every rollout cut at 32k continued to 64k from the same prefix.

| | DAPO | hard math |
|---|---|---|
| cut at 32k | 25 % | 42 % |
| finished rollouts correct (32k) | 92 % | 88 % |
| cut rollouts still reasoning / looping | 98 % / 0 % | 99 % / 1 % |
| cut at 32k, correct by 64k | 45 % | 46 % |
| accuracy at cap 32k / 48k / 64k | 0.69 / 0.78 / 0.80 | 0.51 / 0.64 / 0.70 |
| cut share at 64k | 12 % | 18 % |

A forced answer at the 32k point is right about half the time, but also about a third of the time for rollouts
that never finish even at 64k: a noisy proxy, given no credit. At 32k a cut is often good reasoning short of room;
at 64k it is a much cleaner failure. Proposed for the run: math cap 64k, a cut scores 0 and its group trains (no
masking), soft overlong zone 0.1 over the last 4,096 tokens, cut share watched and the cap raised again if it stays
high. To check in the rehearsal: 64k sequences in the FSDP2 trainer.

## Rehearsal plan (8 GPUs, 2026-10-21)

Teutonic SFT starting model (teutonic-i-sft-full-v2), trainer FSDP2 on 4 GPUs, validator + audits on 1, two or three
miners (single-turn and episodes), the gVisor sandbox host.

| # | Check | Why |
|---|---|---|
| 1 | Math at the 64k cap: trainer memory and step time | never run |
| 2 | An adversarial miner that replaces end picks: its groups unpaid, honest groups paid | the skipped-end rule |
| 3 | Cut rollouts score 0 and their groups train: in-zone share, length trend | the cut rule |
| 4 | Cooldown (shortened): prompts coming back, leasable pool per env | prompt rotation |
| 5 | Real lag; gap and match floors on the 9B | validated on the 2B only |
| 6 | Audit capacity against the out-of-zone volume (every miner starts in probation) | audit backlog |
| 7 | An 18 GB checkpoint published and swapped in by the miners (single-turn and episodes) | swap on the 9B |
| 8 | SWE and TMax episode throughput; sandbox sessions needed | prod sizing |
| 9 | Kill and resume the validator, the trainer and a miner | robustness |
| 10 | Pay, prices and quota over time, including a miner joining late | economics |

## Must test in the rehearsal (2026-10-20)

- A rollout that goes on where the draw ended it (an EOS pick, any gap) is re-graded ended at that pick; a reward
  that moves leaves its group unpaid. Run an adversarial miner that replaces end picks with the runner-up and check
  that its groups go unpaid while honest groups at lag 0-4 keep their pay (re-grade rate, moved-reward rate).
- Math at the 64k cap: FSDP2 trainer memory and step time on 64k sequences, lease lifetimes, cut share per env
  (watch it during the run; raise the cap if it stays high).
- Cut rollouts score 0 and their groups train: in-zone share and length trend of the math envs over the first steps.
