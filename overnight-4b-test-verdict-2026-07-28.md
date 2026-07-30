# Overnight 4B GPU test — can we merge `feat/4b-migration-bft16k-throughput`?

**Date:** 2026-07-28 (run on a rented H100 80GB; base **Qwen3.5-4B**, real branch params)

> Historical experiment record. Several conclusions below were superseded by
> production-path measurements and the final scheduler/capacity design. Use
> [docs/4b-auction-v3-production-runbook.md](docs/4b-auction-v3-production-runbook.md)
> for merge and deployment decisions.

## Verdict: the branch code is correct and tested, but it is **NOT deploy-ready** — two operational blockers the 4B+16k surfaces must be resolved first.

| | result | merge impact |
|---|---|---|
| Branch code (throughput, BFT 16k, DAPO mask, budget) | all unit suites green | ✅ sound |
| **Incentive — code miners** | **10% of honest code groups rejected** | 🔴 blocker 1 |
| Incentive — math miners | clean (BFT forces, DAPO-masked) | ✅ |
| Memory — GRAIL verify @16k | 17.4 GB | ✅ fine |
| **Memory — training @16k** | **OOM (85.7 GB) on 80GB** without kernels | 🔴 blocker 2 |

---

## Exp A — do miners behave as intended? (real params)

**CODE @ max_new_tokens 32768, n=8, 80 held-out prompts:**
- truncation (hit cap, no EOS): **5.3%**; length median 2870 / p90 10290; only **5.5% exceed 16384**.
- truncated-per-group: `{0:59, 1:13, 2:6, 3:1, 6:1}`.
- **groups rejected (≥2 truncated → `MAX_TRUNCATED_PER_SUBMISSION=1`): 8/80 = 10%.**

→ **Blocker 1 confirmed.** A 4B loops on ~5-10% of code, so ~1 honest code group in 10 is rejected wholesale (the miner loses all 8 rollouts). This is the pre-existing `MAX_TRUNCATED` check being too strict for an honestly-rambling model. Math is immune (BFT forces an answer → never truncated → never hits the check).

**MATH @ thinking budget 16000, n=8:** close `</think>` on its own **65%**, would-need-BFT-force **35%**. The 35% forced rollouts are masked from the loss (DAPO) — as designed.

## Exp B/C — memory (measured directly on the 80GB H100)

**GRAIL verify (forward, 1 seq):** 2k 9.7 / 4k 10.8 / 8k 13.0 / **16k 17.4 GB** → comfortable.

**Training (policy + frozen KL reference = 2 models, grad checkpointing, + 8-bit optimizer):**
| seq | peak + 8-bit optim |
|---|---|
| 4k | 40.6 GB ✅ |
| 8k | 55.6 GB ✅ |
| **16k** | **85.7 GB → OOM (>80GB)** 🔴 |

Fits up to ~14.5k; **a 16k rollout OOMs.** Two models are resident (17 GB weights) and the run is on the **torch fallback** — the GatedDeltaNet fast kernels (`flash-linear-attention` + `causal-conv1d`) are absent because the box has no `nvcc` to JIT/compile them (`fla` TileLang wants Hopper `wgmma`, `causal-conv1d` needs nvcc). On the prod validator (which has the CUDA toolkit) the kernels install and cut this materially — but that must be **verified on a box with nvcc**; it could not be measured here.

**Code check:** the branch's DAPO masking zeros the forced rollout's *advantage* but `_build_microbatch_items` (training.py:1256) still **forwards** it (the KL term traverses it), so the 16k forced rollouts are processed → they drive the OOM.

---

## What has to happen before merge+deploy

**Blocker 1 — code `MAX_TRUNCATED` (10% honest reject).** Decide:
- (a) per-env allowance (e.g. 2-3 for code) — simplest, weakens the manufactured-loser guard;
- (b) an honest-loop-vs-manufactured check (reuse the `reward_shape` entropy signal) — robust, more work;
- (c) accept 10%. Recommend (b) or (a); do not ship blind.

**Blocker 2 — training 16k memory. ✅ RESOLVED 2026-07-28: THE BLOCKER DID NOT EXIST.**

Measured with the REAL `train_step()` (prod `PagedAdamW8bit`, gradient checkpointing, frozen KL reference, `MICROBATCH_MAX_PADDED_TOKENS` packing) on an H100 80GB, torch fallback (no fast kernels — the pessimistic case):

| scenario (cap 16384) | peak | rollouts forwarded |
|---|---|---|
| worst case — all 8 rollouts terminated AT the cap | **40.3 GB** | 16 |
| realistic (terminated ~p90 11k, 3 forced) mask off | 40.4 GB | 16 |
| realistic, mask **on** | 40.4 GB | **10** |

**~40 GB of 80 — roughly 2× headroom, in every scenario.** The earlier "85.7 GB → OOM" figure was an artefact of the synthetic probe, which called `model(ids, labels=ids)` and therefore materialised full logits over a 248k vocab; the real path (`_batched_completion_logprobs`) gathers only the selected token logprobs. The lesson: measure the function that production calls, not a stand-in for it.

**Correction to the mask's rationale.** The masked-rollout skip *does* take effect (16 → 10 rollouts forwarded — verified, not assumed), but it saves **compute, not peak memory**: peak is set by the longest single sequence in a micro-batch, which is a terminated rollout either way. The skip remains correct as DAPO fidelity (a masked rollout owes no PPO or KL gradient) — the memory justification in commit `1412a8b` is superseded by this measurement.

**Original analysis (superseded), kept for the record:**
- ✅ **DONE (commit `1412a8b`): skip masked (0-advantage) rollouts from the training forward.** The DAPO mask now skips the forced rollouts' forward entirely (not just zeroes the PPO advantage), so a forced ~16k rollout is never materialised — the longest trained sequence is the longest *kept* (terminated) rollout (~11-14k) → fits 80GB. Also excludes them from the N_e denominator, and is more DAPO-faithful (whole loss masked, incl. KL). 3 tests, 244 passed. **Must be re-verified with the real `train_step` + prod config on a kernel-enabled card** (the deferred test).
- Complementary: **install the GatedDeltaNet kernels on the validator** (fla + causal-conv1d; needs the CUDA toolkit) — also ~3.3× speed, and further cuts memory.

**Neither blocker is in the branch's new code being wrong** — the code is correct and tested. They are 4B+16k deploy-readiness issues. Once (1) is decided and (2) is arranged, the branch is mergeable.

## Appendix — numbers
Exp A code trunc 5.3% / >16k 5.5% / reject 10%. Exp A math close 65% / forced 35%. Verify 16k 17.4GB. Train(2 models,+8bit) 4k 40.6 / 8k 55.6 / 16k 85.7(OOM). Torch fallback (no fla/causal-conv1d; box has no nvcc). Host: H100 80GB, torch 2.11+cu130, sm_90.
