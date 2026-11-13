# SWE-bench Verified, full set, leaderboard settings (2026-11-13)

Goal: an agentic eval order that runs the whole of SWE-bench Verified with the settings of the
official bash-only leaderboard (swebench.com, mini-SWE-agent), so a lab can order it for its
model and compare the number with the published ones.

Status: code on local branches, nothing pushed, merged or deployed.

- Catalyst `feat/swe-verified-full`: set exclusions, episode profiles, report fields.
- Platform `feat/swe-verified-full`: the set, the `profile` order field, gateway group `swe-full`
  with a read-only registry mirror, migration 0062.

## 1. The set

`reliquary-swe-eval-r0-n500-x7-k493-s7`: SWE-bench Verified as `reliquary-swe` serves it (split
`eval`, 500 rows), with 7 rows excluded and the other 493 in a seeded order (seed 7).

```
reliquary eval build-set --source reliquary-swe --split eval --start 0 --count 500 \
  --sample 493 --seed 7 --exclude-file exclusions.json --out <dir>
```

- `prompts_sha256` 1c476360c360caa8ac475e368f065e41b2ae571a4e2f154a3dd30ca913a32103
- `grading_sha256` 5b65c824d19503f5e9e820d57882f67cb8e1e00c8d514565389e573959ce28ca
- The card lists the exclusions with their reasons (`selection.exclude`).
- The first 100 rows are exactly `reliquary-swe-eval-r0-n500-x7-k100-s7` (same seed, same
  exclusions), so a 100-task order on either set plays the same tasks.

`--exclude-file` was on an unmerged branch (`feat/agentic-eval-sets`, 7d8f1d60). It is
cherry-picked here, and the existing 100-task set was built with it.

The 7 excluded tasks have tests that need the internet. Our sandbox boxes have none, so even the
reference patch scores 0 on them:

| task | why |
|---|---|
| psf__requests-1724, -1766, -1921, -2317 | `test_requests.py` at the base commit calls live http://httpbin.org |
| sphinx-doc__sphinx-7985, -8269, -8475 | `tests/test_build_linkcheck.py::test_defaults` checks live https://www.google.com links |

The leaderboard scores over all 500. When a whole set with exclusions is ordered, the report
restates pass@1 over 500 twice (`full_benchmark`): once with the 7 counted as failed, once as
passed. The leaderboard number to compare with lies between the two, which are 1.4 points apart.

## 2. Task images from our registry

All 493 images are mirrored into the Hippius bucket `reliquary-task-images` (distribution 2.8.3
layout). Each keeps its original repo path (`swebench/sweb.eval.x86_64.*`) and manifest digest.
These are exactly the digests `reliquary-swe`'s `sandbox-images.lock.json` pins (493/493 equal),
so the gateway's task images, the lock and the set agree.

### Gateway group `swe-full`

`swe-full` is a new image group (platform `GATEWAY_SETS`) that serves the full set. Its kit
carries `groups/swe-full/{images,index}.txt`: the 493 pinned references and source indices, in
set order.

`bootstrap.sh` sets up the box as follows:

1. When the provisioner pushed `/root/registry.env`, the box runs `registry` 2.8.3 (pinned by
   digest) on `127.0.0.1:5000`.
   - It sits on the bucket in read-only mode: `maintenance.readonly`, no upload purging.
   - It uses path-style addressing, region `decentralized`.
   - Blob reads redirect to presigned Hippius URLs.
2. Docker lists that registry first in `registry-mirrors`, then mirror.gcr.io.
3. Docker asks the mirror for every Docker Hub image. Pulls stay by digest, so the content is the
   same whichever source serves it, and a miss falls through to the next source.
4. The lock check, the parallel pre-pull and the task-cache warm-up are unchanged.

### Credentials

The box holds only `registry.env`: `REGISTRY_STORAGE_S3_ACCESSKEY` and `..._SECRETKEY` of a
Hippius sub token.

- The token exists: sub token `gateway-registry-ro`, read-only, on bucket `reliquary-task-images`
  only. Reading works; writes, deletes and other buckets are refused. Its env file is in the
  operator's kit directory, never in a repo.
- The bootstrap refuses a key that does not start with `hip_`.
- The provisioner pushes the file (mode 0600) only to the groups that pull from the registry. It
  reads the file from `GATEWAY_KIT_DIR/registry.env`.

### Boot and disk

The 493 images are ~131 GB compressed and ~340 GB extracted (2.6x, measured on 40 of them), plus
each session's writable layer. So `swe-full` gateways are rented as `cpu-large-amd`
(`GROUP_RESOURCE`: 16 vCPU, 1.5 TB disk) whatever `TARGON_GATEWAY_RESOURCE` says.

The Worker allows a `swe-full` gateway 2 h of boot (`GROUP_BOOT_HOURS`) and 90 extra minutes
before giving it up (`GROUP_READY_EXTRA_SECONDS`). The provisioner waits 90 extra minutes too.
At the measured rate, the pulls alone take ~40 min on 4 vCPU.
The bootstrap's own `docker pull` has a 1,800 s timeout per image.

### Validated on a throwaway Targon VM (cpu-small-amd, 4 vCPU, 2026-11-13)

Setup: the bootstrap's registry block and daemon.json, with our master key on the box (the VM was
destroyed after the test).

- 3 pinned images (django-11163, sympy-12419, pytest-5840) pulled in 48 s, 8 in parallel.
- The registry log shows the manifests (9 hits) and blobs (30) served from our bucket
  (`?ns=docker.io`).
- `busybox:1.36.1`, which is not in the bucket, fell through to the next mirror.
- A push to the registry was refused with 405 (read-only).
- A first run found that `REGISTRY_STORAGE_MAINTENANCE_UPLOADPURGING_ENABLED=false` makes
  registry 2.8.3 panic, and the pulls silently fell back to mirror.gcr.io. The bootstrap now
  passes `REGISTRY_STORAGE_MAINTENANCE_UPLOADPURGING='{"enabled": false}'` and exits if the
  registry is not up.
- A batch of 40 set images, 8 in parallel, all came from our bucket. Every blob GET answered
  307 (a presigned Hippius URL).
  - Time: 364 s.
  - Size: 21 GB compressed, 55 GB extracted.
  - Operator notes: `~/runbooks/swe-full/vm_registry_test.out`, not in the repo.

## 3. The leaderboard profile

Reference: mini-SWE-agent `config/benchmarks/swebench.yaml` (v2.0.0; the 2026 leaderboard rows
are labelled 2.0.0 and their trajectories say 2.1.0). Its settings:

- `step_limit: 250`
- `cost_limit: 3`
- per-command `timeout: 60`
- outputs over 10,000 characters shown as their first and last 5,000
- submission by `echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT`
- temperature unset for 2.x
- one attempt per task, 500 tasks
- a run that hits a limit (`LimitsExceeded`) submits nothing: a failure

No context cap is documented: the model's own window applies.

### What the profile pins

`episode.profile = "swebench-bash-only"` (`corpus.job.EPISODE_PROFILES`). It is an optional
manifest field, written only when set, so other manifests hash as before.

| | default SWE eval | `swebench-bash-only` |
|---|---|---|
| harness | mini-swe@2 | mini-swe@2 (required) |
| turns (LM calls) | <= 64 (48 offered) | <= 250 (250 offered) |
| context (prompt + transcript) | <= 60,000 | <= 131,072 |
| per-command timeout | 600 s | 60 s (required) |
| sandbox calls | 192 | 512 (required >= the 250-step cap) |
| sandbox wall | 7,200 s | 14,400 s (required; the gateway's cap) |
| run cut by a model limit | turns, context: scored on the box state | scores 0, listed apart |
| run ended by an infra limit | slot redone | slot redone (same draw), counted per task |

- `mini-swe@2` (reliquary-harness, digest cecffb26) is already the leaderboard harness: same
  system and instance texts, bash tool calls, format-error text and submit command, with an
  upstream parity test at mini-swe-agent 04d809ce.
- The profile is accepted only on eval jobs, on `reliquary-swe`, in the signed sandbox.
- The corpus trajectory wire takes the profile's bounds (`MAX_PROFILE_TRAJECTORY_*`). Every job's
  own bounds are still checked at intake. The service-contract and RL caps are unchanged.

### Limits: who controls them

A limit is the model's or the infrastructure's. We cannot tell why a miner's episode died (slow
GPU, crash, a busy gateway), so only what the model's own actions decide may cost it a point.

- **Model limits score 0** (as the leaderboard's `LimitsExceeded`): the step cap (`max_turns`),
  the context (`context_length`), and the box's budgets the model's calls spend. Those are the
  call budget (`max_calls`), CPU time (`cpu_s`) and the transcript cap (`transcript_bytes`); the
  machine closes the episode `budget_exhausted` (stop `episode_closed`).
  - The call budget is required at least the step cap, so it does not cut before it when the
    model makes one call per step. A model that makes several calls per step can still hit it:
    that is the model's doing.
  - Such a close is submitted like a graded episode: the session holds its slot
    (`CorpusEngagements.submittable_close`). The miner returns it for submission, and the intake
    takes it as a paid row with reward 0. The grader marks it `closed`, and the report counts it
    under `limit_cuts` as `budget:<reason>`.
  - Without a profile nothing changes: such a close stays unpaid and its slot is redone.
- **Infra limits score nothing; the slot is redone.** These are the wall budget (`expired`), a
  failed box (`box_failed`), a lost miner (`lapsed`), a lost gateway (`voided`), a failed open or
  an abort. The redo is not an extra attempt. The sample of a slot is fixed by the job's public
  draw (`corpus.draw_leases`, `corpus.eval_draw`), so whoever replays the slot replays the same
  sample; only tool-output nondeterminism differs.
  - The wall budget is sized generously: 14,400 s, the gateway's cap, for 250 steps of at most
    60 s each plus generation.
  - The report counts the redone slots per task and reason (`provenance.redone_slots`).

### How an order picks it

Platform `POST /evaluations/quotes` takes `profile: "swebench-bash-only"`.

- The profile is offered on the two SWE sets and listed in the catalog entry (`profiles`).
- The order's other knobs stay bounded by the profile.
- The quote's `pricing.episode` and the order view carry `profile`, and the job's `episode`
  carries it to the subnet.
- The delivered report's `provenance.episode` records env, harness, profile, turns, context, the
  per-command timeout and the wall budget.

### Quote: hours and price

The gateway quote takes the profile's episode estimates: 2,700 s expected and 7,200 s for a long
one, against 900 s / 2,700 s by default. The single longest episode is taken at the wall budget
(14,400 s).

The worst case for 493 tasks x 2 samples (986 episodes) is:

```
boot 2 h + 986 x 7,200 s / slots + 14,400 s + 0.5 h
```

With the default 78 slots this is 31.8 h, over the 24 h run cap (`EVAL_RUN_TIMEOUT_SECONDS`). So
`swe-full` plans 156 sessions (`GROUP_ORDER_SESSIONS`): 4 gateways of 39, giving 19.1 h. That
needs 4 of the 5 `GATEWAY_HOSTNAMES`, and each of the 4 gateways pulls all 493 images.

The token bound must fit too. A quote's `max_tokens` is rows x min(max_new_tokens x turns,
context) = 986 x 131,072 = 129 M. With `max_hours = max_tokens / eval_subnet_throughput`, a full
order fits 24 h only if the prod subnet throughput row is at least ~5.4 M tokens/h. Otherwise the
quote refuses with `exceeds_generation_deadline`. To check before launch.

### Differences left (to state in the report or fix)

- **Grading.** We grade the box's final state. mini grades the `patch.txt` the agent submits.
  Both apply the test patch over it. A stray reproduction script is harmless in both. A submitted
  patch that leaves out an edited file would differ.
- **Redone slots.** A redo replays the same draw, but the tools' outputs (timings, test order)
  may differ. A task redone many times is listed in `redone_slots`. A slot whose redos exhaust
  its attempts counts as a failure in the headline (missing row).
- **CPU budget.** We count `cpu_s` as a model limit: the CPU time of the commands the model ran.
  The leaderboard has no CPU cap. If test-heavy tasks hit 3,600 s, raise the gateway's
  `EPISODE_MAX_CPU_S` and the profile's budget, rather than reclassify it.
- **Cost limit.** No equivalent: there is no $3 cost cap. The 250-turn and context caps bound a
  run.
- **Transcript budget.** The gateway's transcript budget is 8 MiB, with outputs capped at 64 KiB
  per call. 250 calls with large outputs could end a run `budget_exhausted` before 250 turns. That
  is a model limit here (a scored 0); the leaderboard has no such cap.
- **Sampling and thinking.** Sampling is the order's. The leaderboard leaves temperature to the
  provider default, so we recommend the model card's sampling. Episodes render with thinking off
  (renderer `qwen38`).
- **Models.** Only Qwen3.5/3.8-family checkpoints have an episode renderer today. Their 262k
  window covers the 131,072 cap.
- **Audit at 128k.** The forced-draw audit re-reads every episode up to its context. Gate M3
  measured 60,000 tokens as one 80 GB audit GPU's prefill. A 131,072-token audit must be checked
  on the audit GPU before a profile order runs. Miners need a vLLM `max_model_len` of 131,072.

## 4. The report

These were already in place:

- pass@1 = mean over problems of correct/samples, with a 95 % bootstrap CI (unit: problem, seed
  fixed).
- pass@k for k <= samples.
- Missing and ungraded samples count as failures in the headline (`excluding_ungraded_and_missing`
  leaves them out).
- `stop_reasons`.

New in this work:

- The signed grade document keeps `stop`, so `stop_reasons` is filled for signed SWE.
- `limit_cuts`: `{rows, by_stop, box_state_passed}`, when a profile counts limit cuts as failures.
  It shows how many cut runs had a passing box anyway.
- `limit_cuts` also counts the model-budget closes (`budget:max_calls`, `budget:cpu_s`,
  `budget:transcript_bytes`).
- `full_benchmark`: the over-500 bracket (section 1).
- `provenance.episode`: the run configuration (section 3).
- `provenance.redone_slots`: `{redone, by_reason, by_problem}`. These are the sandbox sessions of
  the job that ended with no row (infra ends), read from the order control's session store
  (`order-sessions/`, its last days). They count sandbox episodes only: a lease that expired
  before its miner opened a box is not among them. The block is absent when the store cannot be
  read.

## 5. Remaining steps, in order

1. Review, then push and merge Catalyst `feat/swe-verified-full` and platform
   `feat/swe-verified-full`. The platform branch adds D1 migration 0062 (table rebuild for the
   CHECK on `image_group`). Apply it with the deploy.
2. Release miners with this branch. Old miners refuse a manifest with `profile` (unknown field)
   and cannot submit a model-budget close. The default-profile SWE and TMax orders are not
   affected.
3. Run a GPU check: draw audit and miner at 131,072 tokens on the audit GPU type.
4. Provisioner: swap its code copy to the merged script. Point `GATEWAY_KIT_DIR` at a directory
   holding `~/runbooks/swe-full/kit/kit.tgz` (kit-prod plus `groups/swe-full`), its
   `registry.env`, and kit-prod's `gw.crt`, `gw.key` and `validator-keys.json`.
5. Publish the set to the subnet bucket (`reliquary eval publish-set <dir>`; the set is in
   `~/evalsets-work/sets-agentic/reliquary-swe-eval-r0-n500-x7-k493-s7`).
6. Add the set to the platform D1 catalog and its platform bucket copy, as was done for the
   100-task set.
7. Check the prod `eval_subnet_throughput` row against 129 M tokens / 24 h (section 3).
8. Order control image with this Catalyst; Worker from main with prod config
   (`~/runbooks/gateway-fleet/GO-LIVE.md`).
9. Test orders:
   - 10 tasks x 2 under the profile on the 100-task set (`swe` group), to check the profile end
     to end;
   - then a few rows of the full set, to check a `swe-full` gateway boot and its pull time;
   - then the full order.
