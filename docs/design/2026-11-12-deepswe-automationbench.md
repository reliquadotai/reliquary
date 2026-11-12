# DeepSWE v1.1 and AutomationBench as order benchmarks

Status: design, 2026-11-12. Implemented here: `reliquary-deepswe` is a bridged eval package
(`corpus.job.BRIDGED_EVAL_PACKAGES`), so `eval build-set --source reliquary-deepswe --bridge-file`,
`jobs create --eval-set` and admin orders accept it once the package exists. The platform branch
`feat/deepswe-gateway-group` adds the `deepswe` gateway image group. Nothing else exists yet.

## 1. DeepSWE v1.1 (signed sandbox, bridged)

Upstream: `datacurve-ai/deep-swe` (Apache-2.0), 113 tasks (35 TypeScript, 34 Go, 34 Python,
5 JavaScript, 5 Rust). Per task: `task.toml`, `instruction.md`, `pre_artifacts.sh` (writes
`/logs/artifacts/model.patch`), `environment/Dockerfile`, `tests/{Dockerfile,test.sh,test.patch,
grader.py,config.json}`. Image `public.ecr.aws/d3j8x8q7/swe-bench-202605:<id>-v1.1` (amd64), workdir
`/app`, 2 CPU / 8 GB / 20 GB, agent 5400 s, verifier 1800 s, `allow_internet = false` for both phases.
The verifier image is `FROM <task image>` plus `COPY tests/ /tests/`.

### Why the Prime tasksets cannot be served as they are

`deep-swe` 0.2.0 and `pi-deepswe` 0.3.0 (PrimeIntellect-ai/research-environments,
`environments/swe/`) both require `verifiers[harbor]>=0.4.1`; our pin b2e4e81 is v0.3.1. They call
`vf.grade_submission`, `HarborTask.grading_task`, `stage_tests`/`run_verifier` and
`HarborData.verifier_image`, none of which exist at v0.3.1, and on v0.3.1 `HarborTask` raises for a task
with a separate verifier. They also rewrite images to private `prime/primeintellect/*` mirrors, and
`pi-deepswe` runs `go mod download all` in setup, which needs network our boxes never have. The bridge
also needs `sandbox-images.lock.json` in the package and caps a grading step at 810 s.

### Design: our own `reliquary-deepswe` package (reliquary-environments)

As `reliquary-terminal` wraps Terminal-Bench: a verifiers v0.3.1 `HarborTaskset`
(`reliquary_deepswe:DeepSWETaskset`), split `eval` only (never train; held-out by construction).

- Tasks vendored in the wheel at upstream commit 8cae598 (text only; no download at load time, so the
  gateway, validator and miners load the same rows offline).
- `ignore_separate_verifier=True`: grading copies `tests/` into a pristine box of the task image, which
  is exactly the upstream verifier image. `Task.score` runs `tests/test.sh` and reads the CTRF verdict.
- `sandbox-images.lock.json`: the 113 public ECR tags pinned by digest (read from the registry, no pull).
- `pi-deepswe`'s fixes ported as data: its v17 grader (restores test files the agent changed, retries
  timeout-only failures), the `kcp-go` exclusion, the `eicrud` file-limit bump and the fixture fixes.
  Epoch AI flags at least 23 flawed tasks: the set excludes those the gold patch does not pass offline.
- Grading inside 810 s: a pre-launch gate runs every gold patch offline on our gVisor host and records
  the verify time. Tasks over ~700 s or failing offline (Go module fetches) are left out of the set.
- Tools `["bash", "edit"]`, harness `mini-swe@2` as the SWE set; budgets as SWE but `memory_bytes` 8 GiB
  and `wall_s` 5400.

### Wiring (after the package is built and gated)

1. Catalyst: nothing more than this branch (`BRIDGED_EVAL_PACKAGES`). Freeze the set on a host with the
   wheel: `reliquary eval build-set --source reliquary-deepswe --split eval --bridge-file bridge.json`
   (113 rows, or the gated subset), publish it.
2. Platform: `GATEWAY_SETS[<set id>] = 'deepswe'`, a `SUBNET_EPISODE_SETS` template (the SWE one with
   `env.package reliquary-deepswe`, `split eval`, its identity and lock), and the pinned values in the
   `deepswe` case of `scripts/sandbox-gateway/bootstrap.sh`; the kit gains the wheel and
   `groups/deepswe/{images,index}.txt`.
3. Disk: 113 images of several GB each per gateway; boot estimated at ~0.9 h (unmeasured).

Effort: package and gate ~4-5 days (gold-patch gate on a gVisor host is most of it), wiring 0.5 day.

## 2. AutomationBench (suite episode, validator replay)

Upstream: `zapier/AutomationBench` (MIT) 1.0.6: 6 scored domains x 100 tasks (+200 `simple`). Each task
is a dict: prompt, `info.initial_state` (a full world seed), `info.assertions`, `info.zapier_tools`.
SaaS APIs are simulated in pure Python (pydantic `WorldState`, ~47 services); grading is deterministic
end-state assertions (no LLM judge). Upstream "api" mode gives the agent 3 flat tools:
`api_search(query, top_k)`, `api_fetch(method, url, params: str, body: str)`, `base64_encode(text)`.
The PI fork's native schemas (`api_fetch` with a top-level `allOf/oneOf` per endpoint) and its
"harness" mode (every Zapier tool of the task's services, up to 550) are what `pi-automationbench`
serves; both Prime tasksets need verifiers >= 0.4.1 and `mcp>=2` (our pin needs `mcp<2`).

### Options

1. Suite episode (recommended). A registry `interaction_mode="episode"` env, like EnvScaler and
   telecom: the miner plays the world on CPU, the validator replays the tool calls and recomputes the
   score. Fits: the world is pure Python and offline; the 3 api-mode tools are flat JSON (no
   `allOf/anyOf`), the AA setup. Needs: seeding `uuid4`/`random` per episode (about 100 sites mint ids)
   and pinning `datetime.now()` to the task's scenario time; the `chatgpt` tool answers its offline
   stub. No gateway, no image, no sandbox.
2. Tool server in the signed sandbox. The gateway routes only `bash` and `edit`
   (`reliquary-tools/1`); arbitrary JSON tools would need a new tools protocol version, a harness that
   renders and forwards them, and record/verify changes in reliquary-sandbox. Weeks, for a CPU world
   that needs no isolation.
3. A CLI in a box (the world in a file, `api_fetch` as a shell command over bash). Works on today's
   signed path but changes the benchmark's interface (scores not comparable) and the model can edit
   the world file directly.

### What option 1 still lacks for orders

- The env package `reliquary-automationbench` (reliquary-environments, `tool_use/`): an
  `EpisodeEnvironment` over upstream pinned by commit, splits per domain plus `all`, the determinism
  patches above, and a registry spec in Catalyst (renderer `reliquary-chatml-tools-v1`). ~3 days.
- A drawn suite episode. Orders are forced-draw evaluations; `corpus.job._check_draw_job` refuses a
  draw on the suite harness. The validator side already takes draw jobs (`suite_intake` skips proofs,
  `suite_grading` marks drawn replays); missing are the miner (`agentic_miner.run_suite_miner` must
  build the forced core and `ForcedDraws`, as the signed path does), the `context_length` stop (a
  drawn episode is submitted however it ends: the corpus suite intake accepts only
  `agent_completed`/`max_turns`; the RL lease path's cut check, `validator.rl_suite_episodes`, is the
  model) and a GPU check of the draw audit on a multi-turn suite sample. ~3-4 days.
- The admin's `order_episode` (accept an `episode` set with the `suite` harness; the renderer check
  must accept `reliquary-chatml-tools-v1` for Qwen3.5-family models) and a platform template for
  suite sets (no gateway). ~1 day.

## 3. Miner side

- Bridged envs: every process serving the job (miner, intake, grader) must have the env package at
  the episode's exact identity, verifiers at b2e4e81, reliquary-sandbox at the job's commit and the
  renderers version (`agentic_swe.episode_support_refusal`, `sandbox_support_refusal`): miners install
  the `reliquary-deepswe` wheel (task text only, no images) as they do `reliquary-terminal`.
- Suite envs: the miner installs the env wheel and runs the world on CPU; no Docker.
- Renderer: episode orders play only Qwen3.5-family checkpoints (`episode_jobs.EPISODE_RENDERER_FAMILIES`,
  `renderers:qwen38`); other architectures need their renderer first.
