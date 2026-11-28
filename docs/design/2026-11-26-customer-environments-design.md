# Customer environments: a private hub, run in the order's own sandboxes

Status: steps 1, 2 and 4 implemented (`reliquary/orders/graders.py`, `reliquary/customer_env/`), not deployed.
Step 3 (platform) on its own branch, with order placement for batch, eval and RL. RL on customer envs done
(GPU smoke). Step 5 (agentic) is open: see
"Implementation status".

## Why

A customer of any product (batch, evaluation, RL training, SFT corpus) may need an environment we do not ship:
their own tasks, their own grader, their own tools. Today they can bring data only: an upload's lines with a
`reference`, graded by one of `orders.graders` (`boxed_math_v1`, `exact_match_v1`). Anything else needs us to
write the environment.

## What Prime Intellect does

* An environment is a Python package for `verifiers` (v1: a `Taskset` of `Task`s, `@vf.reward` methods,
  optional `setup`/`finalize` hooks and tools), pushed to the Environments Hub with versions and a private
  visibility, installed with `prime env install owner/name@version`.
* Self-hosted prime-rl runs each environment in subprocess workers of the orchestrator.
* Hosted training (`prime train`) gives each run its own trainer, inference, orchestrator and env-server pods; the
  customer names a Hub taskset id in the config. The environment's Python runs in the run's env-server pod.
* Prime Sandboxes (an OCI image per task, outbound allow-list, encrypted secrets) run the agent's commands and
  tools. Task hooks and rewards run on the evaluator, outside the sandbox, unless scoring is isolated.
* We found no signing, content hash or review of Hub environments; their automated push checks were removed.

Their isolation is the tenant's own pods. We take that, and add what our setting needs.

## What differs here

1. **Validators run environment code in-process.** `env.compute_reward` is called by the batcher
   (`validator/batcher.py`) and by the honesty re-check (`validator/verifier.py`). A customer's code there would
   sit next to the hotkey, the state and our credentials.
2. **Miners must never see the grader or the references**, or they optimise against them and the customer's data
   leaks.
3. **A score must be reproducible**: the audit re-grades, and pay follows the grade.
4. **Admin and eval hosts serve several customers.** Orders grade on the admin host today.

An RL training already has its own trainer machine and, when agentic, its own sandboxes. The hole is single-turn
environments, graded on the validator, and order grading on the admin host.

## The rule

**A customer environment is an image, and it only ever runs in the sandboxes of the order that uses it.** The
validator and the admin host serve frozen prompts and receive numbers; they never import customer code.

An agentic environment runs its episode and its scoring there. A single-turn environment is the degenerate case:
zero tool turns, scoring only.

## The customer's hub

Each account has a private environment hub on the platform:

* **Push**: a `verifiers` v1 package (a source archive with `pyproject.toml` and a lockfile), or an import from
  the Prime Environments Hub by `owner/name@version` (we fetch the wheel once and treat it like a push).
* **Version** = sha256 of the archive. A version is immutable; an order pins it, with the image digest it built.
* **Visibility**: private to the account (and its organisation's members) by default. Its owner may make a
  qualified version **public**: any account may then order it (batch, evaluation, RL), never change it, and pays
  its scoring like its own. Public is final for that version (orders placed on it keep it); the owner may take it
  out of the catalog (**listed**) for new users to find, not out of service. A version is public only with a
  license, and its prompts are then visible to whoever orders it: a hidden benchmark stays private.
* **Source**: an imported env says where it comes from (`huggingface` or `prime_hub`, the upstream owner, name,
  link and revision, the upstream license). It is part of the archive, so frozen in the version's sha256, and is
  shown wherever the version is listed: an import is never shown as ours, the account that imported it is not
  its author.
* **Uses**: one environment serves every product. A qualified version may serve RL, batch and evaluation alike:
  RL is a task like any other (see Determinism below).
* **Secrets**: the customer may attach secrets (an API key for their judge or their tools). They are stored
  encrypted and injected only into that environment's sandboxes.

A qualified version becomes a catalog entry owned by the account: `customer:<account>/<name>@<sha256[:12]>`. Only
that account's orders may name it, unless it is public.

Our public catalog of imported envs (Hugging Face Harbor datasets, the Prime Hub) is a Reliquary account that
imports them as any customer would and publishes those that pass: a redistributable license, no network, no
judge, examples (or reference solutions) that score as expected.

## The package

A source archive (`.tar.gz`, at most 50 MB packed, 200 MB unpacked; regular files and directories only, no
absolute path, no `..`, no link, no device) holding:

* `pyproject.toml`, with `[tool.reliquary]`: `taskset` (the verifiers v1 taskset id it registers), optional
  `args` (the taskset's config fields), `kind` (`single_turn`, the default, or `agentic`), `license` (an SPDX
  id; else `[tool.reliquary.source] license`, else `[project] license` when it is a short name) and, for an
  import, `[tool.reliquary.source]`: `hub`, `owner`, `name`, optional `url` (https), `revision`, `license`;
* `requirements.lock`: every dependency pinned with `--hash=sha256:` (pip `--require-hashes` format);
* `examples.jsonl`: 2 to 64 lines `{"key", "answer", "expected_score"}`, at least one at or above 0.5 and one below;
* the package's sources.

The version is the archive's sha256. The environment id is `customer:<account>/<name>@<sha256>`; `<account>` is
the platform account id, `<name>` matches `[a-z0-9][a-z0-9-]{0,62}`.

## Admin API (platform to subnet admin)

The platform stores the archive in its bucket under `customer-envs/<account>/<name>/<sha256>.tar.gz` and calls:

* `POST /admin/v1/customer-envs` `{account, name, object_key, sha256}`: starts (or returns, idempotent on
  `sha256`) the qualification. Answers `{env_id, status}`.
* `GET /admin/v1/customer-envs?account=<id>`: that account's versions.
* `GET /admin/v1/customer-envs/<env_id>`: one version: `{env_id, account, name, sha256, kind, status, reason,
  uses, image_digest, prompt_set, examples, cost, created_at, qualified_at, license, source, public, listed,
  published_at}`.
  `status`: `queued` | `building` | `qualifying` | `qualified` | `refused` | `disabled`; `reason` says why a
  refusal (`package_invalid`, `build_failed`, `start_failed`, `freeze_failed`, `examples_failed`,
  `infrastructure` (retried, never billed)); on a qualified version, `nondeterministic` is a note, not a
  refusal.
  `uses` ⊆ `rl`, `batch`, `eval`. `prompt_set`: `{count, sha256}`. `cost`: `{qualification_cpu_seconds,
  scoring_cpu_ms_per_item}`.
* `DELETE /admin/v1/customer-envs/<env_id>`: disables it; running orders keep their pinned version. Refused
  (409) for a public version: its owner unlists it instead.
* `POST /admin/v1/customer-envs/visibility` `{env_id, account, public?, listed?}`: the owner's change (the
  platform checked who asks). `public: true` needs a qualified version with a license, lists it unless
  `listed: false`, and is final (409 `customer_env_public_is_final`); `listed: true` needs public.
* `GET /admin/v1/customer-envs/catalog`: the public, listed, qualified versions, newest published first.

An order names `env: "customer:<account>/<name>@<sha256>"`; the admin refuses it unless the order's account is
`<account>` or the version is public, and the version's `uses` has the order's product.

## Build

* The archive is built into an OCI image by a builder that holds no secret and has egress only to package
  indexes. Dependencies come from the lockfile, hash-pinned; an archive without hashes is refused.
* The base image is ours (Python, `verifiers` at the pinned `VERIFIERS_COMMIT`, the reward server below). The
  customer's image may extend it for an agentic environment (tools, a repository, system packages), with limits on
  size and layers; `--privileged`, host mounts and setuid binaries are refused.
* The image is pushed to our registry by digest. The sandbox fleet pulls it like any pinned env image
  (`sandbox/fleet.py` already places a session only on a machine holding the pinned image and package).

## The reward server

The image's entrypoint is a small server, in the spirit of the zero-secret grader agent
(`environment/grader/remote.py`):

* `POST /score`: a batch of `{problem_id, frozen_sha256, completion}` (or a signed episode trace for an agentic
  task) returns one float per item, or an infrastructure error.
* It loads the taskset once at start, then scores with `Task.score` in a child per call with a timeout, as
  `eval/verifiers_source.py` does today (`ChildScorer`).
* It runs under `runsc --network=none`, read-only root, no secret mounted (except the customer's own, only when
  they declared that scoring needs it), CPU, memory and pid limits.
* Infrastructure failures keep the `GraderInfrastructureError` rule: a dead or slow reward server never scores a
  zero and never faults a miner.

The validator side is a client of the same shape as `grader_client.py`: a group is scored in one call (the
existing `precompute_rewards` path), and the honesty re-check calls the same server.

## Where it runs

| product            | runs on                                                                  |
|--------------------|--------------------------------------------------------------------------|
| RL training        | the training's own sandboxes (CPU slots for single-turn, the episode sandboxes for agentic) |
| batch, SFT corpus  | sandboxes allocated to the order, only when the order filters or annotates with the env |
| evaluation         | sandboxes allocated to the evaluation                                     |

For a single-turn environment the isolation is the container, not the machine: its reward server may run on the
training's trainer box (already the customer's) or as one `runsc` container per environment version on a shared
pool of CPU reward hosts. Only an agentic environment needs gateways of its own.

Never on a validator, never on the admin host, never on a miner's machine. A miner gets prompts (token ids) and,
for agentic tasks, a session on our gateway; never the image, the grader or the references.

## Billing of scoring

* Catalog environments (math, code, ...): unchanged; their grading is part of the token price.
* Agentic environments, ours or a customer's: gateway-hours, as today.
* A customer's single-turn environment: qualification measures its scoring CPU-seconds per item. Under a
  threshold (a math-like check), scoring is included in the token price. Above it (long customer tests), the
  order is quoted with scoring CPU-seconds at the reward pool's cost, billed on actual use, capped at the quote.

## Prompts are frozen at qualification

The validator must not run customer code to obtain prompts. At qualification, the taskset is loaded once in its
sandbox and its tasks are written as a prompt set (token ids, `set.json`, sha256), the format uploads and our eval
sets already share (`orders/uploads.py`, `eval/verifiers_source.py` `freeze`). References stay in the image. A
taskset whose tasks are generated per draw (procedural environments) is served by the reward server's
`POST /task` instead, seeded by the draw, and its output is recorded with the sample.

## Qualification

Run once per version, in the customer's sandbox, before any order may use it:

1. Static checks: the package exposes a v1 `Taskset`; size and row limits; refusal reasons as in
   `verifiers_source.refusal` (prompt not text, conversation in the prompt).
2. Build and start; the reward server must answer within its start budget.
3. Freeze the prompts.
4. **Examples**: the push carries a few examples, `{problem_id, completion, expected_score}`, at least one
   scoring high and one low. Each must score as expected. No model runs during qualification: it needs no GPU.
5. **Determinism**: grade the examples twice, in two fresh sandboxes; a difference is reported
   (`reason: nondeterministic`) and refuses nothing. One reward host scores each completion and the validators
   receive its floats: nobody re-scores it, so there is no second score to disagree with, and the miner's
   tokens are fixed by the draw, so a noisy grader gives it nothing to exploit. An LLM judge is
   non-reproducible by construction; training on one is the customer's choice.
6. Measure scoring time and sandbox seconds per item: the order's price includes them.

The steps run cheapest first and stop at the first failure, with its reason. The result is a signed
qualification record naming the version, the image digest, the prompt set's sha256 and the uses granted.

### Who pays

Qualification is CPU only (build, freeze, a few scorings). Each account gets a monthly allowance of
qualification CPU time; beyond it, qualification is billed at cost, pass or fail. A failure caused by our
infrastructure (builder, registry, fleet) is never counted. An account runs at most a few qualifications at once,
and build and start have fixed time budgets, so pushes cannot be used to tie up the builder.

## Data-only graders (no code)

Many customers need no code. `orders.graders` holds declarative graders, run where they run today:
`numeric_v1` (tolerance), `choice_v1` (multiple choice) and `json_match_v1` are in. Still to come: JSON schema
(no `pattern` keywords: the admin host runs no customer regex), code against customer tests given as data (run by
our existing code grader), and an LLM judge for evaluations and annotation only. These stay the default path.

## Our own environments

Catalog environments are our code and may keep grading in-process. Moving them behind the same reward server is a
later choice: one path for everything, at the cost of a round trip per group.

## Plan

1. Declarative graders and LLM judge for orders.
2. Reward server, validator client, and single-turn customer tasksets in RL and evaluation orders.
3. Platform hub: push, versions, secrets, qualification report.
4. Prime Hub import.
5. Agentic customer images on the gateways (after agentic orders serve training splits).

## Open questions

* The allowance's size, once qualification costs are measured.
* Latency of the reward server round trip on the RL critical path: to be measured on a group of 512.

## Implementation status

| piece | where | state |
|---|---|---|
| declarative graders `numeric_v1`, `choice_v1`, `json_match_v1` | `orders/graders.py` | done |
| package checks (shape only, never unpacked on the admin host) | `customer_env/package.py` | done |
| our Dockerfile, base image, build and push by digest | `customer_env/image.py`, `docker/customer-env/Dockerfile.base` | done; base not pushed (`BASE_IMAGE` is a placeholder digest) |
| reward worker (stdio JSON lines, protocol on its own descriptor) | `customer_env/worker.py` | done |
| sealed container (gVisor, no network, read-only, nobody, limits) and its client | `customer_env/runner.py` | done |
| reward host and builder host (mTLS, our registry only, warm or fresh sandbox) | `customer_env/host.py` | done; no host deployed |
| version index, qualification (examples, determinism, warm cost) | `customer_env/registry.py`, `qualify.py` | done |
| admin routes, order sets through the upload compiler | `customer_env/admin.py`, `orders.py` | done |
| eval grading of customer rows on the reward host | `eval/grading.py`, `customer_env/scoring.py` | done |
| customer env as a registry environment (frozen rows, remote rewards) | `customer_env/rl.py`, `environment/registry.py` | done |
| packing tool, Prime Hub import (`prime env pull`) | `customer_env/pack.py` | done |
| RL training orders naming a customer env (carried contract, lease settings, miner opt-in, rows from the controller) | see `2026-11-26-rl-settings-per-task.md` | done; GPU smoke passed |
| platform orders on a customer env: batch and eval quotes from `order-sets`, RL environments with settings | reliquary-platform `feat/customer-environments` | done (contract tests), not deployed |
| scored batch rows | an eval order on the env | done: every sample with its score (`graded.parquet`), all audited |
| batch `filter`/`annotate` at a partial audit | orders delivery | **open**: no platform order declares a job filter today, on any env; needs `CreateJob.filter` and a customer-env `job_grader` |
| agentic customer images on the gateways | reliquary-sandbox | **open** (refused today as `agentic_not_supported`) |

Verified: unit tests (`tests/unit/test_customer_env_*.py`, `test_order_reference_graders.py`) and a Docker run
(`RELIQUARY_DOCKER_TESTS=1 pytest tests/unit/test_customer_env_docker.py`, runc here; gVisor to run on the reward
host): the toy env packs, builds, qualifies for every use at 2.5 ms per item, and from inside the sealed container
has no egress, no write outside `/tmp`, no exec in `/tmp`, no setuid.

## Deploying

1. Push the base: `docker build -f docker/customer-env/Dockerfile.base` to `registry.reliqua.ai/customer-envs/base`,
   set `image.BASE_IMAGE` to its digest.
2. A builder host (no secret; registry push; egress to package indexes): `python -m reliquary.customer_env.host
   --build` with TLS cert, key and client CA.
3. Reward hosts (no secret; registry pull; `runsc` installed as a Docker runtime): `python -m
   reliquary.customer_env.host` with TLS. One per training for RL (the training's own sandbox), a shared pool for
   evaluations.
4. Admin host: `RELIQUARY_CUSTOMER_ENV_DB`, `RELIQUARY_BUILDER_URL`, `RELIQUARY_REWARD_HOST_URL`,
   `RELIQUARY_REWARD_HOST_CLIENT_CERT`/`_CLIENT_KEY`/`_CA`. Unset, the routes answer 503.
5. Run the Docker test on a reward host with `RELIQUARY_DOCKER_RUNTIME=runsc` before opening the routes.
