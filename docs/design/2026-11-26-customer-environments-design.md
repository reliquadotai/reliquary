# Customer environments: a private hub, run in the order's own sandboxes

Status: draft, nothing implemented.

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
* **Visibility**: private to the account (and its organisation's members). No sharing with other accounts and
  no public listing for now.
* **Uses**: one environment serves every product. Qualification (below) decides which uses it may have, as the
  catalog does for ours (`env_catalog.py`: `rl`, `batch`, `eval` rows).
* **Secrets**: the customer may attach secrets (an API key for their judge or their tools). They are stored
  encrypted and injected only into that environment's sandboxes.

A qualified version becomes a catalog entry owned by the account: `customer:<account>/<name>@<sha256[:12]>`. Only
that account's orders may name it.

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
5. **Determinism**: grade the examples twice, in two fresh sandboxes; any difference refuses the
   environment for RL (it may still serve evaluations and annotation, marked non-reproducible). An LLM judge
   is non-reproducible by construction.
6. Measure scoring time and sandbox seconds per item: the order's price includes them.

The steps run cheapest first and stop at the first failure, with its reason. The result is a signed
qualification record naming the version, the image digest, the prompt set's sha256 and the uses granted.

### Who pays

Qualification is CPU only (build, freeze, a few scorings). Each account gets a monthly allowance of
qualification CPU time; beyond it, qualification is billed at cost, pass or fail. A failure caused by our
infrastructure (builder, registry, fleet) is never counted. An account runs at most a few qualifications at once,
and build and start have fixed time budgets, so pushes cannot be used to tie up the builder.

## Data-only graders (no code)

Many customers need no code. `orders.graders` grows declarative graders, run where they run today:
numeric tolerance, multiple choice, regex, JSON schema, and code against customer tests given as data (run by our
existing code grader). An LLM judge is offered for evaluations and annotation only. These stay the default path.

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
