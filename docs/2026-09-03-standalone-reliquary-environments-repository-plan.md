# Standalone Reliquary Environments Repository

Research, architecture, and end-to-end implementation plan, 2026-09-03.

Status: implementation-ready proposal. This document creates no repository,
publishes no package, and changes no active Reliquary protocol profile.

This plan supersedes only the repository-placement assumptions in the
2026-09-01 verified-improvement plan. Its product thesis remains unchanged:
Reliquary should interoperate with open environment and trainer standards while
owning qualification, replay, verified-improvement lineage, compute routing,
and settlement.

## Executive decision

Create a separate public repository:

```text
github.com/reliquadotai/reliquary-environments
```

Use it as Reliquary's curated, first-party environment catalog and conformance
repository. It should be a monorepo in source control, but every environment
inside it must be an independently installable and independently versioned
Python project.

Do not move or delete the three current Episode v1 implementations from the
`reliquary` repository yet. Freeze them as legacy consensus and historical
replay implementations. Port their behavior into new, Verifiers-v1-native
packages with new environment identities, then cut Reliquary over through a
new signed protocol profile after cross-repository qualification.

The target relationship is deliberately one-way:

```text
Verifiers v1 + optional OpenEnv transport
                 |
                 v
       reliquary-environments
      tasksets, worlds, rewards,
      tools, fixtures, manifests
                 |
        immutable release artifacts
        wheel + OCI + digests
                 |
                 v
             reliquary
   admission, replay, GRAIL, market,
   trainers, Bittensor, settlement
```

An environment package must never import Reliquary core. Reliquary may consume
an exact released environment artifact. This prevents a circular dependency
and keeps the environment catalog useful outside Bittensor.

## The most important migration rule

The current v1 manifests do not merely identify source code semantically. They
hash repository-relative path names and file bytes with
`sha256-path-nul-file-sha256-newline-v1`. The resulting manifest digests are
hard-coded into the environment registry and the signed development protocol
profile.

Therefore, moving byte-identical v1 files to a new repository would still
change their implementation digest. Renaming an import shim would also change
the committed implementation. An in-place move would invalidate the qualified
contract and make historical replay ambiguous.

The safe rule is:

> Copy and version forward; do not move and pretend identity was preserved.

Concretely:

1. Keep embedded `reliquary_*_tools_v1` code and fixtures byte-for-byte stable.
2. Create standards-native packages under new IDs, initially alpha releases.
3. Compare old and new semantic results in shadow mode.
4. Bind the new wheel and OCI digests in a new release lock.
5. Activate only through a new Reliquary protocol profile and explicit network
   upgrade.
6. Retain old artifacts for as long as any checkpoint, receipt, archive, or
   dispute can refer to them.

## Research findings

### Prime's useful pattern is separation, not cloning

Prime separates its environment catalog, environment abstraction, and trainer:

- [`prime-envs`](https://github.com/PrimeIntellect-ai/prime-envs) is its
  first-party catalog. Environments are grouped by category but installed and
  evaluated as individual packages.
- [`community-environments`](https://github.com/PrimeIntellect-ai/community-environments)
  separates lightly reviewed community supply from the first-party catalog.
- [`verifiers`](https://github.com/PrimeIntellect-ai/verifiers) defines the
  task, taskset, harness, runtime, trace, episode, and evaluation contracts.
- [`prime-rl`](https://github.com/PrimeIntellect-ai/prime-rl) consumes those
  environment contracts in a separate training system.

The current Verifiers documentation says that new environments should target
v1. A package exports a `Taskset`; tasks own setup and scoring, harnesses own
interaction behavior, runtimes own execution, and traces retain the interaction
graph. See the current [Taskset guide](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/tasksets.md)
and [architecture overview](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/architecture.md).

Current `prime-rl` configs consume this shape directly. Each training source
selects an `env.taskset.id`, agent harness, runtime, and optional relative
sampling ratio. This makes a native Verifiers package the shortest path to an
unchanged `prime-rl` smoke and training run. See the official
[`prime-rl` configuration guide](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/configuration.md).

### OpenEnv is an adapter target, not the canonical reward implementation

[OpenEnv](https://github.com/huggingface/OpenEnv) provides a useful neutral
deployment and interaction socket built around `reset()`, `step()`, and
`state()`, plus container and HTTP/WebSocket support. Its own README still
describes the API as experimental. MCP adoption is also still evolving: the
[MCP guide](https://github.com/huggingface/OpenEnv/blob/main/docs/source/tutorials/mcp-environment.md)
distinguishes the training control plane from the agent-facing tool surface.

The design decision is therefore:

- Verifiers v1 is the canonical task/reward source for the first release.
- Pure domain logic is shared by thin adapters.
- OpenEnv is a separately installable facade and container, not a required
  dependency of every native environment wheel.
- MCP is used when tools cross a process or network boundary. Local pure
  functions do not gain an MCP dependency just for branding.
- Reliquary does not create its own competing generic `step()` protocol.

At the 2026-09-03 research snapshot, OpenEnv's Verifiers importer still looked
for the older `load_environment` convention rather than current v1 Tasksets.
The initial implementation must not depend on automatic `openenv import` for
correctness. It should ship and test its own small v1-aware facade until the
upstream importer supports the current contract.

Also, `openenv validate` is a useful packaging and live-endpoint gate, not a
complete semantic, statistical, provenance, or security qualification system.
The current main branch [reverted the proposed RFC 008 validation stack](https://github.com/huggingface/OpenEnv/commit/b9d8c1f953e0c3e0bbee2f3f6f6c73d8eae61f5f).
Pass the [actual OpenEnv validator](https://github.com/huggingface/OpenEnv/blob/b9d8c1f953e0c3e0bbee2f3f6f6c73d8eae61f5f/src/openenv/cli/_validation.py),
but keep Reliquary's stronger qualification schema additive and independently
versioned; do not claim that OpenEnv itself certified learning quality or
sandbox safety.

### Independent packages matter

The environment families will eventually require mutually incompatible or
heavy dependencies: SQL, browser automation, terminal sandboxes, scientific
libraries, proprietary APIs, and dataset clients. A single runtime package or
one universal lock would force every consumer to inherit all of them.

Use:

- one root development environment for linting, schema validation, catalog
  generation, and conformance orchestration;
- one `pyproject.toml` and `uv.lock` per environment;
- a fresh virtual environment for each package's CI lane;
- one wheel and one optional OCI image per environment release.

This follows `prime-envs` and avoids the single-version intersection imposed by
a universal uv workspace. The uv documentation explicitly notes that
[workspaces share one lockfile](https://docs.astral.sh/uv/concepts/projects/workspaces/),
which is unsuitable for deliberately heterogeneous packages.

### The current Prime CLI imposes a layout constraint

Use a flat package directory inside each environment project rather than the
otherwise-preferable Python `src/` layout. At the research snapshot, Prime's
environment push preflight discovered a root Python file or an immediate child
package; a nested `src/<module>` package could fail discovery. The behavior is
visible in the current
[Prime CLI environment command](https://github.com/PrimeIntellect-ai/prime/blob/384bda2f9d59fc89cbcff6ab8578faab90a574dd/packages/prime/src/prime_cli/commands/env.py#L1087-L1180).

This is an interoperability accommodation, not a permanent Reliquary ABI. Add
a conformance test so the layout can change later if the upstream CLI improves.

## Product and repository boundaries

| Layer | Owns | Must not own |
| --- | --- | --- |
| `reliquary-environments` | first-party Tasksets, tasks, deterministic world engines, rewards, tools, small fixtures, dataset loaders, manifests, goldens, adapters, conformance tests, example eval/training configs | Bittensor, validator/miner policy, settlement, GPU routing, full trainers, provider credentials, live network activation |
| `reliquary` | signed protocol profiles, allowlists, miner/validator services, admission, replay, assistant-span proofs, GRAIL, payload construction, training backends, checkpoint lineage, Bittensor identity and settlement | editable copies of every future environment, third-party dataset blobs, a generic environment hub |
| Reliquary Hub | federated metadata, immutable artifact references, qualification/lift cards, hosted execution, private listings, access policies | canonical source code for every third-party environment, consensus activation by mere listing |
| Reliquary Network | compute sourcing, scheduling, receipts, provider quality, payment, verified-improvement campaigns | environment source authoring conventions |

The concise product story is:

> The open repository creates environment supply. The Hub qualifies and indexes
> it. The Network buys and settles auditable model improvement.

The repository itself is not the moat. The compounding assets remain the
environment-by-model-by-trainer learning curves, qualification and failure
history, checkpoint lineage, provider reliability, routing data, validator
participation, and private customer workflows.

## Naming

Recommended public names:

| Thing | Name |
| --- | --- |
| Git repository | `reliquadotai/reliquary-environments` |
| Developer product | Reliquary Environments |
| Future registry/control plane | Reliquary Hub |
| Compute and settlement system | Reliquary Network |
| Optional benchmark collection label | Reliquary Worlds |
| Initial implementation branch in the new repo | `bootstrap/environment-suite-v2` |

There is intentionally no `codex` reference in the branch name.

Use namespaced human identifiers and separate Python-normalized names:

```text
Reliquary ID:        reliquary/stateful-tools@0.1.0a1
Distribution:        reliquary-stateful-tools
Import module:       reliquary_stateful_tools
Verifiers ID:        reliquary-stateful-tools
Prime Hub ID:        <owner>/reliquary-stateful-tools@0.1.0a1
Artifact ID:         sha256:<wheel-or-image-digest>
```

The semver name is convenient for humans. The artifact digest is authoritative
for execution and replay. The distinction between the Reliquary logical ID and
the Verifiers/Prime install ID is intentional: the current Verifiers loader
strips the owner and version, replaces hyphens with underscores, and imports
the resulting module. Thus `reliquary-stateful-tools` resolves exactly to
`reliquary_stateful_tools`.

Do not call the repository itself a community Hub until it includes real
third-party authors. Do not market it as an official Prime or OpenEnv product.
The correct wording is `interoperable with`, supported by pinned conformance
evidence.

## Licensing and contribution policy

Recommended default for new environment code: Apache-2.0. It is permissive,
provides an explicit patent grant, and aligns with `prime-envs` and `prime-rl`.
The [Apache-2.0 license](https://www.apache.org/licenses/LICENSE-2.0) does not
grant trademark rights.

Existing Reliquary environment code is MIT-licensed. Copying it into an
Apache-licensed repository does not erase its existing notice or provenance.
Before the first public release:

1. Confirm that Reliquary has the copyright authority required for any desired
   relicensing or dual licensing.
2. Retain the MIT notice for imported files unless counsel confirms a clean
   relicensing path.
3. Put `LICENSE`, `NOTICE`, and `THIRD_PARTY_NOTICES` at the correct levels.
4. Record an SPDX expression separately for code, datasets, test cases, model
   weights, judges, containers, and APIs.
5. Treat missing dataset or benchmark licenses as `not redistributable` until
   permission is documented.

Python project metadata should use current PEP 639-compatible license fields;
see the [PyPA project metadata specification](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/).

Start with DCO 1.1 sign-off and per-environment `CODEOWNERS`, not a copyright
assignment. The [Developer Certificate of Origin](https://developercertificate.org/)
records that contributors have the right to submit their changes. Add a CLA
only if counsel identifies a concrete dual-licensing need.

## Target repository layout

```text
reliquary-environments/
├── README.md
├── LICENSE
├── NOTICE
├── CONTRIBUTING.md
├── GOVERNANCE.md
├── SECURITY.md
├── CODEOWNERS
├── pyproject.toml                 # development tools only
├── uv.lock                        # development tools only
├── compatibility.toml             # exact tested upstream matrix
├── catalog.json                   # generated, never hand-edited
├── schemas/
│   ├── environment-source-v2.schema.json
│   ├── release-lock-v1.schema.json
│   └── qualification-v1.schema.json
├── environments/
│   ├── tool_use/
│   │   └── reliquary_stateful_tools/
│   │       ├── pyproject.toml
│   │       ├── uv.lock
│   │       ├── README.md
│   │       ├── LICENSE
│   │       ├── THIRD_PARTY_NOTICES
│   │       ├── environment.toml
│   │       ├── reliquary_stateful_tools/
│   │       │   ├── __init__.py
│   │       │   ├── taskset.py
│   │       │   ├── engine.py
│   │       │   ├── tasks.py
│   │       │   ├── rewards.py
│   │       │   └── servers/
│   │       │       └── tools.py
│   │       ├── tests/
│   │       └── goldens/
│   ├── retrieval/
│   │   └── reliquary_retrieval_tools/
│   └── swe/
│       └── reliquary_workspace_tools/
├── adapters/
│   └── openenv/
│       ├── reliquary_stateful_tools/
│       │   ├── pyproject.toml
│       │   ├── uv.lock
│       │   ├── openenv.yaml
│       │   ├── server/
│       │   │   └── app.py
│       │   ├── Dockerfile
│       │   └── tests/
│       └── ...
├── configs/
│   ├── eval/
│   └── prime_rl/
├── conformance/
│   ├── reliquary_envkit/
│   └── tests/
├── scripts/
│   ├── new_environment.py
│   ├── validate_changed.py
│   ├── generate_catalog.py
│   ├── build_release.py
│   └── verify_release.py
└── .github/
    └── workflows/
        ├── pr.yml
        ├── nightly.yml
        └── release.yml
```

Do not publish a shared runtime SDK in phase one. Verifiers v1 is the runtime
contract, and duplicated domain utilities should remain small until a stable
common boundary is proven. The root `reliquary_envkit` is development-only at
first. Publish a tiny SDK later only if at least three environments require the
same stable code.

## Canonical environment package contract

Every native environment project must satisfy the following rules.

### Python and Verifiers surface

- The project builds an isolated wheel and sdist.
- Its module name is the normalized taskset name.
- `__all__` exposes exactly one Verifiers v1 `Taskset` subclass.
- It may expose one custom `Env` or `Harness` only when a built-in does not fit.
- Tools use `vf.Toolset` and `@vf.tool` when an agent-facing MCP surface is
  needed.
- Pure reward and world-state logic has no Reliquary, trainer, Bittensor,
  provider, or vLLM imports.
- The first compatibility band should be narrow, such as
  `verifiers>=0.3.1,<0.4`, and updated only through a tested compatibility PR.
- Python 3.12 is the required `prime-rl` integration lane; a broader package
  range may be declared only when clean-venv tests prove it.

### Authored source manifest

`environment.toml` is human-authored and contains behavior and provenance, not
the digest of the artifact that contains it. Required fields include:

```toml
schema = "reliquary/environment-source/v2"
id = "reliquary/stateful-tools"
version = "0.1.0a1"
taskset = "reliquary-stateful-tools"
entrypoint = "reliquary_stateful_tools:StatefulToolsTaskset"
license = "MIT"
verification_tier = "deterministic-replay"

[compatibility]
python = ">=3.12,<3.13"
verifiers = ">=0.3.1,<0.4"

[execution]
network = false
secrets = []
resource_class = "cpu"
max_turns = 8
max_observation_bytes = 65536

[reward]
minimum = 0.0
maximum = 1.0
components = ["state_invariants", "completion"]

[data]
train = "procedural:v2"
eval = "procedural-heldout:v2"
redistributable = true
```

The final schema must also represent owners, task families, stop conditions,
determinism, randomness and seed policy, data revisions, tool side effects,
runtime isolation, timeouts, judge use, expected costs, licenses, and supported
harnesses.

### Generated release lock

`release-lock.json` is generated after artifacts exist. It binds:

- source repository and exact commit;
- source manifest digest;
- wheel and sdist SHA-256 values;
- OCI repository and immutable image digest;
- environment dependency lock digest;
- SBOM digest;
- exact Verifiers, Renderers, MCP, OpenEnv, and trainer test revisions;
- dataset repository, revision, file/content digests, and license expression;
- build workflow identity and provenance attestation;
- schema and golden fixture digests.

Do not put the release lock's own digest inside itself. Sign or hash the
canonical completed document externally.

### Qualification record

`qualification.json` is evidence about one exact release lock. It must not
rewrite source metadata. It records:

- test commit and runner image digest;
- CPU/GPU and operating-system details;
- deterministic replay results;
- reward range, mixed-outcome, and tamper results;
- eval baseline distribution;
- `prime-rl` smoke or sustained-run evidence;
- OpenEnv round-trip evidence where supported;
- sandbox and network-policy evidence;
- known limitations and failure cases;
- operator signature and independent reproduction status.

### Catalog

`catalog.json` is generated from released manifests and locks. It is a discovery
index, never production truth. Reliquary and the Hub may read it to find
artifacts, but a live profile pins the exact lock and artifact digests.

## Dataset policy

Reliquary may reuse Prime-associated or other public datasets only when the
specific dataset license, redistribution rights, and commercial-use terms
permit it. A permissive code repository license does not automatically cover
its data, benchmark tests, model judge, or external service.

Use three data paths:

1. Small deterministic fixtures and procedural generators live in the
   environment package.
2. Large public datasets are referenced by exact repository revision and
   content digest, with an offline preparation step.
3. Private datasets remain outside the public repository and are mounted under
   an explicit runtime access policy.

Where compatible, use a Harbor registry rather than inventing another dataset
downloader. `prime-envs` documents Git-ref- and SHA-addressed datasets in its
[Harbor registry guide](https://github.com/PrimeIntellect-ai/prime-envs/blob/main/HARBOR.md).
Do not release an environment whose training split silently follows a mutable
dataset branch.

## What stays in the current Reliquary repository

The core repository retains all network- and consensus-sensitive behavior:

- `EnvironmentSpec` activation and policy resolution;
- signed `ProtocolProfile` and `EpisodeProfile` limits;
- `EpisodeMetadata`, private replay fields, and submission schemas;
- GRAIL episode binding and assistant-only policy spans;
- miner rollout collection and episode policy;
- validator admission, deterministic replay, batching, quarantine, service,
  and archive logic;
- training payload construction and optimizer backends;
- checkpoint publication and lineage;
- Bittensor coordination, routing, rewards, and settlement.

The current `compat.py` behavior is control-plane compatibility, not portable
world logic. The final implementation should leave that thin bridge in core or
give it an explicitly separate integration package. The existing
`prime_v1.py` dictionary export is not native Verifiers v1 compatibility and
must not be advertised as the `prime-rl` boundary.

## What moves forward into the new repository

Port, rather than destructively move, the following concepts:

- episode task definitions and deterministic generators;
- state machines and domain tool functions;
- parsers and reward/verifier logic;
- stable observation/error serialization;
- goldens and procedural reference actions;
- source manifests and authoring documentation;
- CPU/adversarial qualification logic;
- Verifiers Taskset, Task, Toolset, and optional Env/Harness adapters;
- optional OpenEnv facade and container;
- example eval and `prime-rl` configurations.

The current portable code is almost standard-library-only. One known inward
dependency is the golden loader's call into Reliquary's environment registry;
replace it with package-resource and standalone-catalog lookup.

## Reliquary's external artifact resolver

The core-side integration must remain fail-closed and deterministic.

### Resolution states

Keep three separate concepts:

1. **Installed:** an artifact exists on the node.
2. **Profile-allowed:** its exact release lock and digests are present in the
   signed protocol profile.
3. **Operator-activated:** this process explicitly enables the allowed
   environment.

Installing a Python entry point or Hub listing must never activate a consensus
environment automatically.

### Startup behavior

Before the miner or validator loads the model:

1. Read the signed profile.
2. Resolve each external environment to an exact local wheel installation or
   OCI image digest.
3. Verify the release-lock signature, source manifest, artifact, schema,
   fixtures, and compatibility matrix.
4. Import the declared top-level Taskset and adapter in a clean probe process.
5. Verify that profile limits agree with environment limits.
6. Fail startup on any mismatch.

Do not download code or a container from the network during validator
admission. Deployment prepares artifacts first; runtime verification is
offline.

### Core contract extension

Add a new external-artifact branch to `EnvironmentSpec` rather than replacing
legacy fields immediately. Conceptually it needs:

```text
environment_id
contract_version
release_lock_sha256
source_manifest_sha256
python_distribution
python_version
wheel_sha256
oci_image_digest
taskset_entrypoint
replay_adapter_entrypoint
renderer_id
qualification_sha256
```

The actual signed profile should contain immutable hashes and behavioral
limits. Local cache paths, worker counts, credentials, and host-specific
settings remain unsigned deployment configuration.

Dynamic Python entry points may be useful for developer discovery, following
the [PyPA plugin mechanism](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/),
but production profile resolution must use its explicit locked entry point.

### Episode conversion boundary

Define one explicit, tested conversion between:

- a native Verifiers v1 `Episode`/trace graph;
- the stable Reliquary execution envelope;
- current consensus `EpisodeMetadata` and assistant policy spans.

Do not treat the current `EpisodeTrace.to_wire()` representation as identical
to commit metadata; today they use different field names and are manually
mapped. The adapter must preserve exact sampled tokens, assistant spans,
actions, observations, state/trace digests, stop causes, model and renderer
identity, and task/data revision.

## New v2 fixes that must not be backported silently

The extracted v2 contracts are allowed to correct known replay hazards, but v1
must remain frozen:

- Normalize tool and filesystem failures into stable error codes. Host-specific
  temporary paths and runtime-specific SQLite or operating-system exception
  wording must never enter a consensus observation.
- Canonicalize map and list ordering before rendering or hashing.
- Make seed derivation, generator version, and task index explicit.
- Make the task metadata fields `family`, `generator_version`, and `difficulty`
  contractual rather than implicit.
- Make private `reference_actions` a documented qualification fixture, not an
  accidental interface.
- Pass the signed profile turn/token/observation limits into replay and assert
  equality with environment declarations.
- Give every rejection a stable machine-readable reason code; human messages
  are diagnostic only.

Any behavioral change uses a new taskset/environment version and new goldens.

## End-to-end implementation plan

### Phase 0 — freeze the baseline and synchronize repositories

Before implementation:

1. Fetch and reconcile the current Reliquary branch with its remote. The local
   research checkout is 11 commits behind its tracking branch, and the remote
   added `reliquarylogic_v1` plus overlapping registry/profile changes.
2. Record the exact Reliquary source commit used for extraction.
3. Run the focused Episode tests and archive their results.
4. Freeze the three v1 manifests, schemas, goldens, and implementation files.
5. Write an ADR approving the repository boundary, IDs, licensing treatment,
   and non-circular dependency rule.
6. Reserve the GitHub, PyPI, and GHCR names.

For the planning audit, 38 focused unit tests covering the environment suite,
goldens, qualification, admission replay, miner controller, and Episode
protocol passed locally. This is a healthy extraction baseline, not evidence
for the future standalone artifact.

Exit gate:

- A clean baseline commit and test report exist.
- No current production/default profile changes.
- Legal ownership and repository license treatment are documented.

### Phase 1 — bootstrap the standalone repository

Create `reliquadotai/reliquary-environments` from an empty protected `main`,
then work on `bootstrap/environment-suite-v2`.

Add:

- governance, security, contribution, DCO, and ownership files;
- root dev-only tooling;
- the three JSON schemas;
- `compatibility.toml` with exact upstream revisions;
- environment scaffolder and changed-package detector;
- catalog generator;
- clean-venv installation and schema CI;
- a minimal example Taskset used only to prove packaging.

For provenance, make the initial import identify the exact original Reliquary
commit and file map. Full filtered Git history is optional; a transparent source
commit reference, copyright notice, and diff are mandatory.

Exit gate:

- Root install does not pull Torch, Bittensor, vLLM, or provider SDKs.
- The example builds and installs in a fresh Python 3.12 environment.
- Catalog generation is deterministic.
- No package depends on `reliquary`.

### Phase 2 — port the mirrorable baseline

Port `stateful_tools` first as
`reliquary/stateful-tools@0.1.0a1` /
`reliquary-stateful-tools`.

It is the correct first baseline because it is deterministic, CPU-cheap,
multi-turn, tool-using, has known reference actions and goldens, and exercises
the lifecycle needed by future retrieval, browser, terminal, and API worlds
without requiring a large external dataset or expensive sandbox.

Tasks:

1. Extract pure state/world logic.
2. Normalize v2 errors and serialization.
3. Implement immutable Verifiers `TaskData`, `Task`, and `Taskset` objects.
4. Expose tools through a Verifiers Toolset when the harness needs MCP.
5. Export exactly one Taskset through `__all__`.
6. Add train, public eval, and private qualification splits.
7. Port reference and rejected goldens under the new identity.
8. Add an old-v1-to-new-v2 semantic parity harness for the unchanged task
   subset.
9. Add a capped Verifiers eval configuration.
10. Build the wheel, sdist, SBOM, and unsigned release-lock dry run.

Exit gate:

- Independent install/load/eval passes.
- Fixed-seed output is byte-identical across repeated processes.
- Reference, wrong, malformed, tampered, and collateral-action cases pass.
- Reward lies in its declared range and a sampled batch has mixed outcomes.
- The native wheel contains no Reliquary core dependency.

### Phase 3 — add interoperability facades

Add:

- a pinned, separate OpenEnv facade for stateful tools;
- its `openenv.yaml`, server entrypoint, Dockerfile, reset/step/state tests, and
  optional MCP tool endpoint;
- an unchanged `prime-rl` example config that references the native Taskset;
- a direct Verifiers eval config;
- compatibility tests against the exact upstream matrix.

Do not route native `prime-rl` runs through OpenEnv merely to claim both
standards. Native Verifiers retains richer trace, reward, artifact, and runtime
semantics. OpenEnv proves external transport interoperability independently.

Exit gate:

- The Verifiers wheel works without OpenEnv installed.
- The OpenEnv facade works in its own clean environment and container.
- Direct MCP calls cannot bypass training step/reward accounting in the tested
  path.
- An unmodified pinned `prime-rl` orchestrator completes a smoke rollout.

### Phase 4 — create reproducible releases

Implement per-environment release automation:

1. A tag selects exactly one package/version.
2. CI verifies that tag, manifest, changelog, and package version agree.
3. Build wheel and sdist from a clean checkout.
4. Generate an SPDX SBOM.
5. Build the optional OCI image.
6. Generate and verify `release-lock.json`.
7. Publish Python artifacts through PyPI Trusted Publishing.
8. Push an immutable GHCR image.
9. Generate build provenance and artifact attestations.
10. Publish a signed catalog update and GitHub release.

[PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/) uses OIDC
and short-lived publishing credentials. PyPI can also carry
[digital attestations](https://docs.pypi.org/attestations/). GitHub documents
[artifact attestations for binaries and container images](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations).
These mechanisms prove artifact identity and provenance, not semantic safety;
qualification remains separate.

Do not substitute Prime Hub's source-content hash for the Reliquary release
lock. At the research snapshot, Prime's archive selection did not include every
relevant root artifact, such as the root lock, license, OpenEnv manifest, or
Dockerfile, in its explicit root-file set. Prime's wheel hash remains useful,
but Reliquary must bind its own wheel, sdist, OCI, lock, license, dataset,
fixture, and manifest digests. See the
[current Prime Hub hashing implementation](https://github.com/PrimeIntellect-ai/prime/blob/384bda2f9d59fc89cbcff6ab8578faab90a574dd/packages/prime/src/prime_cli/commands/env.py#L625-L689).

Exit gate:

- A clean machine verifies artifact hashes and attestations.
- A pinned wheel and pinned image produce the same declared environment
  semantics.
- No runtime consumer uses `main`, `latest`, or a floating dependency.

### Phase 5 — add the Reliquary consumer in shadow mode

In the `reliquary` repository:

1. Add the external-artifact `EnvironmentSpec` branch.
2. Add offline lock/artifact verification during startup.
3. Add the native Verifiers episode-to-Reliquary envelope adapter.
4. Add a profile-disabled shadow resolver.
5. Run old v1 and new v2 side by side on matched seeds where their semantics
   intentionally overlap.
6. Preserve current embedded v1 factory/scorer/replay paths unchanged.
7. Add integration tests for miner flattening, admission replay, tamper reject,
   GRAIL signatures, assistant-only spans, detached training payloads, and
   optimizer updates.

Exit gate:

- Installing the new artifact has no activation effect.
- Unknown, missing, mutable, unsigned, or digest-mismatched artifacts fail
  before model load.
- Existing profiles produce unchanged manifests and behavior.
- The new path passes all protocol integration tests in shadow mode.

### Phase 6 — CPU and H200 qualification

CPU matrix:

- source checkout and built wheel;
- Ubuntu 22.04 and 24.04;
- repeated processes and allowed Python patch versions;
- at least 64 fixed tasks per environment;
- reference, wrong, malformed JSON, duplicate key, NaN, over-limit, reordered,
  tampered action, tampered observation, state, and trace cases;
- network-denied and offline startup;
- clean-venv and clean-container install;
- no host-specific path or exception text in committed observations.

H200 matrix:

1. Baseline eval on a pinned small model.
2. Native Verifiers rollout smoke.
3. Unmodified pinned `prime-rl` orchestrator/trainer/inference run.
4. Sustained run long enough to prove more than one optimizer step.
5. Confirm non-zero assistant-token masks and actual weight change.
6. Held-out before/after evaluation with a frozen split.
7. Reliquary miner to validator admission replay.
8. GRAIL span and signature tests.
9. Detached training-payload preservation.
10. Artifact/release-lock and checkpoint lineage capture.
11. Restart/resume and deterministic replay.
12. OpenEnv container round-trip, separately from native training.

The existing H200 evidence proves the embedded versions only. It must remain as
historical evidence and must not be presented as proof of the extracted wheel.

Exit gate:

- The release is train-ready, not merely importable.
- Held-out results and confidence intervals are recorded.
- Every result names exact model, tokenizer, environment, dataset, renderer,
  trainer, container, hardware, and source revisions.
- Known failures are included rather than filtered from the report.

### Phase 7 — signed canary and protocol cutover

1. Publish the qualified stateful-tools release.
2. Create a new opt-in Reliquary protocol profile with the exact external lock
   and artifact digests.
3. Keep current v1 environments in the old profile.
4. Run private/testnet shadow windows.
5. Run a bounded canary with rollback to the old profile.
6. Require validator agreement, archive completeness, no replay divergence,
   and stable training payloads.
7. Activate only through the normal explicit network-upgrade process.

Do not assume protocol version 8 is available: the tracking branch already
uses it for another development profile. Select the next coordinated version
after synchronizing the core repository.

Exit gate:

- The network profile, environment lock, and checkpoint lineage are mutually
  bound.
- At least two independent nodes reproduce qualification.
- Rollback has been rehearsed.
- No default profile changes accidentally.

### Phase 8 — port the remaining first-party suite

Port in this order:

1. **Retrieval tools v2** — adds evidence selection, citation discipline, and
   long-context tool decisions while staying CPU-cheap and deterministic.
2. **Workspace tools v2** — adds planning, file edits, state persistence, and
   sandbox policy; it is higher-value but has a larger security/replay surface.
3. **Terminal/SWE environment** — use an established container/task format and
   immutable images; target real code-repair capability.
4. **Browser research environment** — target source gathering and grounded
   synthesis with recorded page snapshots, not the live mutable web in a
   consensus replay.
5. **Multi-agent coordination environment** — only after the trace envelope
   explicitly supports multiple agents and role-specific policy spans.

Each environment is its own release, qualification record, and activation
decision. Do not wait for all five before shipping the first qualified package.

### Phase 9 — federated Hub

After the first-party contract is stable:

- accept signed external Git, wheel, OCI, Hugging Face, Verifiers, and OpenEnv
  references without copying their source when licenses or author preferences
  do not permit it;
- run the public conformance kit and publish qualification cards;
- separate `listed`, `qualified`, and `network-eligible` status;
- support private environment metadata and access policies;
- retain every artifact referenced by an improvement receipt;
- attach model and trainer learning evidence to exact artifact digests;
- allow environment authors to share paid execution revenue based on
  attributable use and verified learning evidence, not stars or downloads.

Exit gate:

- At least one environment from an independent author is indexed and qualified.
- Hub listing cannot silently change network eligibility.
- Historical versions remain resolvable.

## CI and release gates

### Pull request: fast and mandatory

For every changed environment:

- validate manifest and license metadata;
- lint, type-check, and run unit tests;
- build wheel and sdist;
- install into a brand-new virtual environment;
- assert exactly one Taskset export;
- run a capped deterministic eval;
- compare fixed-seed goldens;
- deny undeclared network and secrets;
- inspect dependencies for forbidden Reliquary/trainer/provider imports;
- scan committed fixtures for secrets and unexpected large files;
- verify catalog regeneration produces no unrelated diff.

### Nightly or scheduled integration

- run larger eval samples and reward-distribution checks;
- run cross-process and cross-OS replay;
- run the exact supported Verifiers/Renderers/prime-rl matrix;
- run OpenEnv facade/container round trips;
- scan OCI images and dependency licenses;
- exercise malicious tool arguments and sandbox escapes;
- verify data URLs, exact revisions, and license metadata;
- compare current upstream heads in a non-blocking experimental lane;
- alert on compatibility drift without silently widening version ranges.

### Release

- two-person review for tasks, rewards, manifests, goldens, and release workflow;
- tag/version/changelog agreement;
- clean and reproducible build;
- immutable wheel, sdist, OCI, SBOM, provenance, and release lock;
- qualification record bound to that lock;
- signed catalog update;
- retention policy preventing deletion while referenced by any receipt.

## Compatibility policy

`compatibility.toml` records both supported ranges and exact tested revisions.
For example:

```toml
snapshot_date = "2026-09-03"

[verifiers]
supported = ">=0.3.1,<0.4"
tested_commit = "<exact commit>"

[openenv]
supported = ">=0.4.1,<0.5"
tested_commit = "<exact commit>"

[prime_rl]
tested_commit = "<exact commit>"
recursive_submodules = true
```

Before implementation, refresh those exact values from the synchronized
upstream repositories. `prime-rl` pins Verifiers, Renderers, and environment
submodules; testing only a loose `prime-rl` version is insufficient.

Rules:

- Release dependencies never point to `main`.
- Direct Git dependencies use exact commits.
- Automated dependency PRs must run the conformance matrix.
- A reward, task distribution, dataset, rendering, or world-semantic change
  requires a new environment version and new evidence.
- Patch releases fix packaging or behavior only when the exact task/reward
  contract remains unchanged.
- Unsupported upstream heads may be tested experimentally but do not enter the
  production compatibility claim.

## Governance and trust ladder

Begin with accountable company-led maintenance:

- a Reliquary Maintainer Council controls releases;
- per-environment maintainers and `CODEOWNERS` own their packages;
- public RFCs govern schemas, ABI, trust badges, and network eligibility;
- two approvals are required for reward, verifier, manifest, or eligibility
  changes;
- qualification operators disclose conflicts with environment authors;
- emergency revocation may stop new jobs without deleting historical artifacts;
- later add independent authors, validators, researchers, and users to a
  technical steering group.

Use distinct statuses:

1. **Listed** — metadata accepted; unreviewed.
2. **Conformant** — interface and schema tests pass.
3. **Replay-qualified** — deterministic replay and goldens pass.
4. **Sandbox-qualified** — declared isolation and security tests pass.
5. **Learning-evaluated** — measured training signal exists.
6. **Network-eligible** — admitted by explicit Reliquary protocol governance.
7. **Independently reproduced** — a second operator replicated the result.

A listing or signature never proves an environment is safe, useful, unbiased,
or capable of improving a model.

## Marketing claim gates

| Claim | Required evidence |
| --- | --- |
| Open-source environments | Public source and explicit OSI-approved license |
| Community Hub | Maintained packages from independent authors; before that, say first-party catalog |
| Verifiers-v1-native | Native Taskset/Task/trace/reward conformance at a pinned revision |
| OpenEnv-compatible | Pinned facade/container reset-step-state and transport tests |
| `prime-rl` interoperable | An unchanged pinned `prime-rl` backend completes the documented run |
| Train-ready | Sustained rollout and optimizer run, not one import or one step |
| Reproducible | Independent rerun within predeclared tolerances |
| Verified improvement | Frozen held-out before/after evaluation, signed lineage, uncertainty, and negative results |
| Cryptographically verifiable | Restrict wording to identity, integrity, and provenance; not semantic truth |
| Decentralized | Multiple independent providers and validators materially participate |
| Community-governed | External participants have documented decision rights |

Do not announce “the same system as Prime.” Announce standards-native
interoperability and the different product layer Reliquary adds: validator-grade
improvement evidence and compute-market settlement.

## Security constraints

- Environment code is untrusted until qualified.
- A public package install is not permission to access host files, network,
  Docker, cloud metadata, or credentials.
- Network, secrets, mounts, writable paths, subprocesses, and resource caps are
  declared and enforced per environment.
- Validator replay uses preinstalled immutable artifacts and a deny-by-default
  runtime.
- Private tests and anti-fraud cases remain outside public packages, but their
  use is committed before scoring where required for fairness.
- Every executed artifact has an SBOM and provenance record.
- Security revocation blocks new execution by digest; it never rewrites old
  receipts.
- Environments that execute arbitrary model-written code require a separate
  sandbox threat review and cannot inherit the CPU taskset trust tier.

## Work split across repositories

### New repository pull requests

1. `ENV-001`: governance, schemas, root tooling, compatibility matrix.
2. `ENV-002`: package scaffolder and isolated-install CI.
3. `ENV-003`: stateful-tools v2 native Verifiers package.
4. `ENV-004`: determinism, adversarial, and parity conformance kit.
5. `ENV-005`: OpenEnv facade and container.
6. `ENV-006`: eval and unchanged `prime-rl` example configs.
7. `ENV-007`: release lock, SBOM, OIDC publish, attestations.
8. `ENV-008`: alpha artifact and qualification report.
9. `ENV-009`: retrieval-tools v2.
10. `ENV-010`: workspace-tools v2.

### Reliquary core pull requests

1. `REL-001`: external release-lock schema reader and offline verifier.
2. `REL-002`: explicit external artifact branch in the environment registry.
3. `REL-003`: native Verifiers episode/envelope conversion boundary.
4. `REL-004`: shadow-mode stateful-tools v2 integration.
5. `REL-005`: cross-repository replay and GRAIL conformance.
6. `REL-006`: opt-in signed canary profile.
7. `REL-007`: qualification and activation evidence.

Dependencies:

```text
ENV-001 -> ENV-002 -> ENV-003 -> ENV-004 -> ENV-008
                         |          |
                         v          v
                       ENV-005    ENV-006
                                      |
ENV-007 ------------------------------+

REL-001 -> REL-002 -> REL-003 -> REL-004 -> REL-005 -> REL-006 -> REL-007
                         ^          ^
                         |          |
                      ENV-003     ENV-008
```

No production cutover PR should mix environment source changes with protocol
activation.

## Definition of complete

The separation is complete only when all of the following are true:

- the new public repository contains three independently installable native
  environments;
- each has a manifest, independent lock, wheel, OCI image where needed, SBOM,
  attestations, release lock, goldens, and qualification record;
- native packages do not import Reliquary core;
- stateful, retrieval, and workspace environments install in clean isolated
  environments;
- pinned Verifiers evals pass;
- an unchanged pinned `prime-rl` run trains on at least the baseline package;
- OpenEnv facades pass their separate conformance lane;
- Reliquary verifies and consumes immutable external artifacts offline;
- embedded v1 environments and historical evidence remain reproducible;
- a new signed profile completes canary, rollback, and independent
  reproduction;
- the Hub distinguishes listing, qualification, learning evidence, and network
  eligibility;
- public claims match the evidence gates above.

## Immediate next actions

The next implementation session should do only Phase 0 and Phase 1:

1. Synchronize the current Reliquary branch and rerun the baseline audit.
2. Approve the ADR and licensing choice.
3. Create the new GitHub repository and protected branches.
4. Bootstrap schemas, root tooling, and isolated-package CI.
5. Create the empty stateful-tools package scaffold.
6. Stop before changing the active Reliquary registry or protocol profile.

After that foundation passes, implement the v2 contract in the
`reliquary-stateful-tools` package as the first real artifact. This keeps the
first code change small, reviewable, and directly reusable as the template for
every later environment.

## Primary sources

- [Prime Environments](https://github.com/PrimeIntellect-ai/prime-envs)
- [Prime community environments](https://github.com/PrimeIntellect-ai/community-environments)
- [Verifiers v1 Tasksets](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/tasksets.md)
- [Verifiers v1 architecture](https://github.com/PrimeIntellect-ai/verifiers/blob/main/docs/v1/architecture.md)
- [`prime-rl` overview](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/overview.md)
- [`prime-rl` configuration](https://github.com/PrimeIntellect-ai/prime-rl/blob/main/docs/configuration.md)
- [OpenEnv repository and experimental-status notice](https://github.com/huggingface/OpenEnv)
- [OpenEnv environment builder](https://github.com/huggingface/OpenEnv/blob/main/docs/source/getting_started/environment-builder.md)
- [OpenEnv MCP environment guide](https://github.com/huggingface/OpenEnv/blob/main/docs/source/tutorials/mcp-environment.md)
- [`prime-envs` Harbor registry](https://github.com/PrimeIntellect-ai/prime-envs/blob/main/HARBOR.md)
- [Python project metadata](https://packaging.python.org/en/latest/specifications/declaring-project-metadata/)
- [Python plugin discovery](https://packaging.python.org/en/latest/guides/creating-and-discovering-plugins/)
- [PyPI Trusted Publishing](https://docs.pypi.org/trusted-publishers/)
- [PyPI digital attestations](https://docs.pypi.org/attestations/)
- [GitHub artifact attestations](https://docs.github.com/en/actions/how-tos/secure-your-work/use-artifact-attestations/use-artifact-attestations)
- [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0)
- [Developer Certificate of Origin 1.1](https://developercertificate.org/)
