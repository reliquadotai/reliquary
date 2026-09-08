# Standalone environment: pinned CPU qualification

`reliquary_stateful_tools_v2` is registered for explicit loading only. It belongs
to no signed profile. Installing it does not enable training or network use.
New environment source belongs in
[reliquary-environments](https://github.com/reliquadotai/reliquary-environments),
not in Reliquary core; the embedded v1 suite remains a historical implementation.

## Logic candidate and native Verifiers interoperability

`reliquary_logic_v2` is a single-turn CPU environment with the
`reliquary/answer-json/v1` ABI. Its signed profile eligibility is separate from
installing the wheel and from enabling fill-closed operation. The existing
Stateful Tools wheel remains an optional conformance fixture outside that
profile; it does not add a fourth training environment.

The candidate pin is committed in `docker/logic-release.json`:

- External source: `f95be97372a9b114376657f075fd0d00b88eb563`.
- Planned release tag: `logic-v0.1.0a1` (the source/candidate build is usable
  before publication; the release URL is usable only after that tag is published).
- Wheel: `reliquary_logic-0.1.0a1-py3-none-any.whl`.
- Wheel SHA-256: `d12e4258fa9b190f29a33a055cee11ae632d813aed5f51b28a20a3727212b527`.
- Artifact SHA-256: `1e4e05cae799d8e71d8876b0f7526c5b09ca1d5a9ab05f364fb35539288c5019`.
- Factory: `reliquary_logic:LogicEnvironment`; native Taskset:
  `reliquary_logic:LogicTaskset`.

From a clean checkout of the exact external source, build the candidate with
`uv run --no-project --python 3.12 python scripts/build_logic_release.py --output /absolute/empty/output`.
The builder verifies two byte-identical builds against the reviewed hashes.
It emits the wheel, `release.json` and relative `SHA256SUMS`. Source changes
require a reviewed release pin; never substitute a wheel under an existing pin.

In a dedicated Python 3.12 virtual environment containing pip, install from
that candidate and run conformance from the core checkout:

```sh
python scripts/install_logic_runtime.py --wheel /absolute/output/reliquary_logic-0.1.0a1-py3-none-any.whl
# Obtain the already published Stateful Tools fixture with the downloader below,
# then install its verified wheel with --no-deps in this same isolated environment.
python scripts/qualify_verifiers_interop.py \
  --logic-wheel /absolute/output/reliquary_logic-0.1.0a1-py3-none-any.whl \
  --stateful-wheel /absolute/output/reliquary_stateful_tools-0.1.0a1-py3-none-any.whl
```

The installer checks the wheel hash before invoking pip, installs the locked
optional dependency set and exact Verifiers source commit, checks installed
dependency consistency, and imports the digest-bound artifact. The qualifier
uses real `vf.load_taskset`, `Task.score`, `Trace` and `WireTrace` objects.
It covers all 12 Logic and 3 Stateful Tools families across train, eval and
qualification, exact native/ABI prompt identity, packaged goldens, malformed
answers, and authoritative reward recomputation after a wire roundtrip. Episode
actions are recovered from the native trace and replayed in Reliquary. Trace
reward fields and tool observations are never accepted as authoritative state.

These are two reviewed first-party Tasksets, not universal Prime catalog
support. Other environments need an explicit deterministic ABI, artifact pin,
resource policy and conformance tests. The legacy `export_prime_v1_*` helpers
remain JSON exports; `native_prime_v1_trace` is the optional actual API bridge.

Build the CPU import image using the same installer as the runtime image:

```sh
docker build -f docker/Dockerfile.environment-qualification \
  --secret id=logic_wheel,src=/absolute/output/reliquary_logic-0.1.0a1-py3-none-any.whl \
  -t reliquary-logic-qualification:reviewed .
docker run --rm --network none reliquary-logic-qualification:reviewed
```

The common CUDA image includes Logic only with `RELIQUARY_INSTALL_LOGIC=1`.
Pass the same BuildKit secret to build from a reviewed candidate; without it,
the installer downloads the pinned published release. CPU executor and signer
images do not install this stack. The image workflow's explicit `include_logic`
input builds a separate `sha-<revision>-logic` tag and cannot promote `latest`.

## Exact release

- Release: [v0.1.0a1](https://github.com/reliquadotai/reliquary-environments/releases/tag/v0.1.0a1).
- Release source: `cdb998b355ee41355823f2a15cdd21ab19c4718a`.
- Annotated tag object: `d4ef87ef377874e21d0256a3be4f48df451ecc5b`.
- External integration PR #1 merge: `73285b192f0359ed75635d6d4ff1694ab3a1b106`.
- Wheel SHA-256: `f4d5480e57e66265faa78c53e36fa8ab781afe0ae907d7dc5749d2b0f9344155`.
- Artifact manifest SHA-256: `9779a3151575687e823567c6c9c5459d93794bab4f27f4bcde987c8414cb50f6`.
- Source manifest SHA-256: `6487e60f82a2f5404fa89eb3acb2366889352a949c66f0c0a977609eb2c60c0e`.
- Prime-RL v0.9.0: `ab5de8fff44b2c4a5c85e24b6e6e3f7d57eee7b1`.
- Verifiers: `b2e4e8157783b2c0dffc7821044c87f29f1c3ccf`.
- Renderers: `cb8243913702367878427c7a7094b350ea1a8e20`.
- Pydantic-config: `65b15dffba82d4be19efdaf8b2b9705cc1756be8`.
- Prime-envs: `26dafdc9582576975ec576f893be7319028daf51`.
- Qwen3-4B-Instruct-2507: `cdbee75f17c01a7cc42f958dc650907174af0554`.

The merge commit records the integration configuration; it is distinct from the
older release source that produced this immutable wheel. The wheel metadata's
Verifiers range alone does not establish the qualified revision.

## Reproduce on CPU

Use a dedicated Python 3.12 environment so optional Verifiers dependencies do
not alter the core CPU test environment. From the repository root:

```sh
uv venv --python 3.12 .venv-external
uv pip install --python .venv-external/bin/python -e '.[dev]' \
  'verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@b2e4e8157783b2c0dffc7821044c87f29f1c3ccf'
mkdir -p /tmp/reliquary-external-artifact
.venv-external/bin/python -c 'from pathlib import Path; from scripts.qualify_external_environment import download_pinned_wheel; print(download_pinned_wheel(Path("/tmp/reliquary-external-artifact")))'
uv pip install --python .venv-external/bin/python --no-deps \
  /tmp/reliquary-external-artifact/reliquary_stateful_tools-0.1.0a1-py3-none-any.whl
.venv-external/bin/python scripts/qualify_external_environment.py
```

The downloader checks the complete wheel before exposing an installable file;
it refuses a wrong hash, an oversized response, or an existing destination.
The qualification command downloads and hashes it again **before** importing
external code. It checks the installed artifact, executes three reference
replays, compares exact task IDs and state hashes with published goldens, and
checks incorrect answers score zero. These scripted policies prove CPU loading
and replay, not learned-model performance or GPU qualification.

The loader verifies every declared installed file on every create. Package
imports execute a snapshot of digest-bound Python source, ignoring shadow
packages and bytecode caches; previously imported unverified module objects
are rejected. The pinned package eagerly imports its only implementation
module. A release using lazy submodules or native extensions needs a reviewed
loader extension before acceptance. Dependencies and an already compromised
Python interpreter are outside this source-integrity boundary. Generic
`environment.abi.EnvironmentRegistry` remains the inactive V1 interchange
contract; `environment.registry` resolves concrete runtime adapters.

## Remaining hardware qualification

Keep the final PR draft and all new capabilities disabled. Use an isolated
supported two-GPU CUDA host (one training GPU, one inference GPU; H100/H200
capacity is suitable for this qualification setup) for the pinned Prime-RL
smoke, reward variance, more than one optimizer step, checkpoint publication
and resume, frozen held-out evaluation, and the full miner/admission/GRAIL/
trainer path. The old Nebius and Verda hosts must not be contacted.

Existing embedded-v1 H200 evidence remains historical and is not evidence for
these wheels or the final three-environment fill-closed profile. Activation
also requires the exact runtime/checkpoint proof-capacity manifest, a durable
drain boundary, checkpoint publication and resume, recovery or abort of the
active window, and economic accounting evidence. Native Prime conformance and
a Prime-RL config dry-run do not replace those Reliquary DAPO/GRAIL gates.
