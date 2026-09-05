# Reliquary environment authoring

This guide covers the small, single-turn environment interface used by
`reliquaryverifiable_v1`. It is intentionally narrower than a multi-turn tool
world: a problem enters as one prompt, a miner returns one completion, and the
validator recomputes a bounded reward.

## The four layers

An environment is usable only when all four layers agree:

1. The environment class generates problems and computes a reference reward.
2. The immutable registry declares its runtime policies.
3. A versioned protocol profile pins its prompt, answer, cap, target, and
   environment-manifest digest.
4. The operator explicitly selects a non-empty subset of environments declared
   by that profile.

Registration alone never activates an environment. A profile alone also does
not activate it. This two-key rule keeps new task code dormant until miners and
validators deliberately select the same contract.

## Smallest implementation

Implement the existing `Environment` protocol:

```python
class ExampleEnvironment:
    name = "example_v1"
    validator_authoritative_reward = True

    def __len__(self) -> int:
        return 1 << 25

    def get_problem(self, index: int) -> dict:
        return {
            "id": "stable-id",
            "prompt": "Complete this deterministic task...",
            "ground_truth": "validator-only canonical specification",
            "task_family": "example_family_v1",
            "generator_version": "example-generator-v1",
            "generator_index": index,
        }

    def compute_reward(self, problem: dict, completion: str) -> float:
        # Never trust the miner's claimed reward. Fail closed on malformed data.
        try:
            return float(check(problem, completion))
        except Exception:
            return 0.0

    def source_health(self) -> dict:
        return {"status": "ok", "external_dependencies": []}
```

Keep the class boring. Dataset access, sandbox execution, parsing, and
materialization should be small helpers with unit tests, not hidden constructor
side effects.

## Exact tweak points

For a new task family inside a future suite version:

- Put deterministic task generation and pure reference operations beside
  [`records_tasks.py`](../reliquary/environment/records_tasks.py).
- Put bounded answer parsing beside
  [`structured_output.py`](../reliquary/environment/structured_output.py).
- Add the class and top-level batch scorer beside
  [`reliquaryverifiable.py`](../reliquary/environment/reliquaryverifiable.py).
- Add one static `EnvironmentSpec` in
  [`registry.py`](../reliquary/environment/registry.py).
- Commit a canonical provenance manifest under
  `reliquary/environment/manifests/` and immutable JSONL goldens under
  `tests/fixtures/`.
- Add a new `ProtocolProfile` only after the environment is deterministic and
  reward-audited while dormant.

Do not edit `reliquaryverifiable_v1` generation, parsing, operation weights, or
expected answers in place. Those values determine index-to-prompt and
completion-to-reward consensus. Create `reliquaryverifiable_v2`, a new manifest,
new goldens, and a new profile instead.

## Registry policies

Each `EnvironmentSpec` declares:

- a lazy factory and top-level picklable scorer;
- whether the validator overwrites miner rewards;
- CPU or sandbox admission resources;
- EOS/cap or Math-BFT termination;
- boxed, fenced-Python, or JSON final-answer policy;
- the attainable reward lattice and its policy version;
- the versioned environment contract and optional manifest digest;
- an optional trusted reward-materializer method.

Callable paths, worker counts, local timeouts, and queue choices are runtime
settings and are excluded from the consensus manifest. Generator/checker IDs,
reward policy, termination policy, answer policy, and the authored manifest
digest are consensus-facing.

## Determinism rules

- Seed from a cryptographic digest of environment ID, generator version, and
  integer index. Never use Python `hash()`.
- Use rejection sampling for bounded integers if byte-for-byte portability is
  required.
- Avoid floats, locale-dependent parsing, dates, unordered-set iteration, and
  network data in a procedural contract.
- Bound completion bytes, parsed item count, nesting, integer magnitude, and
  sandbox execution.
- Reject duplicate JSON keys and non-finite values.
- `compute_reward` must return a finite value in the declared lattice and must
  not raise for arbitrary completion text.
- Record task family, generator version, operation ID, and difficulty so the
  archive can detect curriculum collapse.

## Local development loop

Use the development profile and explicit environment selection together:

```bash
export RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-reliquary-verifiable-v6-dev1
export RELIQUARY_ENVIRONMENTS=reliquaryverifiable_v1

python scripts/qualify_environment_suite.py \
  --samples 10000 \
  --output tmp/reliquaryverifiable-v1-qualification.json
```

During uncommitted development, add `--allow-dirty`; the report records that
override. Never use a dirty override as release evidence.

The fast contract gate is:

```bash
pytest -q \
  tests/unit/test_environment_registry.py \
  tests/unit/test_reliquaryverifiable_environment.py \
  tests/unit/test_environment_qualification.py \
  tests/unit/test_environment_evidence.py \
  tests/integration/test_third_environment_smoke.py
```

## Before activation

Offline success is necessary but insufficient. Complete the private shadow,
proof-capacity, queue-latency, detached-training, retention, evaluation, and
rollback gates in the canary runbook. A trained checkpoint stays a candidate
until an operator explicitly promotes it.
