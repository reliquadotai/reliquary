# Authenticated async trainer opt-in

`reliquary train-worker --attested-job /absolute/path/job.json` replaces the
current process with the separately installed authenticated rollout consumer.
Supervisor signals and the consumer's exit status pass through directly.
Without this option, the existing detached trainer is unchanged.
`--shadow` and `--attested-job` cannot be combined.

This is a separate rollout queue and checkpoint publication path. It does not
import legacy schema-4 payloads, activate validator routing, change miner
rewards, or publish checkpoints to the legacy R2 manifest. Its rollout
controller accepts complete signed GPU generation receipts and
operator-verified reward bindings; the consumer revalidates their signatures,
GPU certificate/workload binding, current trust policy, group completeness,
policy lag and assignment age before an optimizer update.

## Install the isolated runtime

The Git requirement pins the runtime source, not its transitive dependencies.
Export the same revision's `uv.lock` and install its hashed dependency set in
a fresh Python 3.12 environment, then install the Git package without resolving
dependencies again. The `gpu` extra includes the CUDA dependencies on Linux;
the optional sandbox package is not selected.

```sh
TEE_RUNTIME_REVISION=2131490288189896127521c57af31f41fa63f432
git clone --no-checkout https://github.com/reliquadotai/reliquary-TEE.git \
  /opt/reliquary-attested-source
git -C /opt/reliquary-attested-source checkout --detach "$TEE_RUNTIME_REVISION"
uv venv --python 3.12 /opt/reliquary-attested
uv export --project /opt/reliquary-attested-source \
  --frozen --extra gpu --no-dev --no-emit-project \
  --output-file /opt/reliquary-attested/runtime-dependencies.txt > /dev/null
uv pip install --python /opt/reliquary-attested/bin/python \
  --require-hashes --only-binary :all: \
  -r /opt/reliquary-attested/runtime-dependencies.txt
uv pip install --python /opt/reliquary-attested/bin/python \
  --no-deps -r requirements/attested-trainer.txt
uv pip check --python /opt/reliquary-attested/bin/python
```

Run these commands from the core checkout. Retain the generated dependency
export with the deployment; no second lockfile is maintained in this repository.
The launcher requires the installed distribution's version and PEP 610 Git
provenance to match the code's exact pin. Editable checkouts and installations
without that provenance are refused. This checks installation identity; it
does not attest the trainer or protect a local environment writable by another
administrator. Use an operator-controlled environment and state directory.

Prepare the authority, worker certificate, TLS pins, immutable model assets,
signed assignments, verified grades and local queue using the pinned runtime's
[async RL runbook](https://github.com/reliquadotai/reliquary-TEE/blob/2131490288189896127521c57af31f41fa63f432/docs/async-rl.md).
Review the runtime's [release notes](https://github.com/reliquadotai/reliquary-TEE/releases)
for its qualification status. Installing or selecting this mode does not
activate a worker fleet or a core protocol profile.
The controller and this trainer use the same SQLite queue on a local
filesystem. A remote database or legacy R2 queue is not supported by this path.

## Create the operator job

The JSON below uses illustrative absolute paths. The contents of `trainer.json`,
`assets.json`, `current-policy.json` and `trust-policy.json` use the runtime's
schemas. The trust policy is the full validator `TrustPolicy`, including
approved profiles and current revocations; it is not `GuestTrustPolicy`.

```json
{
  "version": "reliquary/attested-training-job/v1",
  "runtime_python": "/opt/reliquary-attested/bin/python",
  "config": "/srv/attested-training/trainer.json",
  "model_dir": "/srv/models/initial",
  "assets": "/srv/attested-training/assets.json",
  "current_policy": "/srv/attested-training/current-policy.json",
  "trust_policy": "/srv/attested-training/trust-policy.json",
  "queue": "/srv/attested-training/queue.sqlite",
  "checkpoints": "/srv/attested-training/checkpoints",
  "device": "cuda",
  "max_steps": 1,
  "idle_seconds": 60,
  "lease_seconds": 300
}
```

The job must be a regular file owned by the launching user, without group or
other write access. All paths must be absolute. Unknown fields, duplicate JSON
keys and out-of-range step, wait or lease limits are refused. This is trusted
operator configuration, not an endpoint for miner-supplied jobs.

```sh
chmod 600 /srv/attested-training/job.json
reliquary train-worker --attested-job /srv/attested-training/job.json
```

CUDA is the default. `cpu-development` is an explicit local testing option;
CPU execution does not turn a synthetic fixture into a hardware-qualified
deployment. The optional runtime owns strict safetensors loading, current
policy publication, durable queue fencing and explicit checkpoint recovery.
If a process stops after reserving an optimizer step, inspect and reconcile
the durable checkpoint using that runtime's `queue-status` and
`queue-recover` commands. Never reset unresolved work for automatic retry.

## Contract and qualification boundary

Actual sampler probabilities `q`, teacher-forced reference probabilities `r`,
and the current trainer policy `p` remain distinct. The consumer accepts
authenticated `behavior_q` generation receipts; a reference forward or a
CPU-only sandbox certificate cannot substitute for them. The current
gradient-bearing `p` forward is used for the update after validation. Complete
GRPO groups, bounded lag and the full-support sampling contract are required.

New checkpoint assets and `current-policy.json` are published locally. Worker
workload approval and worker restart for that checkpoint remain explicit
operator actions. Existing core task routing, payments and publication require
their own rollout and review; selecting this launcher does not change them.
CPU integration tests verify dispatch and refusal boundaries. They do not
qualify confidential hardware, CUDA performance, throughput or training
convergence.
