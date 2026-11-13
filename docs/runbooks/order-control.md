# Order control: build and run

The GPU-less control serving every order job, evaluations (`${RELIQUARY_ADMIN_TASK_PREFIX}eval-`) and
generation orders on any model (`${RELIQUARY_ADMIN_TASK_PREFIX}gen-`). Design:
`docs/design/2026-10-01-any-model-datasets-design.md`; rulings: the "Any-model datasets" section of
`docs/design/2026-10-01-evaluation-orders-rulings.md`.

## Image

`docker/Dockerfile.order-control` layers on the validator image, as the production corpus image does: the
reviewed wheels of `reliquary-logic`, `reliquary-dapo-math`, `reliquary-instruction-following` and
`reliquary-code` (from reliquary-environments, at the pins the catalog digests name) are installed from a
build context named `envs`. The build then runs `reliquary corpus order-control-check`, which fails unless:

- `bittensor_drand` imports (it comes with `bittensor`; without it every record of a sampled job is audited,
  which multiplies executor load);
- `huggingface_hub` and `transformers` import (model files and tokenizers are read on CPU);
- each order source's package is installed, its artifact verifies, and its digest is the catalog's.

For signed episode orders (SWE, TMax) the image also carries reliquary-sandbox with its verifiers bridge at
`SANDBOX_COMMIT`, `reliquary-swe` and `reliquary-terminal` at the reliquary-environments commit the agentic eval
sets pin (`ENVS`, checked against each set's `env_package`) and `renderers`; reliquary-harness comes with the
base. The Verifiers tasksets single-turn sets are graded with go to `/opt/evalsets-site`, at the end of
`sys.path`. A second build check fails unless all of these load at their pins.

## Run

```bash
docker run -d --name order-control --network host \
  --env-file <subnet bucket credentials> -e RELIQUARY_ADMIN_TASK_PREFIX=order- \
  <order-control image> --netuid 81
```

Signed episode orders also need the order control's own sandbox token key (an Ed25519 PEM, mode 0600,
`python -m reliquary_sandbox.attest.signing generate PATH`; never the corpus validator's key): mount it read-only
and set `RELIQUARY_ORDER_SANDBOX_KEY_FILE`, `RELIQUARY_ORDER_SANDBOX_KEY_ID`,
`RELIQUARY_ORDER_SANDBOX_HOTKEY` (the ss58 address miners sign session requests for), optionally
`RELIQUARY_ORDER_SANDBOX_RETIRED_KEYS`, and `RELIQUARY_SANDBOX_MACHINE_IDS`. The key settings have their own
names so the corpus validator's (`RELIQUARY_SANDBOX_VALIDATOR_*`) are never read here; the sessions are stored
under `reliquary/sandbox/order-sessions/`, which no other validator restores. Each gateway lists the key's
public half in its validator keys. Without these settings, or when the sandbox services fail to start,
episode orders are refused (the reason is in the control's status) and every other order is served.
An order job's sandbox sessions are `/corpus/jobs/{job_id}/sandbox/sessions`, inside the order jobs' location.

Rented gateways. The platform rents sandbox gateways while agentic orders run and registers each one in the
machine directory through the admin service, under `RELIQUARY_ADMIN_SANDBOX_MACHINE_PREFIX` (for example
`rlq-gw-prod-`): the admin's `/admin/v1/sandbox-machines` routes register, drain or revoke, and read only the
machines named with it. Set the same prefix in the order control's `RELIQUARY_SANDBOX_MACHINE_PREFIXES`
(`["rlq-gw-prod-"]`, alongside or instead of `RELIQUARY_SANDBOX_MACHINE_IDS`): a gateway registered after the
start is polled and placed on from the next directory read (30 s), and one set `revoked` is left out, with no
restart. The control records each accepted heartbeat in the directory (`last_heartbeat`), which is how the
platform sees a gateway ready.

An episode order is kept from miners until a gateway can take its sessions: a fresh, active machine serving
its sandbox env at its exact package and options digest (its first task's), holding some of its images, with
slots. Until then `/corpus/jobs` lists it `starting` (not in `jobs`), its writes answer 409 `job_paused`, and
the control status shows `jobs[id]["held"] = "no_gateway"` (`"awaiting_audit"` while it waits for its audit
executor instead). It opens on the first directory read and heartbeat poll that see such a gateway. Once open it
stays open: if the fleet later drops to zero, session opens are refused retryably (`quota_live`, 429 with
`Retry-After`) rather than the job vanishing from miners.

The control's status (`GET /admin/v1/eval-control/status`, every 30 s) gives each episode order a `sandbox`
object the platform sizes the gateways from (design `docs/design/2026-11-12-sandbox-capacity-autoscale.md`):
`episodes_total`/`_finished`/`_left`, `sessions_held`, `episode_seconds` (median of the last 50 finished),
`tokens_per_episode`, `audit_episodes_per_second` and `quota_refusals_fleet_full` (last 10 min). A null is
not measured yet.

It needs network to drand and to the Hugging Face hub. Route it with the locations
`reliquary corpus order-nginx --port 8791` prints for the same `RELIQUARY_ADMIN_TASK_PREFIX`; miners need
no setting. The admin service and the order control must run with the same prefix.

## Code-execution grading (LiveCodeBench)

A Verifiers set whose reward runs code (`needs_runtime: true` on its card) is graded on the admin. For
LiveCodeBench the taskset's own `verify.py` (pinned by sha256 in `reliquary/eval/sandboxed_runtime.py`) runs in
the admin's scoring child, and each test program goes to the gVisor grading service's stdio mode: no Docker
socket in the admin. The service runs as a sidecar from the same image (it carries `runsc` and the grader bundle).
It holds no secret; like every local runsc grader it needs `--privileged`, and the hostile-code boundary is gVisor.

```bash
sudo install -d -m 0770 /run/reliquary-grader
docker run -d --name eval-grader --restart unless-stopped --init --privileged --cgroupns host \
  --network none -v /run/reliquary-grader:/run/reliquary-grader --entrypoint python <admin image> \
  -m reliquary.environment.grader.server --use-runsc \
  --socket /run/reliquary-grader/grader.sock --pool-size 16 --metrics-port 0
# the admin container: the socket (mode 0660, root) and a persistent Hugging Face cache
#   -v /run/reliquary-grader:/run/reliquary-grader \
#   -e RELIQUARY_EVAL_GRADER_SOCKET=/run/reliquary-grader/grader.sock \
#   -v <disk with ~12 GB free>:/root/.cache/huggingface
```

The taskset loads `livecodebench/code_generation_lite` (v6, ~4.5 GB, plus its Arrow cache) the first time a
grading opens it. Without the sidecar, rows grade as `grader_error` (`unreachable`), never as wrong answers.

## SciCode

SciCode (`verifiers:scicode-steps`, `tasksets/scicode_steps`) asks each sub-step in a fresh single-turn prompt that
carries the model's own code for the earlier steps, as upstream does. So one SciCode evaluation is a chain of sets,
one per step position (1..15 on the test split), each built from the graded answers of the one before
(`scicode_steps.chain`):

```bash
echo '{"split": "test", "step": 1, "previous": {}}' > args-1.json
reliquary eval build-set verifiers:scicode-steps --out sets/scicode-p1 --taskset-args @args-1.json
# order it on the model, grade; previous = chain.record(previous, {step_number: answer text}) -> args-2.json, ...
# when every position is graded: chain.summary("test", {step_number: passed}) -> main-problem and sub-step rates
```

Its tests need numpy/scipy/sympy/h5py and the 1 GB reference targets, so its programs go to a second grading
service whose sandbox root is the SciCode grading image (`docker/scicode-grader`, stored by digest in the
Hippius registry, bucket `reliquary-task-images`, `reliquary/scicode-grader@sha256:484ab5f18a608c5a7568f2afcc145458d6e02f08632ba9fb0d46d831dc498488`,
0.57 GB; the raw targets file is `datasets/scicode/test_data-48b0272a88b1.h5` in the same bucket). Pull it through a
local `registry:2.8.3` on that bucket (the settings of the image mirror runbook). On the eval host:

```bash
# the bundle: the image's filesystem as the rootfs, this checkout's worker and bundle config
sudo install -d /opt/scicode-bundle
sudo install -m 0644 docker/scicode-grader/bundle-config.json /opt/scicode-bundle/config.json
sudo BUNDLE_DIR=/opt/scicode-bundle WORKER_SRC=reliquary/environment/grader/worker.py \
  PY_IMAGE=<registry>/reliquary/scicode-grader@sha256:484ab5f1...8488 bash scripts/build_grader_bundle.sh
sudo install -d -m 0770 /run/reliquary-grader-scicode
docker run -d --name eval-grader-scicode --restart unless-stopped --init --privileged --cgroupns host \
  --network none -v /run/reliquary-grader-scicode:/run/reliquary-grader-scicode \
  -v /opt/scicode-bundle:/opt/scicode-bundle:ro -e GRADER_BUNDLE_PATH=/opt/scicode-bundle \
  --entrypoint python <admin image> -m reliquary.environment.grader.server --use-runsc \
  --socket /run/reliquary-grader-scicode/grader.sock --pool-size 16 --metrics-port 0 --stdio-max-seconds 600
# the admin container:
#   -v /run/reliquary-grader-scicode:/run/reliquary-grader-scicode \
#   -e RELIQUARY_EVAL_SCICODE_GRADER_SOCKET=/run/reliquary-grader-scicode/grader.sock
```

A step has 300 s of wall clock (upstream: 1800 s) and 8 GiB of address space (10.11's reference needs ~4), one
BLAS thread: size `--pool-size` to the host's memory. Without
`RELIQUARY_EVAL_SCICODE_GRADER_SOCKET` a SciCode row is a grader error, never a zero: the default service's
sandbox has no numpy.
