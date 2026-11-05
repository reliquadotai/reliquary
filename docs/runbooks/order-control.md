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

It needs network to drand and to the Hugging Face hub. Route it with the locations
`reliquary corpus order-nginx --port 8791` prints for the same `RELIQUARY_ADMIN_TASK_PREFIX`; miners need
no setting. The admin service and the order control must run with the same prefix.
