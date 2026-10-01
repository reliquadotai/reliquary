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

## Run

```bash
docker run -d --name order-control --network host \
  --env-file <subnet bucket credentials> -e RELIQUARY_ADMIN_TASK_PREFIX=order- \
  <order-control image> --netuid 81
```

It needs network to drand and to the Hugging Face hub. Route it with the locations
`reliquary corpus order-nginx --port 8791` prints for the same `RELIQUARY_ADMIN_TASK_PREFIX`; miners need
no setting. The admin service and the order control must run with the same prefix.
