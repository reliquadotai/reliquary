# Reliquary

Decentralized GRPO training for large language models on Bittensor subnet 81.

Reliquary is a coordination protocol that turns a set of independent GPU operators into a single distributed RLHF pipeline. Miners generate cryptographically-proven rollouts; the validator aggregates them into a GRPO training batch, updates a live LLM checkpoint, and publishes the result to Hugging Face — all without trusting any single participant.

## The incentive shift, in one line

**Old subnets:** miners are paid per rollout. The competition is "do as many rollouts as you can."

**Reliquary:** miners are paid for useful verified contributions to the trainer. The competition is "find the rollouts I need to train on" — i.e. predict which prompts sit at the policy's current learning frontier (group-σ in the trainable band, not yet in cooldown). A miner who picks well lands on winning prompts and earns emission; a miner who picks poorly burns their own rollouts on rejects such as `OUT_OF_ZONE` or `REWARD_DISTRIBUTION`.

This converts DAPO's reactive Dynamic Sampling filter into an ex-ante prediction market: the generate-then-discard cost is pushed out of the validator and onto the miner who guessed wrong. As the policy matures and the learning frontier narrows, selection intelligence becomes more valuable, not less. See [docs/concepts.md](docs/concepts.md#the-thesis) for the full argument.

## What it does

Each training window contributes to a possible GRPO step. The cadence is event-driven: a window seals once enough valid distinct-prompt rollout groups land, then final selection is ordered by drand/canonical rules rather than validator-side TCP latency. The live policy is `Qwen/Qwen3.5-4B`, and the current trainer can run a mixed OpenMath + OpenCode environment. The validator recomputes rewards itself, quarantines suspicious selected windows from training, and retains clean partial batches in a bounded, checkpoint-bound accumulator until every active environment reaches its configured share. It then runs one PPO-clipped step with a KL penalty against the frozen reference. Updated weights are published to a public HF repo on the configured publish cadence.

The network produces three artefacts: a continuously-trained model (published to HF every ten trained windows), a per-window rollout dataset (archived to R2), and a signed checkpoint manifest (served from `/checkpoint`) that lets anyone verify the chain of custody from a base model through every training step. The audit trail is cryptographic — each rollout carries a GRAIL sketch that lets the validator re-run the forward pass and confirm the generation came from the announced checkpoint.

Validators hold stake and run the training loop. Miners hold hotkeys, run GPU inference, and earn emission proportional to their smoothed share of selected prompt rewards over a rolling 72-window EMA; the main optimization surface for a miner is predicting which prompts sit at the policy's learning frontier while producing clean, verifiable, naturally terminated rollouts. Downstream consumers — researchers, fine-tuning pipelines — pull the published HF checkpoint or the R2 rollout dataset directly.

## Quickstart

- To mine: see [docs/mining.md](docs/mining.md)
- To validate: see [docs/validating.md](docs/validating.md)
- To understand the mechanism: see [docs/concepts.md](docs/concepts.md)

## Architecture at a glance

```
┌─────────────┐    HTTP    ┌─────────────┐   HF push   ┌──────────────┐
│   Miners    │ ─────────▶ │  Validator  │ ──────────▶ │   HF Hub     │
│  (N nodes)  │ ◀───────── │  (1 node)   │             │ (model repo) │
└─────────────┘ /submit    └──────┬──────┘             └──────┬───────┘
     ▲         /state             │                            │
     │         /checkpoint        │ weights                    │ pull
     │                            ▼                            │
     │                   ┌──────────────┐                      │
     │                   │  Bittensor   │                      │
     │                   │  chain       │                      │
     │                   │  (set_weights│                      │
     │                   │   every 360  │                      │
     │                   │   blocks)    │                      │
     │                   └──────────────┘                      │
     │                                                         │
     │                   ┌──────────────┐                      │
     └───────────────────│     R2       │◀────── archive ──────┘
                         │ (rollouts +  │         (per window)
                         │  dataset)    │
                         └──────────────┘
```

Miners submit rollout groups to `/submit` and poll `/state` for checkpoint updates. The validator trains healthy windows, publishes to HF, and broadcasts weights on-chain once per subnet epoch (~360 blocks on netuid 81), aligned to the epoch boundary so all validators converge on identical weights. Miners pull new weights via `/state` → HF `snapshot_download`. R2 stores the per-window rollout archive, including reject summaries and training-quarantine metadata; the validator reads it at startup to rebuild cooldown/hash state.

## Status

- **v1** — verifiable-inference dataset production (shipped, deprecated)
- **v2** — GRPO market with in-subnet training (shipped)
- **v2.1** — batch-driven windows, HF checkpoint distribution, EMA scoring (shipped)
- **v2.3** — drand ordering, multi-miner-per-prompt, prompt emission split (shipped)
- **v2.4** — Qwen3.5 reset, mixed OpenMath/OpenCode training, private validator-authoritative OpenCode grader (current rollout)
- **v2.5 direction** — private/generated reward tasks and stronger anti-selection protocol design (planned)

## License

MIT — see `LICENSE`.
