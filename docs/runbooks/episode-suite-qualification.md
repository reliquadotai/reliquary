# Episode suite CPU/GPU qualification

The episode profile is dormant by default. Activate it only on a development
validator/miner pair:

```bash
export RELIQUARY_PROTOCOL_PROFILE=qwen3-4b-reliquary-episode-v7-dev1
```

Use the same exact model revision, repository commit and environment manifest
on every participating host.

## CPU gate

Run before copying the branch to a server:

```bash
python scripts/qualify_episode_suite.py \
  --tasks-per-env 64 \
  --json-out /tmp/reliquary-episode-cpu.json
```

Every case must report reward `1.0`, success, and exact replay. Repeat on two
independent hosts and compare every task ID and trace digest. Latency is
diagnostic and need not match.

## Real-model smoke gate

On a GPU host with the pinned local checkpoint:

```bash
python scripts/qualify_episode_suite.py \
  --tasks-per-env 4 \
  --rollouts 2 \
  --model-path /models/Qwen3-4B-Base \
  --device cuda:0 \
  --json-out /tmp/reliquary-episode-model.json
```

This checks turn-by-turn model generation and reports syntax, reward, turns,
assistant token volume and wall time. It is a smoke test, not a benchmark and
does not promote weights.

## Full development-network gate

Run the normal miner, validator, detached trainer and archive path with only
the episode profile. Keep training disabled for the first shadow window.
Require:

- zero canonical transcript, replay, state-digest and GRAIL v8 mismatches;
- assistant-only token counts preserved through the detached payload;
- mixed reward groups at the configured 16-rollout frontier;
- proof and admission p95 below the development window budget;
- no Math/Code profile contract snapshot changes;
- replay-identical archives after validator restart;
- a wrong-model and tampered-observation adversarial suite that always rejects.

Enable optimizer steps only after those gates pass. Start with
`reliquary_stateful_tools_v1`, add retrieval second, and add workspace only
after sandbox throughput is measured.

## Benchmark and announcement gate

Freeze the pre-RL Qwen3-4B baseline before training. Evaluate the baseline and
candidate under the same pinned harness on sealed Reliquary world-family
holdouts and separately on BFCL V4 Multi-Turn and Memory. Keep Math, Code,
IFBench and LiveCodeBench as retention lanes.

An announcement may say Reliquary generated, replay-verified, selected and
trained on these trajectories only when the published evidence includes the
model revisions, code revision, signed profile, environment manifest, trace
lineage, training-window manifests and exact evaluation harness/results.
