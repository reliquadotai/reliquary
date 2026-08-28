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
python3 scripts/qualify_episode_suite.py \
  --tasks-per-env 64 \
  --json-out /tmp/reliquary-episode-cpu.json
```

Every reference case must report reward `1.0`, success and exact replay. The
same command also proves that a tampered observation cannot reproduce the
canonical transcript and that a known-wrong action receives `0.0`. The report
records repository dirty state, runtime/package versions, profile identity,
generation-contract digest and aggregate environment-manifest digest. Repeat
on two independent hosts and compare every task ID and trace digest. Latency is
diagnostic and need not match.

## Real-model smoke gate

Start on a GPU host with only the flagship environment and the pinned local
checkpoint:

```bash
python3 scripts/qualify_episode_suite.py \
  --environment reliquary_stateful_tools_v1 \
  --tasks-per-env 8 \
  --rollouts 16 \
  --model-path /models/Qwen3-4B-Base \
  --model-revision 906bfd4b4dc7f14ee4320094d8b41684abff8539 \
  --device cuda:0 \
  --json-out /tmp/reliquary-episode-stateful-model.json
```

This checks turn-by-turn generation and exact replay, hashes the local model
artifact, verifies the revision, and reports reward mix, GRPO-eligible groups,
invalid actions, assistant/total tokens, latency and peak VRAM. It fails closed
unless the environment has at least one success, one failure and one rollout
group whose reward standard deviation reaches the active protocol threshold.
It is a qualification gate, not a benchmark, and does not promote weights.

After stateful passes, repeat with retrieval, then workspace. Finally omit
`--environment` to qualify the combined suite. A uniformly failing base model
means the environment is technically replay-safe but not yet learnable with
the current prompt/model frontier; do not weaken the verifier to force a pass.

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
