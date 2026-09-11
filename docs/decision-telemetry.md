# Decision telemetry (opt-in, observational)

This patch does not change admission, priority, proof budgets, rewards or the training payload schema.

Set `RELIQUARY_DECISION_TELEMETRY_ENABLED=1` and `RELIQUARY_DECISION_TELEMETRY_DIR` to a private, writable persistent directory **when starting a qualified runtime**. Default is disabled. Merely copying files into a running Python container does not activate hooks.

Events include eligible candidates, proof ready sets/reservations/job identities/results, selection ready sets, durable assembly, miner generation/proof-building/submit outcomes, trainer filtering and post-optimizer receipts. Metadata construction errors are caught before they can affect application behavior. The writer uses a bounded queue, 16 MiB files, 16 files per process run, 0600 permissions and a 256 KiB event ceiling. Archive old runs separately; retention is per run. Collectors must record missing run prefixes/tails, writer drops and restart boundaries.

`optimizer_success` means `optimizer.step()` returned successfully. The subsequent group receipt reconciles eligible microbatch rows against processed rows; `reconciled=false` must never be treated as actual group attribution. Success can precede a later scheduler/publication failure. Decoded group origins retain journal/window/environment/checkpoint identity across accumulation. A generic trainer return or cursor advance is not evidence of a parameter update.

The reference miner emits local generation failures, incomplete generation and stale-release discards. Private miner overlays require their own patch/review, particularly filters before generation; missing attempts must remain unknown. Durations are call wall time, not CUDA kernel timing. No token vectors, signatures or wallet secrets are emitted.

## External baseline without application restart

```sh
python3 scripts/collect_decision_baseline.py --directory /private/path/samples --container CONTAINER
# Controller: additionally pass --health-url http://127.0.0.1:PORT/health
python3 scripts/decision_report.py /private/path/samples/baseline-*.jsonl
```

This bounded collector runs for 48 hours, every approximately 10 seconds. It records GPU utilization, container identity/restarts and selected existing log lines. Hourly files are capped at 16 MiB with 48-file retention. It records tail-limit and command failures; collection stops on host reboot. GPU percentages are sampled utilization, not generated-work accounting. Logs overlap by one second; deduplicate timestamped lines before counting optimizer steps. No service is restarted by the collector.

## Conditional comparison

```sh
python3 scripts/decision_report.py /private/path/events-*.jsonl > report.json
```

The eligible-order alternative only uses each **observed** ready set. It does not model different proof completion times, earlier alternate picks or changed miner behavior. Missing eligible identities exclude that comparison. It is not a causal learning-quality result or a production policy switch.

Before a decision: obtain consecutive complete windows across checkpoint transitions; reconcile eligible → reserved → terminal → picked → durable → actual optimizer use. Pair learning experiments on held-out evaluation, pinned source/checkpoint and controlled token budgets. Measure one small pilot's cost before expanding. Winner-only archives cannot label rejected candidates as useless.

Deployment must preserve trainer optimizer state and controller/proof transport authorization. The transport hash includes `batcher.py`; deploying only that file can break mTLS proof compatibility. Do not bypass the authorization checks or restart a live trainer solely to install telemetry when optimizer state is not persisted.
