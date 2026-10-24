# Agentic corpus contracts

Agentic corpus jobs use immutable checkpoint and environment identities,
signed trajectory submissions, and per-assistant-span activation proofs.
The native protocol validates token and span bounds before admitting work.
The selected task contract controls verification and grading behavior.

Execution, proof verification, grading, delivery, and incentive settlement
are recorded separately. Qualification must cover the selected runtime,
checkpoint, environment artifacts, and proof rules before live execution.

See the [native execution reference](../runbooks/agentic-corpus-swe.md) and
[split-process runtime reference](2026-10-02-corpus-split-processes.md).
