# Control-plane and cooldown hardening

Status: draft implementation. Nothing in this document authorizes deployment.

## Wire contract

`GET /miner-state` is additive; legacy `GET /state?env=...` remains available
for rolling compatibility. Schema version 1 returns all active environments in
one response. Each environment has:

- `prompt_range: [start, end]`
- `encoding: "bitset-v1"`
- a base64 cooldown bitmap whose bit `N` represents `start + N`
- `cooldown_count`, validated against the bitmap

Only the deterministic active prompt slice is represented. At the default
`PROMPT_RANGE_SIZE=5000`, the raw bitmap is at most 625 bytes per environment,
regardless of training-run age. Responses are serialized and gzip-compressed
once, carry `ETag` and `Vary: Accept-Encoding`, and support conditional 304
polls. The pure-ASGI fast path also serves constant-size between-window 503s
with `Retry-After: 1`.

A local 100,000-entry cooldown fixture measured 589,145 bytes for legacy JSON
(209,115 gzip) versus 1,254 bytes for one-environment `/miner-state` JSON (278
gzip), a 469.8x uncompressed reduction. Bitmap construction took 1.3 ms and
probes only the 5,000-entry active slice, not the full cooldown history.

The reference miner prefers `/miner-state`, caches it by ETag, and falls back to
legacy per-environment state only when the new endpoint returns 404. Legacy
cooldowns are fetched once per window, must all match the base window and
randomness, and are never copied across environments.

## Health and feedback

- `/livez` is dependency-free process liveness.
- `/readyz` returns 503 for current serving failures such as unusable prompt or
  registration data, required proof-scheduler loss, critical process health,
  excessive event-loop/endpoint latency, or archive backlog.
- `/health` remains the rich diagnostic document, refreshed off the event loop
  and served from pre-serialized bytes. It includes control-plane payload size,
  compression size, cache/304 counters, response bytes, and 60-second request
  counts.
- `/miner-verdicts/{hotkey}?after=N` adds a monotonic cursor and explicit ring
  rollover detection. Legacy timestamp-based `/verdicts` remains available.

## Cooldown persistence

Periodic prompt/content snapshots run in a background task and cannot delay the
next window OPEN. JSON encoding and gzip compression run in worker threads.

`COOLDOWN_DELTA_SNAPSHOTS_ENABLED=0` is the safe default. When enabled, prompt
cooldown persistence writes small contiguous gzip deltas every snapshot cadence
and publishes a full compacted snapshot every
`COOLDOWN_COMPACTION_INTERVAL_WINDOWS` (default 1000) before deleting obsolete
deltas. Expired entries are pruned during compaction.

Do not enable deltas until every validator binary that might be used for
rollback supports schema version 2 delta replay. An older binary reads only the
full snapshot and can otherwise restore state older than the archive replay
horizon.

## Draft cutover sequence

1. Deploy the validator code with delta snapshots disabled. Verify `/state`,
   `/miner-state`, `/livez`, `/readyz`, and both verdict endpoints in shadow.
2. Confirm `/health.control_plane.miner_state_payload_bytes`, gzip bytes,
   cache-hit count, 304 count, request rate, event-loop p99, and endpoint p99.
3. Release miners that prefer `/miner-state`; legacy validators remain usable
   through automatic 404 fallback.
4. Wait until the active miner fleet predominantly uses the new endpoint, then
   rate-limit or eventually retire legacy `/state` in a separate change.
5. Only after every rollback validator understands deltas, enable
   `COOLDOWN_DELTA_SNAPSHOTS_ENABLED=1` on one canary. Verify delta restore in a
   non-serving restart rehearsal and observe at least one full compaction.
6. Roll out the delta flag gradually. Keep the last full snapshot and local
   content snapshot as rollback evidence.

## Rollback

The endpoint/miner change is backward compatible: revert miners first and they
will use legacy `/state`; reverting the validator leaves new miners on their
automatic legacy fallback. Do not roll back to a pre-delta validator while
delta mode is enabled unless a fresh full snapshot has been successfully
published after the most recent delta.

Atomic miner checkpoint activation stages both generation and proof models and
advances the local checkpoint hash only after both loads succeed. This has a
peak-memory cutover risk because the old pair remains referenced during staging;
GPU qualification must demonstrate sufficient headroom before release.
