# Checkpoint-epoch proof streaming gate

## Decision

Ticket selection may happen before generation, but proof results are still
consumed at seal in the current experimental epoch runtime.  Proof-on-arrival is
not enabled for checkpoint epochs by this change.

This is intentional.  The fill-window proof scheduler currently offers one
live plan per environment, append-only ranks, and process-local result handles.
A checkpoint epoch has 16 concurrently open logical lanes.  Reusing the fill
plan independently from each lane would therefore either reject the second
lane, make arrival order choose the scheduler order, or lose the ability to
distinguish completed from interrupted proof work after a restart.  None is an
acceptable protocol fallback.

## Staging contract

`TicketedEpochProofCoordinator` is the integration boundary for the future
streaming implementation.  It provides the following fail-closed rules:

1. Only `primary` and already activated `active_backup` tickets can be staged.
2. Every candidate is bound to the epoch, manifest, checkpoint, profile,
   contract, exact logical lane, and that lane's generation randomness.
3. The frozen ticket `selection_rank`, unique inside each lane, not request
   arrival, defines dispatch order.
4. One payload digest can have only one proof owner in the epoch.
5. A repeated dispatch or conflicting terminal result is rejected.
6. Every candidate in a lane must be passed or rejected before the lane can be
   finalized. Partial proof populations never flow into payout or training.
7. A lane finalization claim is idempotent. Its exact replay is observable as
   already claimed; a different replay is rejected.
8. A persisted in-flight proof becomes `quarantined` after restart and cannot be
   dispatched again. The epoch fails closed rather than guessing whether the old
   call ran.

The coordinator snapshot is canonical JSON so a durable journal can persist it
atomically before dispatch and after a terminal result. Recovery rejects
unknown nested fields, illegal pre-freeze state, duplicate ranks or lane
claims, and any finalization digest that cannot be recomputed from terminal
passed records.

## Runtime activation requirements

Streaming can be connected to the live epoch path only after the proof plane
provides all of these capabilities:

- isolated concurrent work for every logical lane in an environment, or an
  equivalent predeclared multiplexed plan with lane-local accounting;
- durable recovery of dispatched and terminal proof results;
- predeclared ticket slots, so a late payload cannot insert an earlier economic
  rank after work has begun;
- rank-independent hydration/extension while preserving the frozen order.

The scheduler publishes these capabilities explicitly. The activation guard
stays closed while any requirement is missing. Once implemented, the runtime
should persist the coordinator's `dispatched` transition before invoking a
proof worker, persist the terminal transition before releasing a result, and
reuse that result at final portfolio selection. It must not submit a second
seal-time proof for a candidate already settled by the streaming coordinator.

## Compatibility

No production profile, generation contract, state response, ranking path, or
payment path changes here. The existing V4/V5 behavior and the disabled epoch
behavior remain unchanged.
