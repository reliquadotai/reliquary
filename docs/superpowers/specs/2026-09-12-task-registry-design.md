# Task Registry — Design

**Date:** 2026-09-12
**Status:** approved in brainstorming, not yet implemented
**Follows:** `docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md`

## Goal

Let a new task be launched by writing one object, with nothing to hand-tune per
box, and with the chain's `Σ weights ≤ 1` guaranteed rather than hoped for.
Later, a product does the same write automatically.

## What a task is

A **task type** carries a validation system, an incentive mechanism, and
parameters. Today there is one type: RL.

A **task instance** is one running validator with its own parameters — its own
starting incentive, its own decay, its own maximum.

Most of the type already exists in code. `ProtocolProfile`
(`reliquary/protocol/profiles.py`) carries the model, revision, sampling,
environments, token budgets and answer formats; there are nine of them, their
ids already encode model + mechanism + version
(`qwen3-4b-base-dapo-fill-closed-v6`), and `to_generation_contract()` plus
`canonical_sha256` already turn one into an immutable, addressable contract.
What a profile lacks is the incentive mechanism.

The instance's maximum also already exists: it is `PriceParams.cap`, today
`1.0`. This design does not add the concept — it makes an existing field
per-task, so the invariant is literally `Σ params.cap ≤ 1`.

## The registry

One object, `reliquary/tasks/registry.json`, written compare-and-swap.

```json
{
  "registry_version": 1,
  "tasks": {
    "default": {
      "profile_id": "qwen3-4b-base-dapo-fill-closed-v6",
      "profile_sha256": "<sha256 of the profile's generation contract>",
      "incentive": {
        "mechanism": "rl-discovered-price",
        "params": { "start": 1.0, "decay": 0.99, "rounds_per_step": 1000,
                    "deadband": 0.80, "snap": 1.20,
                    "floor": 0.05, "cap": 1.0, "median_rounds": 9 }
      },
      "status": "active"
    }
  }
}
```

**The profile is named, not copied.** The entry pins `profile_id` and the
sha256 of its generation contract. The code stays the source of profiles —
miners compile them — and the registry only says which one, exactly. Copying a
profile into R2 would create a second source of truth that can drift from the
code.

**`mechanism` is explicit.** Today there is one value. A validator that does
not recognise the named mechanism refuses to start rather than guessing.

**Price parameters become per-task.** This reverses a recorded decision:
`PriceParams` is documented *"Versioned in the image, never env-overridable"*.
The reason was that a per-box environment variable can diverge silently and
leave no trace. A single versioned R2 object, readable by everyone and
hashable, does not have that failure mode. The rule becomes "never settable per
box", not "never settable".

## Lifecycle

**Create.** Read the registry with its ETag, validate, add the entry, write
back with `If-Match` (`If-None-Match: *` on first creation). On a 412 the
writer re-reads, **recomputes the sum against the winner's entry**, and either
succeeds or fails naming the winner. The `Σ cap ≤ 1` check lives inside that
compare-and-swap window; that is the only thing that makes the invariant a
guarantee rather than an observation.

Validated before the write: the id is normalised and free; `profile_id` exists
in the image; `profile_sha256` matches that profile's generation contract;
`mechanism` is known; parameters are in range; `Σ cap ≤ 1`.

The existing compare-and-swap in `reliquary/trainer/publisher.py:511` is the
precedent — same endpoint, same pattern, in production. It does **not** handle
the precondition failure, so the registry writer must add that.

**Start.** The validator reads the registry and finds its own `TASK_ID` entry.
Four refusals, each with a message naming the cause and a distinct exit code,
matching the GPU lease's shape:

- entry absent — the task was never declared;
- `Σ cap > 1` — the registry is oversubscribed;
- `profile_sha256` disagrees with the image's profile — the registry and the
  binary disagree about what this task is;
- unknown `mechanism`.

Otherwise it builds this task's `PriceParams` from the entry and keeps them for
the life of the process. The cap is not re-read per window: a running task is
not retuned. Changing parameters means stop, edit, restart.

**Retire.** Removing a task does **not** immediately free its cap. The weight
EMA replays 216 archives with a time constant near 28 hours, so a retired task
keeps paying while it decays. Retiring A (cap 0.5) and launching B (cap 0.5)
straight away would put the real sum above 1 for hours with nothing incorrect
ever written.

The entry therefore becomes `status: "retired"` with a `retired_at` round, and
its cap stays counted until its contribution has decayed.

**The common case is exempt.** Relaunching a task with different parameters
reuses the same `task_id`: its archives form a single EMA series bounded by its
declared cap, so there is no double-count and no waiting. The reservation only
costs when budget moves from one task to a *different* one.

Rejected alternatives: releasing the cap at the measured rate of decay (exact,
but it imports an EMA computation into registry validation, coupling two things
better kept apart); accepting the transient overshoot and letting
`_replay_ema`'s clamp absorb it (it dilutes the running task's miners, which is
the invariant this work exists to protect).

## Where it lands

- **`reliquary/shared/task_registry.py`** (new) — pure: the entry type,
  canonical parsing, per-entry validation, `Σ cap ≤ 1`. No I/O, so the rule is
  testable in milliseconds and lives in exactly one place, like
  `reliquary/shared/task_id.py`.
- **`reliquary/infrastructure/task_registry_store.py`** (new) — read with ETag,
  write compare-and-swap, handle 412 by re-reading and recomputing, give up
  after five rounds, naming the winner.
- **`reliquary tasks create | list | retire`** — the simple way to launch a
  task, with `PRODUCTION_PRICE_PARAMS` as pre-filled defaults. A product later
  calls the same pure function, not the CLI.
- **`reliquary/cli/main.py`** — the startup read and the four refusals, beside
  the GPU lease.
- **`reliquary/validator/emission_price.py`** — `PRODUCTION_PRICE_PARAMS` stops
  being what the controller reads and becomes the defaults offered at creation.
- **`reliquary/validator/weight_only.py`** — second line of defence: if the
  tasks seen in R2 exceed their declared caps, or a task with archives is
  absent from the registry, **abstain** instead of clamping silently.
- **`reliquary/constants.py`** — `RELIQUARY_TASK_EMISSION_SHARE` is removed.
  The cap comes from the registry, not from a per-box variable.

**Flow.** `tasks create` writes the registry → the validator starts, reads its
entry or refuses → each window pays `price ≤ cap` → the archive stamps
`task_id`, the pinned `profile_sha256` and the cap → the weight replay sums per
task and cross-checks the registry.

**The archive field keeps its name.** Archives already carry
`task_emission_share`; its value becomes the entry's `cap`. Renaming it would
be a reader-side change needing the whole fleet, and it buys nothing.

## Error handling

An unreadable registry **refuses startup**. This is deliberately the opposite
of the GPU lease, where an unusable lease directory warns and continues: a
missing lease is an environment fault that cannot make a payment wrong, while
an unreadable registry means we do not know what we are allowed to pay. The
lease protects hardware; the registry protects money.

## Testing

- The pure rule, without R2: `Σ` accepting exactly 1.0 and rejecting 1.0 + ε,
  negative caps, booleans; unknown mechanism; mismatched `profile_sha256`;
  out-of-range parameters.
- **The test that matters**: a fake S3 client that fails the first `If-Match`
  with a 412, proving the writer re-reads, recomputes the sum against the
  winner, and then either succeeds or fails naming it. A test that does not
  simulate the race does not test the invariant.
- The four startup refusals, each with its own exit and message.
- **The identity test**: a registry holding only `default` at `cap: 1.0` with
  today's profile must produce the same window pool, the same archive fields
  and the same weights as today. As with task isolation, the proof is that no
  existing assertion on today's values was deleted or relaxed to make the
  change pass.
- Two tasks: A at 0.6 and B at 0.4 each converge to their cap, `Σ = 1`, burn
  zero; A retired, and B cannot take A's cap until A's tail has decayed.

## Out of scope

- **Arming the price.** This makes per-task parameters possible; it does not
  flip the switch. The blocker stands: the walk lives in
  `self._price_shadow_state`, so a restart resets it to `start` and would hand
  the whole pool back to miners. It must be seeded from the last archive first.
- **On-chain commitment of the caps** — needed only once third parties launch
  tasks.
- **A second task type.** `mechanism` makes room for one; none is written.
- **The task contract applied by the miner** (§7 of the pricing design).
- **Scoping `CANDIDATE_MANIFEST_KEY` and stamping `task_id` on training
  payloads at enqueue** — prerequisites for a second *trainer*, not a second
  task.
