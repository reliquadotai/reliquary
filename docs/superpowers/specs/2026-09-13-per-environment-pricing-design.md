# Per-Environment Pricing — Design

**Date:** 2026-09-13
**Status:** approved in brainstorming, not yet implemented
**Follows:** `docs/superpowers/specs/2026-09-10-task-scoped-emission-pricing-design.md`
and `docs/superpowers/specs/2026-09-12-task-registry-design.md`

## Goal

Give each environment its own discovered price, so that one starving
environment stops taxing every other one, and so the price can buy back the
thing a slow environment actually costs us: window cadence, which is DAPO
updates per day.

## The defect this fixes, verified

`window_ready_round` (`reliquary/validator/emission_price.py`) returns, in its
own words, "the round the SLOWEST environment reached its target", and returns
`None` if **any** environment never reached its own target. The window-level
ratio `r` is therefore confounded across environments:

- one starving environment snaps the price up for **every** environment;
- an abundant environment's surplus is never measured, because its own ready
  round is computed and then discarded;
- and when an environment starves for reasons that are ours — a corpus fault, a
  grading gate, a content cooldown — the mechanism pays every miner more
  because of our own bug.

The per-environment data already exists. `window_ready_round` computes each
environment's ready round and keeps only the maximum. This design stops
discarding it.

**A correction to an earlier claim.** The cap of `r` at 1 is *not* a
consequence of window-level measurement. It comes from missing stage timings:
`outcome_from_archive` uses `max(training_rounds, validation_rounds)` when
those are recorded and falls back to the window span otherwise, and they are
recorded nowhere today. Instrumenting them uncaps `r` on its own, without any
of this design. The two gains are separable and this spec claims only the
second.

## Decisions taken before the design

- **Every environment must fill.** `FILL_CLOSED_MAX_SECONDS` (1800 today) is
  raised; windows wait. The operator picks the value, under two constraints: it
  must be long enough that an ordinary shortage resolves within one window
  rather than tripping the breaker, and short enough that `breaker_timeouts`
  consecutive timeouts still fire in a time a human would call prompt. This
  keeps `N_e` at target, so the trainer's per-environment loss weights — which
  are renormalised over the environments *present* (`training.py:710`) — can
  never give a full gradient share to a handful of noisy samples.
- **The gradient mix stays ours.** What each environment is *paid* is the
  market's business; what proportion of the gradient it occupies is a training
  decision and stays in `ENV_LOSS_WEIGHTS`. Independent per-environment sealing
  was considered and rejected for exactly this reason: an abundant environment
  would produce more steps and quietly own the gradient.
- **On timeout: pay the accepted submissions, burn the rest, do not train.** A
  miner whose work was admitted, proven and graded did the work whether or not
  the window trained; a partial batch fails the training bar and is skipped.
- **On an in-flight failure: unchanged.** Tombstone, nothing paid, nothing
  trained.

## Section 1 — The sensor

The numerator becomes per environment; the denominator stays shared:

```
r_e = (ready_round_e - open_round) / incompressible_rounds
```

`ready_round_e` is when environment `e` reached its own target, counted in
distinct prompts (a target counts groups, and a group is one prompt).
`incompressible_rounds` remains a property of the window —
`max(t_training, t_validation)` — because collecting one environment faster
than the shared training step can consume buys nothing.

Shortage becomes per environment. `filled` was a window boolean; it becomes
`filled_e`. In a normal window every environment fills, so every `r_e` exists.
At a timeout, only the late environments carry `filled_e = False`, and only
those snap.

**On a timed-out window no ratio is computed for anyone.** No training ran, so
`training_rounds` does not exist and the denominator is undefined. Taking
`t_validation` alone would inflate `r_e` precisely on bad windows; carrying the
previous window's denominator would charge one window for another's cost. The
existing semantics already cover this: `r = None` means "no signal", *not*
shortage, and shortage is carried by `filled_e`. The cost is that an
environment which filled quickly during a timed-out window does not get its
price walked down that round — acceptable, since descent continues on normal
windows, and frequent timeouts are the circuit breaker's business, not the
price's.

The archive carries `collect_ready_round` as a scalar today and will carry a
per-environment map. Archives already written keep the scalar and must still
replay, or the history this is meant to be calibrated against is lost.

## Section 2 — Caps, state, and where the price applies

**In the registry.** A task entry carries one scalar `cap`. It gains a
per-environment **split key** alongside it: the task cap says how much the task
may spend in total, the split key says how that budget divides between its
environments. The two move independently, which is the point — the total
budget and the relative worth of an environment are different decisions.

Concretely, the entry's `incentive` block gains one field beside `params`:

```json
"incentive": {
  "mechanism": "rl-discovered-price",
  "params": { "...": "unchanged, shared by every environment", "cap": 0.7 },
  "env_split": { "openmathinstruct": 0.55, "opencodeinstruct": 0.45 }
}
```

`env_split` names a proportion per environment, and `cap_e = cap * env_split_e`.
The proportions must cover exactly the environments the task's protocol profile
declares — an absent or unknown environment is refused at write time, because a
silent default would hand a real budget decision to a fallback.

The invariant doubles and stays inside the same compare-and-swap that already
guards `Σ cap_t ≤ 1`: `Σ_e cap_e ≤ cap_task ≤ 1`. Checking it anywhere else
would turn a guarantee back into an observation.

**Parameters are shared, state is not.** One `PriceParams` for all
environments; one `PriceState` per environment — its price, its `last_good`,
its rolling fill history. Per-environment parameters would multiply by N a
calibration problem that is not solved for N = 1, and would multiply the
numbers that have to be published and defended.

**In the assembler.** `FillClosedBatchAssembler(window_pool: float)` becomes a
per-environment map. `batch_pool_per_env` stops being an even division of one
scalar and becomes each environment's own pool divided by its assembled
batches.

**How a pool divides *within* an environment is not this design's business, and
that is the point.** Since PR #253 the split is policy-driven:
`split_fixed_environment_pool` pays one fixed share per selected group
(`share = pool / slots`) and, in its own words, *"unfilled slot shares burn"*;
`split_environment_pool` remains for the legacy policy that divided by
`eos_tokens`. This design changes only *which pool each environment receives*,
never how that pool divides inside it, so it composes with either policy.

That independence is worth stating, because the fixed-group policy answers a
question this design otherwise had to: whether supply should be counted in
groups or in tokens. Counting groups removes the length incentive, and the
per-environment price then absorbs the fact that one environment's groups cost
a miner more than another's — which is what a price is for. We do not have to
model the cost ratio; the market finds it.

And it makes the burn principle uniform at every level: unfilled slots burn
inside an environment, an unspent environment pool burns inside a task, an
unspent task cap burns inside the subnet. PR #253 reached that choice
independently, one level below this design.

**The residue burns with no new code.** An environment whose price sits below
its cap simply never has the difference assigned to anyone, and
`burn_weight = max(0, 1 - registered_total)` absorbs it. The same property
holds at every level of the tree: subnet, task, environment.

**This is writer-side only.** A weight-only validator replays
`rewards_by_hotkey` verbatim from the archive; it does not recompute the split.
Changing how the pool divides between environments therefore changes no reader
and needs no fleet coordination — the same property that made the original
price design deployable.

## Section 3 — The failure path

**Three window statuses instead of two.** `outcome_from_archive` skips
`window_status == "aborted"` because "their timings describe the abort, not the
market". A timeout describes the market exactly — it is the shortage signal —
so it needs its own status: `completed` / `timed_out` / `aborted`, with the
replay skipping only `aborted`. Without this, the window carrying the most
useful information is the one thrown away.

**The timeout seal is nearly free.** The assembler accrues payment as batches
are assembled (`_accrue_payment_locked`), so at the timeout what was earned is
already computed. It is emitted as the archive's rewards, with **no training
payload**, and each environment's unearned remainder burns.

**The partial remainder is not paid, deliberately.** `close(allow_partial=True)`
can force a final incomplete batch. Paying it would over-reward its groups,
because the pool divides per assembled batch and a half-empty batch gives each
of its groups a larger share than a full one — an incentive to arrive in the
last carriage. The cost is real: a miner whose group sits in that remainder
worked without being paid.

**The circuit breaker freezes the price and does not touch the mix.** After
`breaker_timeouts` consecutive timeouts attributable to the same environment —
a new `PriceParams` field, starting at **3** and calibrated in shadow like
every other number in that block — that environment's price stops rising and
`logger.critical` names the environment, the count and the frozen price. The
frozen state is stamped into the archive so the freeze is auditable after the
fact rather than only visible in a log. The default hypothesis flips
from "the market is short" to "this environment is broken on our side". It does
**not** drop the environment from the training mix — that would change the
gradient's composition automatically, handing the market the decision this
design exists to keep.

The breaker is mandatory rather than optional: because every environment must
fill, a structurally dry environment no longer degrades the gradient, it halts
the subnet. Without the freeze, its price rises indefinitely while nothing
unblocks.

## Section 4 — Testing

- **The test that decides everything:** a window where one environment fills
  quickly and another never fills must snap **only the late environment's
  price** and leave the other's untouched. This fails today, and it is the
  whole purpose of the work. Write it first.
- **The witness:** declaring the split key in equal proportions must reproduce
  today's payouts exactly. As everywhere on this branch, the proof is that no
  existing assertion on today's values was edited to get there.
- `r_e` per environment against the shared denominator; at a timeout, no ratio
  for anyone but `filled_e` false for the late environment only.
- An archive carrying the old scalar `collect_ready_round` still replays.
- `Σ cap_e ≤ cap_task ≤ 1` refused at write time, inside the CAS, at both
  levels.
- The timeout seal pays what was accrued, writes no training payload, and does
  not pay the partial remainder.
- An in-flight failure still produces a tombstone paying nothing.
- The breaker freezes the price after N timeouts, raises the alert, and leaves
  `ENV_LOSS_WEIGHTS` untouched.

## Out of scope

- **Arming the price.** This is measurement and allocation shape only. The
  controller stays in shadow, and its blocker stands: the price walk lives in
  memory and a restart resets it.
- **Per-environment controller parameters.** Shared parameters, independent
  prices.
- **Independent per-environment sealing.** Rejected: it would give the market
  the gradient's composition.
- **Automatic removal of an environment from the mix** on a tripped breaker,
  for the same reason.
- **Stage timing instrumentation** (`training_rounds` / `validation_rounds`).
  Separable: per-environment pricing works without it, using the fallback
  denominator. But it is what uncaps `r` and makes a proportional descent
  implementable, and it is small and worth doing on its own.
