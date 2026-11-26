# Order caps that follow what an order pays

Status: coded (`admin/cap_sizing.py`), on in production (`RELIQUARY_ORDER_CAP_SIZING=on`); revised 2026-11-16 to follow pay. Applies
to order jobs (Batch and Evals). Builds on the order price auction (`2026-11-10-order-price-auction-design.md`).

## 1. Problem

Every order job reserves a fixed share of the miners' emission: 0.25 for a generation order, about 0.042 per
eval job (0.125 for a three-job eval order). The registry refuses a new task once the declared caps would
total more than 1.0, and the platform then holds the order in `waiting_for_cap`. With generation orders, at
most four run at once.

An order pays far less than it reserves. It pays `min(cap, price x verified tokens)` per period, and its
verified tokens are bounded by its audit cards. Measured on 2026-11-10: a busy eval job reserved 0.042 and
paid 0.0098 in a period (2.49M tokens at 0.58 alpha per million). The whole pool, 147.6 alpha per period,
pays about 250M tokens at that price. The pool is never near full; the reservations are. What is not paid
burns.

The sum-of-caps rule cannot simply be relaxed: `validate_registry` enforces it on every read, in every
validator. So the caps themselves must shrink to what each order can pay.

## 2. What an order can pay

An order's paid tokens per period are bounded by its audit cards, not by its miners:

```
spend/period  <=  (completion tok/s x price + prompt tok/s x input price) x PERIOD_SECONDS x cards / audit share
```

in pool share, with the job's price ceiling (`max_price_per_token`, and the input price when there is one).

- **tokens per second per card**: the dispatcher measures the completion and prompt tokens a card audits per
  second it holds a lease (`tokens_per_second`, beside `items_per_second` from #476), per model, published in
  the control status (`audit_tokens_per_second`, `audit_prompt_tokens_per_second`). A model not measured yet
  takes 2000 tokens a second for each, high on purpose: such an order is sized close to the cap it asked for.
- **audit share**: the job's `audit_q` (Evals 1.0, Batch 0.2). Records that pass unaudited are paid without
  using the card, so a Batch card lets about five times its own throughput be paid. Using `q` and not the
  measured share (which also counts probation audits) overestimates, which is the safe side.
- **cards**: the order's `audit_executors`. Several jobs of one model share its cards; each is sized as if it
  had them all, which overestimates again.

The price auction only lowers the price under the ceiling, so the real spend stays under this bound.

## 3. Cap sizing

2026-11-16 revision. The audit bound of section 2 grew past every asked cap: the ceilings rose to about
2.08 alpha per million and a 27B card audits 2,700 tokens a second, so one Batch card "could" let 0.86 of a
period through. Taken as the cap (the first version used `max(estimate, paid)`), every Batch order sat at
0.25 again: two drill orders of about $8.80 each and one draining order held 0.75 of the pool, and five
eval orders waited. The estimate says what an order could pay, not what it pays.

The settler pays a period with the cap a task holds **when that period settles**. A cap therefore need not
be reserved ahead; it only has to cover the period's pay by then, and a resize runs every 5 minutes against
72-minute periods. So the cap follows the order's real pay, and the audit estimate is only a bound on the
part that is projected:

```
paid      = max(paid in the last 3 periods, accrued in an open period)
projected = accrued in an open period / max(0.25, share of the current period passed)
ahead     = projected x HEADROOM, or START while the order has paid nothing
cap       = min(cap_max, max(CAP_FLOOR, paid x HEADROOM, min(ahead, estimate x HEADROOM)))
```

- What the order really paid always counts, past the estimate: the settler pays it with this cap.
- `START` is `START_CAP` (0.02), or less when the whole order is worth less: its prompts x samples x the
  most tokens a sample can hold (an episode's `max_total_tokens` when it plays one) at the ceiling, with
  headroom. An order whose miners produce more raises its cap within a resize or two.
- The projection is at most x4 early in a period (`MIN_ELAPSED_SHARE` 0.25).

- **At creation**, with the switch on and a price ceiling, the admin sets `cap = floor` to the formula with
  nothing paid yet (the start) and keeps the asked cap as `cap_max`. `HEADROOM` is 1.5, `CAP_FLOOR` 0.002.
  A replay of the order compares the asked cap, so it is not taken for another contract. The registry guard
  is unchanged: a new order waits only when the sized caps no longer fit, which now means the orders
  running really pay close to the whole pool.
- **Every 5 minutes** the admin resizes each active task with a `cap_max` with the formula, from the control
  status (skipped when older than 10 minutes).
  - The control publishes per job (`jobs.<id>.sizing`): its model, cards, what the settler paid in the last 3
    periods (`recent_paid`) and the most an unsettled period has accrued at the ceiling (`open_value`).
  - The estimate is recomputed with the current rate and card count, so a card added mid-order raises the cap.
  - A change under 10 % of the current cap is not written.
  - A raise goes as far as the free room allows with `NEW_ORDER_RESERVE` (0.05) kept free: room for one new
    order's start, a corpus order's `START_CAP` or a training order's `DEFAULT_TRAINING_CAP`
    (`admin.service.bounded_raise`, from reliquary#593). The reserve binds every order alike
    (`cap_sizing.raise_room`), so a new order keeps its start until a period of it settles: sized down in its
    first minutes, the reserve could keep it from climbing back.
    - The free room keeps declared caps (retired ones included) within 1.0 and paying corpus caps within
      `RELIQUARY_ADMIN_POOL_MAX` (`admin.service.cap_room`). The cap is rounded down to the micro-share, never
      up into the reserve.
    - Every raise first lists the retired orders, as every resize did: a drained one gives its room back even
      while raises wait.
    - It is written only if it raises the cap by 10 % or more. Only an open order is raised: never a paused one
      (it can still be lowered) or one at cap 0 (being closed).
    - Every resize's guard re-checks on the registry written that the task is still active at the cap read
      (`unchanged_guard`), a raise's also that it is still open and the room still there. A resize therefore
      never undoes a concurrent write to the registry: an operator's PATCH, a close. A pause is seen if it was
      set before the write read the registry (admission lives in its own object). A replay of the same write
      after a lost response goes through.
    - The room goes to whichever order asks first in a pass (task-id order). A raise the registry would have
      taken whole can be held short by the reserve.
  - Before reliquary#593, a raise the registry refused kept the cap: on 2026-11-25 an order asking 0.25 every
    5 minutes stayed at 0.0323 for hours when it could have held 0.152 (0.1197 was free).
- The settler pays a period with the cap it holds when it settles that period, so a cap lowered mid-period
  also bounds the work already done in it. A resize therefore never goes under the open period's accrued
  value x `HEADROOM`. Lowering keeps `tail_cap` (the highest cap paid), so weights stay bounded by
  `pay_ceiling` as for any lowered task.

## 4. When the pool is really full

- **Admission:** the platform already holds an order refused by the cap guard in `waiting_for_cap` and
  retries. With sized caps this happens only when the orders running can pay close to the whole pool.
- **Backstop:** the weight combination already clamps each task to its cap and rescales every hotkey
  proportionally when the combined total exceeds 1.0 (`WeightOnlyValidator`, logged). Sized caps keep that
  from happening; it stays as the last guard.

## 5. Order of magnitude (first version, 2026-11-11 prices)

At the 0.58 alpha per million ceiling, one Batch card (audit share 0.2) lets about 12.5M tokens a period be
paid: about 7 alpha, 5 % of the pool, cap 0.075 with headroom. About 13 such cards fit in the pool at the
ceiling, against 4 generation orders today; more as the auction lowers the price. The audit fleet limits
(`FLEET_MAX_LIVE_EXECUTORS`, daily spend) stay the binding limit, and they are raised when paid demand calls
for it.

## 6. Changes

| Piece | Change |
|---|---|
| Admin order creation | cap sized from the estimate, `cap_max` kept (`RELIQUARY_ORDER_CAP_SIZING=on`) |
| Admin | resize loop every 5 minutes through the existing cap write (`set_task_cap`) |
| Dispatcher | audited completion and prompt tokens per second per model |
| Settler | `recent_paid`, `open_value` (the public `open_periods` view is unchanged) |
| Control status | `at`; per model token rates; per job `sizing` |
| Registry, settlement rules, weights | unchanged |

## 7. Open questions

- `HEADROOM`, `START_CAP`, `MIN_ELAPSED_SHARE`: check them on the resize log (how often a period settles
  with its cap binding).
- An order whose miners outrun the estimate between two resizes is paid under its price for that period
  (the cap binds). The headroom is meant to cover it; the resize log shows how often it binds.
