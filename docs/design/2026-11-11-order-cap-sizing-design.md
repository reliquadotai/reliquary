# Order caps sized to what an order can pay

Status: proposal. Applies to order jobs (Batch and Evals). Builds on the order price auction
(`2026-11-10-order-price-auction-design.md`).

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
tokens/period  <=  items/s per card x PERIOD_SECONDS x tokens per item / audit share x cards
spend/period   <=  ceiling (alpha per Mtoken) x tokens/period / 1e6
share          =   spend / miner_alpha_per_period
```

- **items/s per card**: the dispatcher already measures it per model (`items_per_second`, #476), over the
  time a card holds a lease. Unknown for a model never audited: a default per parameter count, set
  conservatively high.
- **tokens per item**: the mean completion of the model's recent orders, else `max_new_tokens`.
- **audit share**: the job's `audit_q` (Evals 1.0, Batch 0.2). Records that pass unaudited are paid without
  using the card, so a Batch card lets about five times its own throughput be paid. Using `q` and not the
  measured share (which also counts probation audits) overestimates, which is the safe side.
- **cards**: the order's `audit_executors`.

The price auction only lowers the price under the ceiling, so the real spend stays under this bound.

## 3. Cap sizing

- **At creation** the admin sets `cap = min(ORDER_CAP_MAX, estimate x HEADROOM)`, with `ORDER_CAP_MAX` the
  current 0.25 and `HEADROOM` 1.5. The registry guard is unchanged: a new order waits only when the sized
  caps no longer fit.
- **Every period** the admin resizes each paying order job:
  `cap = clamp(max(paid share of the last periods, estimate) x HEADROOM, CAP_FLOOR, ORDER_CAP_MAX)`.
  - The paid share comes from the settler (`pay.last_period.paid_pool_share`); its max over the last 3
    periods smooths a quiet period.
  - The estimate is recomputed with the current measured rate and card count, so a card added mid-order
    raises the cap before the work it allows is paid.
  - Lowering always applies. Raising applies while the registry guard accepts it; an order that cannot be
    raised keeps its cap and is logged.
- The settler pays a period with the cap it holds when it settles that period, so a cap lowered mid-period
  also bounds the work already done in it. The resize therefore runs right after a period settles, and
  never lowers a cap under what the open period has accrued so far x `HEADROOM`. Weights stay bounded by
  `pay_ceiling` (the cap, or the `tail_cap` a lowered task was paid at).

## 4. When the pool is really full

- **Admission:** the platform already holds an order refused by the cap guard in `waiting_for_cap` and
  retries. With sized caps this happens only when the orders running can pay close to the whole pool.
- **Backstop:** the weight combination already clamps each task to its cap and rescales every hotkey
  proportionally when the combined total exceeds 1.0 (`WeightOnlyValidator`, logged). Sized caps keep that
  from happening; it stays as the last guard.

## 5. Order of magnitude

At the 0.58 alpha per million ceiling, one Batch card (audit share 0.2) lets about 12.5M tokens a period be
paid: about 7 alpha, 5 % of the pool, cap 0.075 with headroom. About 13 such cards fit in the pool at the
ceiling, against 4 generation orders today; more as the auction lowers the price. The audit fleet limits
(`FLEET_MAX_LIVE_EXECUTORS`, daily spend) stay the binding limit, and they are raised when paid demand calls
for it.

## 6. Changes

| Piece | Change |
|---|---|
| Admin order creation | cap from the estimate instead of `DATASET_JOB_CAP` / `EVAL_JOB_CAP` |
| Admin | a resize loop once per period, through the existing cap write (`set_cap`) |
| Control status | already exposes `items_per_second` per model; add the mean completion tokens per model |
| Registry, settlement, weights | unchanged |

## 7. Open questions

- `HEADROOM`, `CAP_FLOOR` and the default rate for an unmeasured model: check them on the first orders.
- An order whose miners outrun the estimate between two resizes is paid under its price for that period
  (the cap binds). The headroom is meant to cover it; the resize log shows how often it binds.
