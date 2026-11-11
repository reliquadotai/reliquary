# Order caps sized to what an order can pay

Status: coded (`admin/cap_sizing.py`), off unless the admin sets `RELIQUARY_ORDER_CAP_SIZING=on`. Applies
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

- **At creation**, with the switch on and a price ceiling, the admin sets
  `cap = floor = min(cap asked, max(CAP_FLOOR, estimate x HEADROOM))` and keeps the asked cap as `cap_max`.
  `HEADROOM` is 1.5, `CAP_FLOOR` 0.002. A replay of the order compares the asked cap, so it is not taken for
  another contract. The registry guard is unchanged: a new order waits only when the sized caps no longer fit.
- **Every 5 minutes** the admin resizes each active task with a `cap_max`, from the control status (skipped
  when older than 10 minutes):
  `cap = min(cap_max, max(CAP_FLOOR, max(estimate, paid in the last 3 periods, accrued in an open period) x HEADROOM))`.
  - The control publishes per job (`jobs.<id>.sizing`): its model, cards, what the settler paid in the last 3
    periods (`recent_paid`) and the most an unsettled period has accrued at the ceiling (`open_value`).
  - The estimate is recomputed with the current rate and card count, so a card added mid-order raises the cap.
  - A change under 10 % of the current cap is not written. A raise the registry guard refuses keeps the cap
    and is logged.
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

## 5. Order of magnitude

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

- `HEADROOM`, `CAP_FLOOR` and the default rate for an unmeasured model: check them on the first orders.
- An order whose miners outrun the estimate between two resizes is paid under its price for that period
  (the cap binds). The headroom is meant to cover it; the resize log shows how often it binds.
