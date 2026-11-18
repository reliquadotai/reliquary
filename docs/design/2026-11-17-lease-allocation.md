# Lease allocation: fill the cheapest miners, at the consumer's pace

Status: approved 2026-11-18, being implemented. Covers every leased task: batch and eval orders (`token_auction`,
`draw_leases`) and async RL (`rl_lease_quota`).

## 1. Goal

Keep exactly as many leases open as the consumer (the audit card for orders, the trainer for RL) can absorb, give
them to the cheapest miners first, and keep those miners full. Ten full miners beat a hundred half-full ones.

Properties wanted:

- the price decides who works, never the moment a miner asks;
- a miner running async vLLM can stay at its batch, forever, while it stays among the needed;
- only one miner is partly served: the last one needed (the marginal);
- holding leases you do not serve costs you; bidding 0 buys priority for your real capacity, not volume;
- a cheaper newcomer can always enter, and pushes the dearest needed miner out gradually, without lost work;
- an idle GPU that shows up gets its whole batch at once: no slow start;
- the price depends on the offers and on R* only, never on who delivered how much.

## 2. What happens today

Orders (batch, eval), `reliquary/corpus/token_auction.py`:

- Allowances: offers sorted by price, each *active* hotkey (asked in the last 180 s, or holding) gets its
  declared capacity until the budget is covered.
- A lease is granted if the hotkey is under its allowance **and** the total leased is under the budget. When the
  budget is full, a freed lease goes to whoever asks first: a dear miner can win it by timing.
- The budget steers on the audit queue every 30 s: over 32 records waiting x0.8, under 8 (and 80 % used) x1.25.
  32 records is a fixed count whatever the card's speed and the records' length.
- Lease time: 300 s + longest sample / 10 tok/s (`draw_leases.lease_seconds`): about 1 h for 32k tokens.

RL, `reliquary/validator/rl_lease_quota.py`: per-hotkey quota measured, never declared: slow start (x2 per
saturated round), Little's law target (throughput x latency x 1.5), cut on expiry. No price, no global rate.

Seen on 2026-11-17:

- 27B batch orders (`6309dffa`, `6f59f675`, one H100 audit card, ~100 miners): queue spikes 57-123 records,
  budgets 851 -> 544 and 550 -> 281 within a minute, 349 leases out for a 281 budget: even the cheapest miner is
  refused (`allowance_full`) until dearer leases drain. Miner report: budget 260 -> 56 at a 3k tok/s peak.
- MoE eval `03628c19` (Qwen3.6-35B-A3B, 17 M tokens): ~920 tok/s average over 4.7 h of work. The auction offered
  300-470 leases; miners declared a capacity of 3 per hotkey and ran ~30 tok/s per stream. The tail (last 5 %)
  waited on 1 h leases left to lapse (one hotkey: 240+ lapsed leases in 15 min).

Seen on 2026-11-18 (9B batch `14401f87`, 68 offers, 1 835 declared slots, budget 1 816):

- Three hotkeys offering 1e-6 alpha/M (328 slots) delivered 72-78 % of the volume. The clearing price is the offer
  under which 80 % of the deliveries were won (#542), so they sit just under 80 % and are paid what the dearer
  miners set. Price of the moment per 10 min: 0.0025, 0.036, 0.042, 0.0014, 0.0079, **0.0000**, 0.022, 0.12
  alpha/M (ceiling 0.706); within one second at 06:03 leases were granted at 0.14, 0.35 and 0.706.
- Any volume-based price has this lever (moving 80 % moves the target), and several hotkeys make it worse: each
  dear hotkey that delivers pulls the quantile up.

## 3. Design

### 3.1 The target rate R*

One number per model's market, in tokens per second (prompt included: the audit reads it):

```
R* = min(audit_rate / q, trainer_rate, needed_rate) x k
```

- `audit_rate` (A): tokens the audit card verifies per busy second, measured (3.2). `q`: audited share (0.2 by
  default for orders, 1 for evals that audit all).
- `trainer_rate`: RL only, tokens per second the trainer consumes.
- `needed_rate`: what the open orders still need (remaining tokens over their deadline); no point opening more.
- `k`: slow correction around 1 (3.7).

### 3.2 Measuring the audit rate A

- From the judge passes already logged: tokens audited / busy seconds, EWMA over ~5 min, per audit executor.
- Start value per (model, GPU) from a table of benches (27B H100, 27B H200, 35B-A3B H200 4.9k tok/s, ...).
- One executor serves every order on its model: A is shared across those orders (open question 8.1).

### 3.3 Each miner: capacity B_i and speed v_i

- `B_i`: the batch a miner declares (its offer's capacity). Never exceeded.
- **No slow start.** A miner gets its whole `B_i` as soon as it is among the needed, counted at the order's median
  speed until measured. An idle GPU that shows up is at its batch from its first minute.
- **First verdict.** After one to two median sample times (1-2 min for a 9B at 2k tokens, 5-10 min for 32k
  thinking) and a minimum of deliveries, its delivered rate is compared to `B_i x median`. Well under: its
  allowance drops straight to what it delivers (`rate_i / median`), not by steps.
- **Then:** grows (+1-2 slots per on-time delivery, up to `B_i`) when full and over `v_min`; loses 20 % per step
  under `v_min`.
- **Speed, two counters, no per-lease tracking:**

  ```
  v_i = output tokens accepted over 10 min / (mean leases held over those 10 min x 600 s)
  ```

  Every held lease counts, delivered or not: a hotkey sitting on leases sees `v_i` drop. Output tokens only (the
  prompt is nearly free and would favour long-prompt tasks). Episodes (agentic) are measured in episodes finished
  on time / episodes held instead.
- **v_min** is relative to the order, so it adapts to the task (short answers, 32k thinking, ...):
  - bulk of the order: 1/4 of the order's median miner speed. It only cuts leases held without being worked; a
    modest card (RTX vs H200, x2-3) stays served: it only counts for what it delivers (3.4);
  - tail (last 10 %, or fewer free prompts than slots asked): 1/2 of the median, and the free prompts go to the
    fastest miners first;
  - no absolute floor (an agentic order counts its tool time); no verdict before a minimum of deliveries.
- **Over-declaring** buys nothing: the merit order counts measured rate (3.4), the first verdict cuts within
  minutes, and undelivered leases stay debited. A brand-new hotkey with nothing to debit can disturb the
  others for a few minutes at most, at the cost of a registration; cuts go through refused swaps (3.5), so the
  miners it displaces lose renewals during that window, not their batch.

### 3.4 Allocation: merit order, one partial miner

Every 30 s: walk the offers cheapest first; each miner takes `n_i = B_i` (or its measured allowance, 3.3) and adds
`n_i x v_i`, measured rate, until `R*` is reached. The last one takes `ceil(missing / v_last)`; everyone after gets 0. The total leases open is
`N* = sum(n_i)`: Little's law miner by miner, so fast and slow cards mix without special rules.

Example, A = 5k tok/s, q = 0.2, R* = 25k tok/s:

| miner | B | v | rate | cumulative | n |
|---|---|---|---|---|---|
| M1 | 100 | 60 | 6 000 | 6 000 | 100 |
| M2 | 200 | 40 | 8 000 | 14 000 | 200 |
| M3 | 50 | 30 | 1 500 | 15 500 | 50 |
| M4 | 300 | 40 | 12 000 | 27 500 | **238** (marginal) |
| M5 | 80 | 50 | | | 0 |

### 3.5 Admission and the swap

- One rule: a lease is granted iff the hotkey holds fewer than its `n_i`. No shared pool, no first-come race.
- Swap: `submit` returns the next lease in the same call (when `held < n_i` after the submission), so an async
  miner never has an empty slot. Plus a prefetch of 1-2 leases over `n_i`.
- Cutting (R* falls, or a cheaper miner enters before the marginal): the marginal's `n_i` drops; its swaps are
  refused until it holds `n_i`. It loses one lease per finished request, never work in flight. At 0, the next one
  up becomes the marginal.
- Leases out over `N*` after a cut drain within one lease time; that overshoot is what `k` watches.

### 3.6 Lease time: a backstop, the speed does the work

Two layers:

- **Deadline (backstop for crashes):** `longest sample / v_floor + margin`, `v_floor` ~10 tok/s (today's rule);
  episodes keep their deadline floor. It only catches a miner that died.
- **Measured speed (3.3)** finds the real batch: a hotkey that holds more than it serves falls under `v_min` and is
  cut within minutes, long before its leases lapse.
- In the tail the second lessee (past half a lease) and the fastest-first rule keep the last prompts moving.

### 3.7 The queue as a correction (k), not as the controller

The queue alone acts 10-20 min late on long generations (the leases already out keep delivering). It corrects
the estimate instead:

- wait = queue tokens / A (seconds), target ~60 s per model;
- busy = card busy share over 5 min;
- lower k in proportion to the excess when wait > target and (wait rising or busy > 95 %);
- raise k when busy < 85 %, wait < target / 2 and some miner is refused for want of rate;
- dead band in between.

### 3.8 Price: the last needed miner's offer

Every lease is paid `max(own offer, clearing price)`, the clearing price being **the offer of the last miner the
merit order needs to reach R*** (the marginal, partly served). Not the first excluded: an excluded hotkey delivers
nothing, so a hotkey offering the ceiling just past R* would set every price for free.

Why it closes the 2026-11-18 lever:

- delivered volume is not an input: staying "just under 80 %" means nothing;
- **adding an offer never raises the price.** 19 hotkeys at 0 and a 20th dear one: either the dear one is not
  needed (price set by a competitor) or it is the marginal, and without it the marginal would be the next offer
  up, dearer still. Several hotkeys gain nothing;
- bidding 0 means "I take the market's price": safe for a small miner, risky for a big one (if the bids at 0 cover
  R*, everyone is paid 0), so big miners bid their cost.

What it does not close: **withholding** (declaring less capacity so the marginal is a dearer offer). It costs real
volume and pays only for a big miner, with a large price step right after the marginal, knowing where R* is. Any
competitor offering in the step cancels it; the book and R* are not published (only the clearing price of past
periods). No price floor: idle GPUs at any price are welcome, and an oversupplied market is allowed to be cheap.

Before R* bounds the merit order, the marginal offer would be the ceiling (the budget admits almost everyone):
that is why #542 used a volume quantile, and why the clearing price switches only together with the allocator.

### 3.9 Prompt supply

Miners stay full only if there is work: keep enough prompts open, hand out the longest first (shorter tail),
keep the existing second lessee on a slot past half its lease, open the next order on a model before the previous
one empties.

### 3.10 Per task type

| | consumer | R* bound | lease time | price |
|---|---|---|---|---|
| batch | audit card | A / q (q = 0.2) | max tokens / v_min | auction (3.8) |
| eval | audit card | A / q (q = 1 today) | max tokens / v_min; episodes: deadline | auction |
| RL | trainer (+ audit) | trainer rate | episode / rollout deadline | RL pay; merit order replaces "anyone up to quota" |

RL keeps its measured quota (`rl_lease_quota`) as the per-miner `n_i` cap; this design adds the global R* and the
merit order on top.

### 3.11 Several orders on one model

One audit card (and one market) serves every order on its model. R* is split in two layers:

1. **Floor:** each order gets `remaining tokens / time left to its deadline` (batch: the deadline the order was
   sold with, 24 h by default; eval: shorter).
2. **Surplus** goes to the order closest to done (least remaining volume): it minimises the mean wait, a small
   eval comes out in minutes, the big batch still meets its deadline thanks to its floor.

Miners see nothing of it: same model, same market, same price; only which order a lease's prompt comes from
changes. When the floors exceed R*, the card is too small for its orders: an alert asks for a second audit card.

## 4. Observability

- Per miner: `n_i`, held, fill ratio (held / n_i), `v_i`, and why it is not full (rate, ramp, no prompts),
  served to the miner on its auction view.
- Per market: R*, A, k, wait, busy, delivered rate vs R*.
- Alert: a needed miner under 90 % fill for 10 min.

## 5. Simulation before code

Replay the 2026-11-17 logs (27B batch `6309dffa`/`6f59f675`, MoE eval `03628c19`) through today's controller and
this one, with miner profiles (fast cheap H200, RTX at 0, dear reliable, hoarder, batch-sync vs async). Compare:
tokens/s, emissions spent per token, fill of the needed miners, time to the last 5 %.

## 6. Rollout

1. Measure A (audit tokens per busy second) and `v_i`; the queue in audit seconds with trend and dead band
   replaces the 32/8 record thresholds (they share the measurement). Fixes the collapsing throughput now.
2. Simulation on the 11-17 / 11-18 logs: `v_min` fractions, first-verdict delay, R*.
3. Allocator: merit order in measured rate, clearing price = marginal offer, capacity declared then measured,
   multi-order split; behind a flag, the 9B market first (where the price lever is used).
4. Swap in the miner protocol (submit returns the next lease): miner release, backward compatible.
5. RL: R* from the trainer and merit order over the existing quota; `v_min` also closes the hoarding hole of the
   Little target (it returns the leases held, so it never sees hoarding, only slowness).

## 7. Not changing

The draw audit and its judge, debits on leases not delivered, the ceiling per token, the customer price grid.

## 8. Decided (2026-11-18)

1. Several orders on one card: floor by deadline, surplus to the order closest to done (3.11).
2. Clearing price: the marginal (last needed) miner's offer (3.8).
3. `v_min`: 1/4 of the order's median in the bulk, 1/2 in the tail (3.3); the simulation
   tunes the fractions.
4. No slow start, no price floor.

## 9. Open questions

1. Prefetch over `n_i`: 1, 2, or a share of `B_i`?
2. Deadlines per order type (batch 24 h?) when the order does not carry one.
