# Lease allocation: fill the cheapest miners, at the consumer's pace

Status: draft for review (2026-11-17). Covers every leased task: batch and eval orders (`token_auction`,
`draw_leases`) and async RL (`rl_lease_quota`).

## 1. Goal

Keep exactly as many leases open as the consumer (the audit card for orders, the trainer for RL) can absorb, give
them to the cheapest miners first, and keep those miners full. Ten full miners beat a hundred half-full ones.

Properties wanted:

- the price decides who works, never the moment a miner asks;
- a miner running async vLLM can stay at its batch, forever, while it stays among the needed;
- only one miner is partly served: the last one needed (the marginal);
- holding leases you do not serve costs you; bidding 0 buys priority for your real capacity, not volume;
- a cheaper newcomer can always enter, and pushes the dearest needed miner out gradually, without lost work.

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

### 3.3 Each miner: batch B_i and speed v_i

- `B_i`: the batch a miner declares (new field in its offer). Never exceeded. A miner without one is treated as
  today (its capacity).
- `v_i`: tokens per second per lease, measured: tokens delivered / lease-seconds held, EWMA. A newcomer starts at
  the market's median.
- Ramp: a newcomer's allowance starts at 4-8 and doubles per round delivered on time, up to `B_i` (the RL
  quota's slow start, reused).

### 3.4 Allocation: merit order, one partial miner

Every 30 s: walk the offers cheapest first; each miner takes `n_i = B_i` (or its ramp) and adds `n_i x v_i`, until
`R*` is reached. The last one takes `ceil(missing / v_last)`; everyone after gets 0. The total leases open is
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

### 3.6 Lease time aligned with the allocation

`lease_seconds = longest sample / v_min + margin`, with `v_min` the same per-lease speed the allocation trusts
(e.g. the 25th percentile of honest miners on that model; episodes keep their deadline floor). A miner that
declares a batch its card cannot serve at `v_min` sees its leases expire: debit (as today) and `n_i` cut.
Holding leases aside becomes a loss; holding your real batch costs nothing.

### 3.7 The queue as a correction (k), not as the controller

The queue alone acts 10-20 min late on long generations (the leases already out keep delivering). It corrects
the estimate instead:

- wait = queue tokens / A (seconds), target ~60 s per model;
- busy = card busy share over 5 min;
- lower k in proportion to the excess when wait > target and (wait rising or busy > 95 %);
- raise k when busy < 85 %, wait < target / 2 and some miner is refused for want of rate;
- dead band in between.

### 3.8 Price (unchanged in spirit)

Uniform price: every lease is paid `max(own offer, clearing price)`. Bidding 0 buys priority for your real capacity
(3.6 bounds it), not volume, and many 0 bids drive the clearing price under their own cost: self-correcting.
Paying each miner its own offer would start the guessing war we want to avoid. The clearing price could become the
marginal miner's offer (known exactly once 3.4 runs) instead of #542's 80 % quantile of deliveries (open 8.2).

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

1. Small PR: queue measured in audit seconds with trend and dead band, replacing the 32/8 records thresholds.
2. Measure A and `v_i` and log them (no behaviour change); show fill ratios.
3. Allocator (3.4-3.6) for batch and eval behind a flag, one model first.
4. Swap in the miner protocol (submit returns the next lease): miner release, backward compatible.
5. RL: R* from the trainer and merit order over the existing quota.

## 7. Not changing

The draw audit and its judge, debits on leases not delivered, the ceiling per token, the customer price grid.

## 8. Open questions

1. Several orders on one model share one audit card: split R* by remaining volume, or by deadline / priority?
2. Clearing price: the marginal miner's offer, or the first excluded offer (more truthful, more volatile)?
3. `v_min`: per model from measurements (25th percentile?), and its floor for a model with no history.
4. Prefetch over `n_i`: 1, 2, or a share of `B_i`?
5. Declaring `B_i`: a new offer field; miners that do not declare keep today's capacity semantics.
