# Order price auction: cheapest first, paced by the audit

Status: coded (`corpus/token_auction.py`), off unless the admin sets `RELIQUARY_ORDER_PRICING=auction`. Applies to order jobs (Batch and Evals), which share one path: forced-draw jobs, light
leases, audit at `audit_q` (Batch 0.2, Evals 1.0). Supersedes the round mechanism of
`2026-11-03-token-price-auction-design.md` for orders. RL is out of scope here; the same shape fits it, paced by the
trainer instead of the audit.

## 1. Goal

- The cheapest GPUs do the work. Price is the only thing that decides who works.
- An order moves as fast as our audit can verify it, no faster, no slower. The audit card never idles while work
  is left, and never builds a long queue.
- The validator never limits how a miner works: any batch size, any engine, any number of samples in flight within
  its allowance.

Today an order hands a free slot to whoever asks first (up to 64 live leases and 25 % of the job per hotkey), at
the token-price ceiling, with nothing tying the hand-out rate to the audit.

## 2. Mechanism

### 2.1 Offers

Each hotkey posts one offer per job: a price (micro-alpha per million verified tokens, at most the job's ceiling)
and a capacity (samples it can have in flight at once). An offer lives one hour unless reposted; reposting changes
it at once. Offers are private; the job's current price is public.

The capacity is taken as declared: the validator does not measure it. A hotkey that declares more than it delivers
pays for it (2.6).

### 2.2 How much work is in flight

The validator keeps a budget `B`: the most leases out at once across the job. `B` is a thermostat on the audit
queue (`audit_backlog` plus the batches sent to executors), not a speed estimate. Every 30 s:

- the queue holds fewer than `auction_backlog_low` records (default 8) and at least 80 % of `B` is leased: `B`
  grows by 25 % (at least 1);
- it holds more than `auction_backlog_high` (default 32): `B` shrinks by 20 %;
- otherwise it holds.

`B` starts at `auction_initial_leases` (default 64). A record waiting out its hold is not in the queue, and an
unaudited pass leaves it at once, so a Batch job at `audit_q` 0.2 runs about five times the leases of an Evals job
on the same card without any weighting. More audit cards drain the queue faster, so `B` grows by itself.

### 2.3 Allowances, cheapest first

Live offers are sorted by price. Walking that order, each hotkey is given an allowance up to its capacity until the
allowances cover `B`. The last hotkey reached gets the remainder. A hotkey past `B` gets 0 and is told the current
price, so it can lower its offer.

Allowances are recomputed at each lease request. A lease is granted while the hotkey holds less than its
allowance and the job holds less than `B`: a dearer hotkey's leases from before keep their place until delivered,
then go to the cheaper one. Only hotkeys that asked for a lease in the last 3 minutes, or hold one, count, so an
idle offer does not hold budget.

### 2.4 Leases in batches

A lease request asks for up to `count` samples (at most 256) in one signed call and gets as many as its allowance
and the budget allow. The miner decides `count`, its batch size and its engine. Releases and submissions stay one
per call.

A lease's lifetime follows the hotkey's measured delivery time (twice its recent median, between a floor and the
current worst-case bound), not one worst case for everyone. A missing or slow miner then blocks the end of an order
for minutes, not hours. The second-lessee rule for the job's tail stays.

### 2.5 Price paid

A lease is paid the price of the moment it is granted: the dearest offer getting work at that instant (the
auction's marginal price), never under the holder's own offer. Everyone working at the same moment is paid the same
price; the settlement pays `tokens x lease price - debit` under the job's cap, and the rest of the ceiling budget
burns.

Why the moment and not the period: the budget follows the audit queue, so within one period it swings between a
few cheap offers and the whole ladder. Priced once per period (the lease price under which 95 % of its tokens were
delivered, from #498), an offer at 0 took the leases of a small budget and was paid the period's price, set by the
dear offers needed at other moments. On 2026-11-12 one coldkey offering 0 delivered 26 % of a 27B order's tokens and
was paid the ceiling, and on another order took 97 of a 100-lease budget. With the price of the moment an offer
under its real cost only wins work at moments that pay under that cost; an offer at its real cost gets the same
leases at the same price every other time. Paying each its own offer instead would make miners bid the price they
guess, not their cost.

The highest lease alone set the period's price at first: on 2026-11-11, offers serving 0.1-1 % of a period's tokens
priced all of them at the ceiling. Leases received before `RELIQUARY_AUCTION_MOMENT_PRICE_FROM` keep the period
rule (the 95 % price, or their own offer when higher).

### 2.6 Leases not delivered

A lease that expires or is released without a submission is charged: its price times the job's mean submission
size (by the share of its time held on a release) is taken from the hotkey's next accepted submissions on the job.
An expiry, or a release past half its time, halves the hotkey's capacity for an hour. A hotkey with an unpaid charge
holds one lease at most until it has paid. Offers, charges, penalties, the budget and delivery times are kept in
`$RELIQUARY_AUCTION_STATE_DIR/<job>.json`, so a restart keeps them.

## 3. What exists and what changes

| Piece | Today | Change |
|---|---|---|
| Light leases, tail second lessee | `corpus/draw_leases.py` | batch grant/release, lifetime per hotkey |
| Offer route and miner client | `CorpusAskRequest`, `--ask-alpha-per-mtoken` | add capacity |
| Audit queue reading | `CorpusAuditor.audit_backlog` | feeds the thermostat |
| Per-lease price in verdicts | `lease_price`, `lease_debit` | the holder's offer; one price per period |
| Settlement at a lease price | `corpus_settlement.py` | unchanged |
| Rounds, target stepping | `corpus/token_auction.py` | replaced by 2.2 and 2.3 |
| Debts in memory | `token_auction.py` | persisted (2.6) |
| Order jobs priced by auction | CLI `tasks set-cap --pricing auction` only | admin `RELIQUARY_ORDER_PRICING=auction` |

The reference miner keeps working: it posts its offer with `--ask-alpha-per-mtoken` and `--ask-capacity` and
requests one sample at a time.

## 4. Open questions

- Thermostat defaults (backlog marks, step sizes): check them on the first orders priced this way.
- Prompt tokens are not paid on a job priced by auction (as before): an offer prices output tokens only.
- Offers per job or per pinned model across jobs.
- Rollout: announce the offer flag to miners before the first order priced this way; jobs without the flag keep
  first-come at the ceiling.
