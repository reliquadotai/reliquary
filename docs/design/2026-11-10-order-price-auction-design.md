# Order price auction: cheapest first, paced by the audit

Status: draft for review. Applies to order jobs (Batch and Evals), which share one path: forced-draw jobs, light
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

The validator keeps a budget `B` of samples in flight: leased to a miner and not yet delivered, plus delivered and
waiting for the audit. It is counted in audit work, so a Batch sample weighs `audit_q` (0.2) and an Evals sample 1.

`B` is a thermostat on the audit queue, not a speed estimate:

- the audit queue (`audit_backlog`, records due and not yet judged) holds less than `T_low` of audit work: grow `B`;
- it holds more than `T_high`: shrink `B`;
- in between: hold.

`T_low` / `T_high` are set in minutes of audit work at the measured audit rate (default about 5 and 15 minutes),
so the card always has work queued and the queue stays short. `B` starts from the job's first minutes and adjusts
every few seconds. More audit cards raise the audit rate, so `B` grows by itself.

### 2.3 Allowances, cheapest first

Live offers are sorted by price. Walking that order, each hotkey is given an allowance up to its capacity until the
allowances cover `B`. The last hotkey reached gets the remainder. A hotkey past `B` gets 0 and is told the current
price, so it can lower its offer.

Allowances are recomputed whenever `B` or the offers change. A hotkey whose allowance drops keeps the leases it
holds and gets no new one until it is back under. Only hotkeys that asked for work in the last minutes count, so an
idle offer does not hold budget.

### 2.4 Leases in batches

A lease request asks for up to `k` samples in one signed call; it gets `min(k, allowance - held)` free slots.
Release and submission accept several samples per call. The miner decides `k`, its batch size and its engine.

A lease's lifetime follows the hotkey's measured delivery time (twice its recent median, between a floor and the
current worst-case bound), not one worst case for everyone. A missing or slow miner then blocks the end of an order
for minutes, not hours. The second-lessee rule for the job's tail stays.

### 2.5 Price paid

At the end of each pay period (the existing period clock), every verified token delivered in the period is paid
the same price: the highest offer that was given a non-zero allowance during the period. Each verdict records the
period price; the settlement pays `tokens x price` under the job's cap, and the rest of the ceiling budget burns.

Paying everyone the marginal price, not their own offer, lets a miner offer its real cost.

### 2.6 Leases not delivered

A lease that expires or is released without a submission counts against the hotkey: its price times the sample's
expected tokens is taken from its next pay on the job, and its allowance is halved for a lease lifetime. A hotkey
with an unpaid charge gets no new lease. Charges and offers are persisted with the job's miner state, so a restart
keeps them.

## 3. What exists and what changes

| Piece | Today | Change |
|---|---|---|
| Light leases, tail second lessee | `corpus/draw_leases.py` | batch grant/release, lifetime per hotkey |
| Offer route and miner client | `CorpusAskRequest`, `--ask-alpha-per-mtoken` | add capacity |
| Audit queue reading | `CorpusAuditor.audit_backlog` | weight by `audit_q`, feed the thermostat |
| Per-lease price in verdicts | `lease_price`, `lease_debit` | one price per period |
| Settlement at a lease price | `corpus_settlement.py` | unchanged |
| Rounds, target stepping | `corpus/token_auction.py` | replaced by 2.2 and 2.3 |
| Debts in memory | `token_auction.py` | persisted (2.6) |
| Order jobs priced by auction | CLI `tasks set-cap --pricing auction` only | the admin sets it on every order job |

The miner client keeps working: it requests one sample at a time, which is a batch of one.

## 4. Open questions

- Thermostat defaults (`T_low`, `T_high`, step sizes): pick them on a replay of the audit logs of past orders.
- Whether the period price should ignore an offer that served only a tiny share of the period's tokens.
- Offers per job or per pinned model across jobs.
- Rollout: announce the offer flag to miners before the first order priced this way; jobs without the flag keep
  first-come at the ceiling.
