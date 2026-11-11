# Audit capacity that survives bad cards

Status: proposal. Applies to order jobs (Batch and Evals). Builds on the order price auction
(`2026-11-10-order-price-auction-design.md`) and the cap sizing (`2026-11-11-order-cap-sizing-design.md`).

## 1. Problem

The audit is the order's critical path: the auction paces miners by it, the customer's order waits for it, and
every card hour is our cost. Today each order rents its own card on Lium, a market of heterogeneous hosts.
What 2026-11-11 showed:

- **A card lost DNS** (`Temporary failure in name resolution` for `api.reliqua.ai`, about one call in two). It
  audited batches but could not post the results; the lease expired, the batch went back to it, and the order's
  judge pass waited on that batch for 23 minutes. The control set it aside only after 3 expired leases in a row
  (`LEASE_EXPIRY_STRIKES` x `AUDIT_LEASE_SECONDS` 300 s = 15 min at best), and each result it did deliver reset
  the count.
- **Two cards never started** (`died_starting`) the same day; each cost a 15-minute wait before the next rent.
- **A control restart stopped auditing for about 9 minutes** while orders were already open to miners: records
  piled up and the auction's budget fell to its floor.
- **Cards idle most of the time** (6-8 % GPU over 2 minutes) while the budget was held down, then 96-100 % once
  work reached them: a card is far from the limit when it gets work steadily.

## 2. Principles

1. Any card can fail at any time. A failure costs seconds of audit, not an order.
2. The audit never waits on one card or one batch.
3. Capacity is a pool, not a card per order: a base we control, and a market for the peaks.

## 3. Design

### 3.1 A card proves itself before it gets work

The platform marks a card `ready` only after a **probe**: the executor resolves and reaches the control, loads
the model, and returns the result of a small canary batch whose answer the control knows. A card that fails the
probe is returned at once and the next one rented.

### 3.2 A card that loses results is set aside within a minute

- The dispatcher counts **undelivered results** (a lease that expires, or a result post the executor reports as
  failed) per executor. Two in a row set it aside (`quarantined`); the platform already replaces a quarantined
  card (`executor_quarantined`).
- A delivered result resets the count only after a clean streak (e.g. 5), so a card failing one call in two
  does not keep its work.
- Audit lease time follows the measured batch time (as miners' lease time does in the auction: twice the recent
  median, within a floor and a ceiling), so an expired lease is seen in about a minute, not five.

### 3.3 The audit never waits on one batch

A judge pass hands its batches to the dispatcher and moves on: a batch that comes back expired goes to another
card, and its records are decided in a later pass. One stuck batch delays its own records, never the job.

### 3.4 Executors ride out network blips

- The executor resolves the control's address once, keeps it, and falls back to a second resolver.
- A computed result that cannot be posted is kept on disk and posted again when the network returns, for as long
  as its lease lives, instead of being dropped after 4 attempts.

### 3.5 A restart does not stop the audit

- An order's auditor starts before the miner-count and settlement backfills, which run behind it.
- An order is not released to miners until its auditor has seeded (the existing hold, extended to it).

### 3.6 Base pool and market burst

- **Base:** one or two GPUs reserved with a reliable provider, always on, each serving several models (the
  multi-model executor of 2026-11-05: a model pool in the executor, the fleet packing orders onto loaded cards).
  Every order's audit goes there first.
- **Burst:** when the base's audit queue stays above its mark, the fleet rents market cards (Lium) for the
  models in demand, and returns them when the queue drains.
- A base card, busy most of the day, costs less per audited token than a market card rented per order and
  busy a tenth of its time. The provider and the size of the base are decided on measured prices and load.

### 3.7 Alerts

An alert (Discord or e-mail) when: a judge pass has not finished for 5 minutes; a card is set aside; an open
order has had no miner for 10 minutes; a card fails its probe twice for one model.

## 4. Order of work

| Step | What | Where | Size |
|---|---|---|---|
| 1 | 3.2 set aside on lost results, adaptive audit lease | control | 0.5 day |
| 2 | 3.3 a pass never waits on one batch | control | 0.5-1 day |
| 3 | 3.4 executor DNS cache and kept results | executor | 0.5 day |
| 4 | 3.5 restart without an audit gap | control | 0.5-1 day |
| 5 | 3.1 probe before `ready` | executor + platform | 1 day |
| 6 | 3.7 alerts | control + platform | 0.5 day |
| 7 | 3.6 base pool: measure, choose a provider, multi-model executor, fleet packing | all | 1.5-2 weeks |

Steps 1-6 make any provider safe to use; step 7 changes what we rent.

## 5. Open questions

- Base provider and size: monthly prices for H100/H200 at two or three providers, against measured audit load.
- How many models a base card holds at once (memory), and how long a model swap takes.
- Whether the base serves Evals (`audit_q` 1, every sample audited) first, since they load it most.
