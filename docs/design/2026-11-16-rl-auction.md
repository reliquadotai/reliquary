# RL on the order auction

Status: proposal, on top of PR #540 (price of the moment by delivered volume). Extends `2026-11-10-order-price-auction-design.md` (orders, live) to the RL controller, following
§12 of `2026-11-03-token-price-auction-design.md`.

## 1. Why

An RL task still runs the pricing of teutonic-rl-async-v1:

- `rl_price.AsyncPriceController`: one multiplier per environment, steered from the trainer queue. It oscillated and
  pinned envs at the ceiling for hours.
- `rl_lease_quota.QuotaController`: AIMD cap and Little's-law target per hotkey. Five manual restores in one run.

Orders have since moved to an auction: offers (price, capacity), a lease budget `B`, leases served cheapest first,
each lease paid the price of the moment it was granted, and undelivered leases charged. RL tasks launched from the API (phase 3 of `2026-11-16-env-catalog.md`) should not ship on the old
mechanism.

## 2. What stays

Everything that is not price or quota is the teutonic run's code, unchanged:

- the lease book (`rl_leases.LeaseBook`): exclusive `(env, prompt, k)` slots, nonces from `S_k`, commit/reveal,
  cooldown, blacklist, stale refusals, validator downtime never charged, `box_failed` releases;
- the forced draw and its audit, the in-zone filter, the buffer and its lag drop, the trainer link, checkpoints;
- the pay events: a trained rollout earns its matched tokens; an out-of-zone first scan earns 25 % of an in-zone
  group's pay; a ban forfeits unsettled accruals;
- settlement per 72-min period under the task cap, the carry, the trainer outage fill, the period archives.

## 3. What changes

### 3.1 One auction per environment

Each RL environment of the task has its own `corpus.token_auction.TokenAuction`. A hotkey posts one offer per
environment: a price (micro-alpha per million matched tokens) and a capacity (groups it can have in flight). Envs
differ in cost (a SWE episode needs a sandbox, a math group does not), so one price across envs would overpay the
cheap ones.

The ceiling is the task's, per matched token, as an order's (`tasks set-cap --max-price-alpha-per-mtoken`). Each
env's market settles under it. A natural first value is `cap per period ÷ planned matched tokens per period`; a
period still pays at most the cap.

### 3.2 The budget keeps n checkpoints ahead

An order cannot know how fast its audit card will drain, so its budget `B` is a thermostat on the audit queue. RL
needs none: the trainer consumes a fixed batch, `batch_groups` groups per step, split by the env shares. The goal is
`lag_target` checkpoints of trainable groups waiting for the trainer (default 2, under the buffer's max lag), so the
trainer never waits for a group.

A lease yields a trainable group only when it comes back in zone; an out-of-zone group (0/16, 16/16) frees its place
and fills nothing. The leases out are therefore the groups missing divided by the env's yield:

> `B_env = (share_env × batch_groups × lag_target − groups of the env buffered) ÷ yield_env`

`yield_env` is measured: the share of the env's last 200 graded groups that train (in zone, not refused by the
trainer), 1/2 until 32 are graded, never under 10 % (a dry env would otherwise ask for unbounded leases). With
a third of the groups in zone and nothing buffered, an env at half of a 512-group batch keeps 1,536 leases out.

Every submitted lease frees its place at once, trained or not, so leases keep turning over: each submission unlocks
the next lease while the buffer is short of its target. Once the buffer holds its `lag_target` steps, `B_env` is
0 and no new lease goes out until the trainer takes a batch.

`B_env` replaces the price rise on trainer wait, the lane reserves and the global pending cap.

### 3.3 Grant

`LeaseBook.grant` keeps its checks and asks the env's auction instead of the quota: a request is served while the
hotkey holds less than its allowance (`TokenAuction.admit`) and the env holds less than `B_env`. A hotkey with no
live offer on the env is refused `no_ask`; one above what the env pays now, `ask_above_price`; an allowance used or
the env's budget out, `full`. The per-hotkey `HOTKEY_MAX_OPEN` cap goes: the allowance is the cap. Each lease row
stores its price and its holder's offer (pool share per matched token).

### 3.4 Pay

A lease's price is the order rule of PR #540: the higher of its holder's own offer and the price of the moment, the
offer under which 80 % of the env's leases delivered in the last 30 minutes were won (counted at their holders'
offers when granted), never above the dearest offer getting work now. One dear offer holding a single lease does not
set the price. 30 minutes is about five trainer steps.

`RlPayLedger.ingest` reads the lease's price instead of `AsyncPriceController.token_price(env)`:

- trained: `matched tokens × lease_price`;
- first scan: `0.25 × lease_price × the env's mean matched tokens per trained group`.

Settlement is unchanged: under the cap, the rest of the ceiling budget burns. `env_pay_k` is no longer needed (each
env's ceiling is its share) and is refused with `pricing = auction`.

### 3.5 Undelivered leases

A lease that expires, is forfeited, or ends on a group refused for integrity charges its holder
`lease_price × the env's mean matched tokens per trained group` (an EMA of the trained groups; 16,384 before
any), taken from its next accruals on the env (`TokenAuction.take_debit` at ingestion). A miss halves the hotkey's
capacity on the env for an hour, and a hotkey in debt holds one lease at most until it has paid. A lease that ends
for our reasons charges nothing, as the quota did: validator downtime (`VALIDATOR_UNAVAILABLE`) and sandbox
releases (`box_failed`).

### 3.6 Lease lifetime

The order rule: twice the hotkey's recent slow deliveries on the env, between the floor and the env's bound
(`lease_expiry_s`).

### 3.7 One interval, cheapest first

The lease collector resolves the requests of one ~1 s interval together. Every requester of the interval counts as
active before any request is served, so the cheaper offer is served first whatever order the requests are
resolved in.

### 3.8 Routes and miner

- `POST /rl/ask` (signed, single use, `reliquary/rl-ask/v1`): `{environment, micro_alpha_per_mtoken, capacity}`.
  Posting an offer makes the hotkey active at once.
- `GET /rl/auction[?hotkey=]`: per env, the price of the moment, the ceiling, the budget, the leases out, the groups
  buffered and the target; with `hotkey`, its allowance and debt. Never an offer or another hotkey.
- `GET /rl/quota` and the lease answers give `quota` = the hotkey's allowances over the envs (0 without an offer),
  so the reference miner's loop is unchanged. New refusals: `no_ask`, `ask_above_price`.
- `GET /rl/pay` shows each env's auction view instead of its multiplier.
- `reliquary rl mine --ask-alpha-per-mtoken 0.5 [--ask-alpha-per-mtoken swe=2] [--ask-capacity N]` posts the offers
  before leasing, every 10 minutes, and at once after a `no_ask`.

### 3.9 A training order's limits

A task created from a training order (`training_order` param: `deadline_at`, `max_steps`, `max_tokens`) stops
leasing once one is reached: the deadline passes, the latest checkpoint reaches `max_steps`, or the accrued pay
covers `max_tokens` matched tokens (a first scan counted at its share of a trained group). Every env's budget is
then 0. The leases already out may still deliver, so the overrun is at most the budget at the stop.

### 3.10 Switch

An async RL task is priced by auction with the order command:

```
reliquary tasks set-cap --task-id <task> --max-price-alpha-per-mtoken <ceiling> --pricing auction
```

plus `rl_auction_lag_target` (default 2) in its params. The registry accepts a ceiling on an async RL task only with
`pricing = auction`, and refuses `rl_env_pay_k` with it (each env is capped by the ceiling). With the auction the
lease book has no quota, no global pending cap and no lanes: the budget is the batch. `AsyncPriceController` and
`QuotaController` stay for a task without it until the legacy cleanup; no RL run is live.

## 4. Plan

1. Auction per env in the RL controller: grant, lease price, debits, budget from the batch, routes, miner flag.
   Unit tests on the lease book, the ledger and the budget.
2. Admin route creating an RL task from a training order, its environments checked against the catalog's `rl`
   rows, priced by auction (`reliquary/admin/training.py`, off unless `RELIQUARY_TRAINING_ORDERS=on`):
   `POST /admin/v1/training` `{training_id, profile_id, environments[{name, share_bps}], limits{max_steps,
   deadline_seconds, max_tokens?}, max_price_usd_per_mtoken, batch_groups, lag_target, cap?}` declares
   `order-train-<id>`, an `rl-discovered-price` async task (the async controller reads `env_split` and `rl_*`;
   nothing async consumes a service contract yet); `GET /admin/v1/training/{task}` and `POST .../retire`. The
   profile must draw by Gumbel-max. The controller and the trainer are still started by an operator per task.
3. A training run on rented GPUs: a small model, our own miners on the rented box, two single-turn envs and one
   episode env, to check the budget keeps the trainer fed without groups dying of lag and the trainer never waits for long.
4. Public environments in the catalog with `rl` rows.

## 5. Open questions

- `lag_target` default (2 checkpoints) and the yield prior (1/2): set from the test run.
- Offers per env or one offer with per-env multipliers (per env proposed).
- Whether the first-scan 25 % should move to its own auction price (kept on the lease price).
