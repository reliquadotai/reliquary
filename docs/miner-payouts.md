# Payouts at J+2 (collector rail)

Most orders pay through the weights: a settled period enters the weights, and Yuma consensus pays it out over the next periods. This is the native rail, and it is unchanged.

An order created on the collector rail (`payout_rail: collector` in the task registry) is paid another way. Its work is verified and settled per period like any other order's. The weights give it no share. Instead, the subnet's collector pays each hotkey what it earned, two UTC days later, by moving alpha stake to the hotkey owner's coldkey.

## What is owed

Each settled period of a collector-rail order owes each hotkey one obligation per unit type:

- **Amount**: the hotkey's settled share of the period, times the miner alpha per period frozen with the order's price ceiling (`pay.price_ceiling.miner_alpha_per_period`), rounded down to the rao. The share follows the same rules as on the native rail: verified output tokens at the order's price (prompt tokens at the input price when the order prices them, each lease at its own price on an auction). It is never more than the period's budget.
- **Id**: a SHA-256 of (order, period, hotkey, unit type). A period settled again later (verdicts that arrived after it was paid) gets one more part, with its own id. Each id is paid once.

`reliquary ledger report` computes the obligations from the settled period archives.

## When

- The work of UTC day D is paid at the pass of D+2, at 12:00 UTC. A period's day is the day it ended.
- A period settled after that pass is paid at the first pass after it settles.
- A period that is not settled owes nothing yet. It waits for its audit and is never cancelled. Past 7 days, a person looks at it.
- If one coldkey is owed less than the minimum in a pass, the amount waits and adds up until a later pass.

## How you receive it

- The collector sends one `transfer_stake` per coldkey: from the collector's coldkey to the coldkey that owned your hotkey when the period settled.
- The alpha arrives staked on the collector's hotkey, on this subnet, in your coldkey's name. It is yours: move it to your own hotkey (`btcli stake move`) or unstake it (`btcli stake remove`).
- If a transfer to your coldkey fails, it is retried alone at the next pass. After three failures it is held for a person to look at. A coldkey reserved as the destination of a coldkey swap, for instance, cannot receive transfers.

## Receipts

Every executed pass publishes a receipt (JSON) under `reliquary/public/payouts/transfers/<day>/`. An index at `reliquary/public/payouts/index.json` lists every receipt. A receipt lists each transfer (coldkey, amount, obligation ids), and each batch's extrinsic hash, block and events.

Any transfer can be checked on chain: find that extrinsic's `StakeTransferred` event, whose fields are (origin coldkey, destination coldkey, hotkey, origin netuid, destination netuid, amount).
