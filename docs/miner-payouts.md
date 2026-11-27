# Payouts at J+2 (collector rail)

Most orders pay through the weights: a settled period enters the weights, and Yuma consensus pays it out over the next periods. This is the native rail, and it is unchanged.

An order created on the collector rail (`payout_rail: collector` in the task registry) is paid another way. Its work is verified and settled per period like any other order's. The weights give it no share. Instead, the subnet's collector pays each hotkey what it earned, two UTC days later, by moving alpha stake to the hotkey owner's coldkey. An order chooses its rail when it is created, and never changes it.

## What is owed

Each settled period of a collector-rail order owes each hotkey one obligation per unit type:

- **Amount**: the hotkey's settled share of the period, times the miner alpha per period frozen with the order's price ceiling (`pay.price_ceiling.miner_alpha_per_period`), rounded down to the rao. The share follows the same rules as on the native rail: verified output tokens at the order's price (prompt tokens at the input price when the order prices them, each lease at its own price on an auction). It is never more than the period's budget. Only an order with a price ceiling can be on the collector rail, and only the frozen rate is paid: an amount valued any other way is held for a person.
- **Id**: a SHA-256 of (order, period, hotkey, unit type, the entry the period was first settled under). A period settled again later (verdicts that arrived after it was paid) gets one more part, with its own id. Each id is paid once, at the amount first sent.

`reliquary ledger report` computes the obligations from the settled period archives.

## When

- The work of UTC day D is paid at the pass of D+2, at 12:00 UTC. A period's day is the day it ended.
- A period settled after that pass is paid at the first pass after it settles, however old its work.
- A period that is not settled owes nothing yet. It waits for its audit and is never cancelled. Past 7 days, a person looks at it.
- If one coldkey is owed less than the floor in a pass, the amount waits and adds up until a later pass. The floor is the chain's minimum stake, 0.002 TAO (`DefaultMinStake`), in alpha at the subnet's price at the pass, raised by 25 % for the price to move before the transfer lands. It is never below the chain's transfer minimum, 0.0001 TAO (`DefaultMinTransfer`): the chain refuses a transfer worth less. Each receipt records the floor, in alpha and in TAO, the price and where it was read.
- If the collector's stake is short, the oldest obligations are paid first, and nothing younger is paid before them.

## How you receive it

- The collector sends one `transfer_stake` per coldkey: from the collector's coldkey to the coldkey that owned your hotkey at the end of the period's work. A hotkey sold or swapped later does not move what it earned before.
- The alpha arrives staked on the collector's hotkey, on this subnet, in your coldkey's name. It is yours: move it to your own hotkey (`btcli stake move`) or unstake it (`btcli stake remove`).
- If you later unstake part of it, a remainder below the chain's minimum stake is unstaked with it, as for any stake.

### When the chain refuses your coldkey

The chain refuses a transfer to some coldkeys, and one refused transfer reverts its whole batch. After a failed batch, each of its coldkeys is sent alone at the next pass. After three failures on chain, every obligation of that coldkey is held until a person clears it. The chain refuses a coldkey that:

- is reserved as the destination of an announced coldkey swap;
- has opted out of receiving stake on new hotkeys from others (`AccountRejectsStakingHotkeyGrowth`);
- already stakes on the most hotkeys the chain allows it to receive on from others (128).

To receive: clear the opt-out, or stake once on the collector's hotkey yourself (any amount; a hotkey you already stake on is always accepted). Then ask for your coldkey to be cleared.

## Receipts

Every executed pass that sends or settles a batch publishes a receipt (JSON) under `reliquary/public/payouts/transfers/<day>/`. An index at `reliquary/public/payouts/index.json` lists every receipt and the hash of every paid batch. A receipt lists each transfer (coldkey, amount, obligation ids), and each batch's extrinsic hash, block, events and fee (in TAO). A batch appears in one receipt; after a crash, it can appear in two, and its id and extrinsic hash identify it.

Any transfer can be checked on chain: find that extrinsic's `StakeTransferred` event, whose fields are (origin coldkey, destination coldkey, hotkey, origin netuid, destination netuid, amount). The event's amount is the TAO value of the stake moved, not the alpha: compare the destinations, and read the alpha from the receipt.
