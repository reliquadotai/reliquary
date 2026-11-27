# Collector, surplus burn and J+2 payouts: owner runbook

For the subnet owner. The steps below set up the collector, route the unpaid miner share to it in tiers, burn its surplus, and, later, pay collector-rail orders at J+2. None of them puts a secret in a file or on a command line: the collector's coldkey stays in its wallet, which asks for its password when it signs.

Every real transfer or burn is the owner's go: a dry run first, then `--execute` with `--confirm-rao` set to the exact amount the dry run printed. Any other amount sends nothing.

## What runs where

- **Routing**: the weight setter gives the collector `PAYOUT_FRACTION` of the unpaid share. `PAYOUT_HOTKEY` and `PAYOUT_FRACTION` are code constants in `reliquary/constants.py`. Each tier is a one-line release; watchtower updates every validator running the image within about 5 minutes.
- **Payer, burner, receipts**: `reliquary ledger pay | burn-surplus | publish | collector-status` run on one machine, the owner's for the pilot.
  - There is one journal, `~/.reliquary/payout/journal.sqlite`, and the receipts go to `~/.reliquary/payout/receipts/`.
  - Back up `~/.reliquary/payout/`: the journal is the record of what was paid and burned. Never copy it to run a second payer, because a second journal would not know what the first paid.
- **Reads**:
  - an archive node (`--network archive`, for blocks days back);
  - the bucket, read-only, for the registry and the period archives. The R2 read credentials go in the environment.

## 1. The collector's keys

1. Create a new coldkey and a hotkey under it, with btcli, on the owner's machine (check `btcli wallet --help` for your version's flags). Write the mnemonic down offline.
2. Never use the owner's coldkey: the chain treats every hotkey of the subnet owner's coldkey as the owner and keeps its incentive. The weight setter refuses such a hotkey too, and burns its part.
3. Send a little free TAO to the collector's coldkey: it pays the fees of burns and transfers. A receipt's `fee_rao` shows what one cost.

## 2. Register the hotkey on the subnet

1. Register the collector's hotkey on netuid 81 with btcli. It costs the registration burn (check it first).
2. The subnet is full, so registering evicts the weakest UID that is not immune. The new UID is immune for about 36 hours. Once the routing sends it weight, it earns more than any miner and cannot be pruned.
3. Do not set a collateral floor on the collector.
4. Check it:
   ```
   reliquary ledger collector-status --collector-hotkey <collector hotkey> --collector-coldkey <collector coldkey>
   ```
   It prints the UID, the coldkey (flagged if it is the owner's), and whether stake transfers are open on the subnet.

## 3. Route the unpaid share, tier by tier

The tiers are 1 %, 10 %, 50 % and 100 %, with no minimum duration. For each fraction `f`:

1. **Before**: note MinerBurned, the share of the miners' incentive the owner's hotkeys kept at the last tempo:
   ```
   reliquary ledger collector-status --collector-hotkey <collector hotkey>
   ```
2. **Release**: a pull request sets `PAYOUT_HOTKEY = "<collector hotkey>"` and `PAYOUT_FRACTION = f` in `reliquary/constants.py`, and nothing else. Merge it on the owner's go; the image is published and watchtower updates every validator.
3. **After the next tempo** (72 minutes):
   ```
   reliquary ledger collector-status --collector-hotkey <collector hotkey> --before <MinerBurned before> --fraction <f>
   ```
   `AS EXPECTED` means MinerBurned fell to about `before × (1 − f)`, within half a point; move to the next tier.
4. **If it did not**, after two tempos: some validator is not running the release (read its weights), or the hotkey has no UID. To roll back, release `PAYOUT_FRACTION = 0`.

Before going above 1 %, publish the collector's hotkey, its policy (what it keeps, what it burns, what it pays) and its receipts.

## 4. Burn the surplus

The collector's surplus is its stake beyond what it owes collector-rail work, what is in flight, and its reserve. In phase 1 (every order on the native rail), it owes nothing by transfer. The cap is the most burned per UTC day.

1. Dry run, which signs nothing and writes only a receipt:
   ```
   reliquary ledger burn-surplus --collector-coldkey <collector coldkey> --collector-hotkey <collector hotkey> --cap <alpha per day>
   ```
   It prints the stake, what is owed, in flight and kept, the surplus, what was already burned today, and this pass's amount in rao.
2. On the owner's go, the same command with `--execute --confirm-rao <the amount printed> --wallet-name <collector wallet>`.
3. Publish: `reliquary ledger publish` lists what it would write; `reliquary ledger publish --execute` writes it.

## 5. Pay collector-rail orders at J+2 (per-order pay, later)

Phase 1 is native only: no order is on the collector rail, so `reliquary ledger pay` plans nothing. Per-order pay starts with one order type, created with `payout_rail: collector`, and only once every validator runs the rule that gives such orders no weight. Otherwise the order would be paid twice.

1. Every day after 12:00 UTC, a dry run:
   ```
   reliquary ledger pay --collector-coldkey <collector coldkey> --collector-hotkey <collector hotkey>
   ```
   It prints the pass's total in rao, what is due, and what waits (under the minimum, unfunded, ownerless, or held after failed attempts).
2. On the owner's go, the same command with `--execute --confirm-rao <the total printed> --wallet-name <collector wallet>`.
3. Then `reliquary ledger publish --execute`.

The defaults are the approved ones, all settings:
- obligations due J+2 at 12:00 UTC, in alpha at each order's frozen rate;
- paid to the coldkey that owned the hotkey when the period settled;
- minimum transfer: the chain's minimum stake. A same-subnet `transfer_stake` has no chain minimum, so this is only a dust floor;
- batches of 32;
- reserve 0.

## When a command stops (exit code 3)

It says why:

- **A batch or burn sent earlier may still land.** Wait, and run again later. Nothing new is sent until it is final or expired.
- **A nonce was taken by another extrinsic.** The collector's key signed something else while a send was in flight. Read the chain, then settle the batch by hand. `paid` is `True` only if its transfers or burn happened after all:
  ```
  python -c "import os; from reliquary.ledger.payout import Journal; j = Journal(os.path.expanduser('~/.reliquary/payout/journal.sqlite')); j.settle_foreign('<batch id>', paid=False, detail='<what the chain showed>'); j.close()"
  ```
- **Obligations held out after three failed attempts.** The chain refused their destination, for example a coldkey reserved by a coldkey-swap announcement. Once that is fixed, clear them with `Journal.clear_attempts([...])` the same way.

Never sign anything else with the collector's coldkey while a payout or a burn may be in flight.
