# Collector, surplus burn and J+2 payouts: owner runbook

For the subnet owner. The steps below set up the collector, route the unpaid miner share to it in tiers, burn its surplus, and, later, pay collector-rail orders at J+2. None of them puts a secret in a file or on a command line: the collector's coldkey stays in its wallet, which asks for its password when it signs.

Every real transfer or burn is the owner's go: a dry run first, then `--execute` with `--confirm-rao` set to the exact amount the dry run printed (and, for a payout, `--confirm-plan` set to the plan digest it printed). Anything else sends nothing.

## What runs where

- **Routing**: the weight setter gives the collector a fraction of the unpaid share. The collector hotkey and the tier schedule are code constants in `reliquary/constants.py` (`PAYOUT_HOTKEY`, `PAYOUT_TIERS`: each tier a fraction and the chain block it takes effect from). A tier is a release; every validator holding it switches at the same block (section 3). They come from the routing pull request, which merges before any tier.
- **Payer, burner, receipts**: `reliquary ledger pay | burn-surplus | publish | collector-status | journal-init | settle | clear-held | find-nonce` run on one machine, the owner's for the pilot.
  - There is one journal, `~/.reliquary/payout/journal.sqlite`, and the receipts go to `~/.reliquary/payout/receipts/`.
  - Back up `~/.reliquary/payout/`: the journal is the record of what was paid and burned. Never copy it to run a second payer: a second journal would not know what the first paid. A restored or copied journal stops at its first pass (see "Restoring the journal").
- **Reads**:
  - an archive node (`--network archive`, for blocks days back);
  - the bucket, read-only, for the registry and the period archives, with R2 read credentials in the environment.
- **Writes to the bucket**: only `publish --execute`, under `reliquary/public/payouts/`. Give it credentials that can write that prefix and nothing else, set only for the publish command, never kept with the read credentials.

## 1. The collector's keys

1. Create a new coldkey and a hotkey under it, with btcli, on the owner's machine (check `btcli wallet --help` for your version's flags). Write the mnemonic down offline.
2. Never use the owner's coldkey or the subnet owner hotkey: the chain treats every hotkey of the subnet owner's coldkey, and the owner hotkey, as the owner's and keeps their incentive. The weight setter refuses such a hotkey and burns its part; `pay` and `burn-surplus` refuse such a collector.
3. Send a little free TAO to the collector's coldkey: it pays the fees of burns and transfers, and keeps the account alive. A receipt's `fee_tao_rao` shows what one cost, in TAO.
4. Do not set an auto-stake destination on the collector's coldkey: the chain would credit the collector's incentive there, while the payer and the burner read only the collector's position on its own hotkey.
5. Initialise the payer's journal at the collector's nonce, once, before any execution:
   ```
   reliquary ledger journal-init --collector-coldkey <collector coldkey> --collector-hotkey <collector hotkey> --reason "first journal"
   ```
   It refuses if the bucket's published receipts, or the local ones, list a paid batch this journal does not hold.

## 2. Register the hotkey on the subnet, when tier 1 is ready

1. Prepare the tier-1 release first (section 3): the new UID is immune only for the subnet's immunity period, and earns nothing until tier 1 takes effect.
2. Register the collector's hotkey on netuid 81 with btcli. It costs the registration burn (check it first). The subnet is full, so registering evicts the weakest UID that is not immune.
3. Do not set a collateral floor on the collector.
4. Check it:
   ```
   reliquary ledger collector-status --collector-hotkey <collector hotkey> --collector-coldkey <collector coldkey>
   ```
   It prints the UID, the coldkey (flagged if it, or the hotkey, is the owner's), whether stake transfers are open, the commit-reveal setting, and the block its immunity ends at. Tier 1 must take effect, and the collector's incentive be above the weakest UIDs', before that block.

## 3. Route the unpaid share, tier by tier

The tiers are 1 %, 10 %, 50 % and 100 %, with no minimum duration. Validators with a permit must route alike: in an epoch where they disagree, Yuma clips the minority, and the tier may not take effect at all. So a tier takes effect at a chain block carried in the release, not when each validator restarts:

1. **Before**: note MinerBurned, the share of the miners' incentive the owner's hotkeys kept at the last tempo:
   ```
   reliquary ledger collector-status --collector-hotkey <collector hotkey>
   ```
2. **Release**: a pull request adds `(<activation block>, f)` to `PAYOUT_TIERS` in `reliquary/constants.py` and updates the expected schedule its test pins (`EXPECTED_PAYOUT_TIERS`), and nothing else. Choose the activation block at least two tempos (720 blocks) after the merge, so the image is built and every validator holds it first.
3. **Before the activation block**:
   - restart the trainer validator, which runs without watchtower, and any validator on a hotfix image, on the release;
   - check that every validator with a permit runs it (its version, or its weights once the block passes). Never advance a tier while one lags.
4. **After the activation block**, from the tempo commit-reveal allows (collector-status prints it: 1 + the reveal period):
   ```
   reliquary ledger collector-status --collector-hotkey <collector hotkey> --before <MinerBurned before> --fraction <f> --from-fraction <previous f>
   ```
   Shares are written 0 to 1 (1 % is 0.01). `AS EXPECTED` means the collector's part of the unpaid share is about `f`, and MinerBurned fell to about `before × (1 − f) / (1 − previous f)`; move to the next tier.
5. **If it did not**: a validator is not running the release (read its weights), or the hotkey has no UID. To roll back, release a tier with a later block and fraction 0. Never roll back to 0 while a collector-rail order is live: its miners are paid only by the collector, and the collector's income would stop.

Before going above 1 %, publish the collector's hotkey and its policy: what it keeps, what it burns (only its own stake: `burn_alpha` never reaches the alpha miners hold on its hotkey), what it pays, and its receipts.

## 4. Burn the surplus

The collector's surplus is its stake beyond what it owes collector-rail work (whatever its age), what that work still awaiting its audit may owe, what is in flight, and its reserve. In phase 1 (every order on the native rail), it owes nothing by transfer. The cap is the most burned per UTC day, by the clock.

1. Dry run, which signs nothing and writes only a receipt:
   ```
   reliquary ledger burn-surplus --collector-coldkey <collector coldkey> --collector-hotkey <collector hotkey> --cap <alpha per day>
   ```
   It prints the stake, what is owed, awaiting audit, in flight and kept, the surplus, what was already burned today, and this pass's amount in rao.
2. On the owner's go, the same command with `--execute --confirm-rao <the amount printed> --wallet-name <collector wallet>`. An execution burns as of now, from the registry and the bucket's archives.
3. It burns nothing when what is owed cannot be known: the registry is missing, a collector-rail archive cannot be valued, a task the journal paid is no longer on the rail, or a collector-rail order has no frozen rate.
4. Publish: `reliquary ledger publish` lists what it would write; `reliquary ledger publish --execute` writes it. Publish after every execution: the published index is how a restored journal is checked.

## 5. Pay collector-rail orders at J+2 (per-order pay, later)

Phase 1 is native only: no order is on the collector rail, so `reliquary ledger pay` plans nothing. Before the first collector-rail order:

- every validator with a permit runs the weight setter that gives collector-rail work no weight (`jobs create --payout-rail collector` asks for `--fleet-skips-collector-rail`): otherwise the order is paid twice;
- pruning protection for that order's miners is in place (D9): they earn no incentive from the weights, and would otherwise be pruned while the collector pays them;
- a testnet pass with at least two destinations, and a testnet burn, through the real chain and signer: the stored extrinsic hash is found on chain and the `StakeTransferred` and `AlphaBurned` events reconcile.

Then:

1. Every day after 12:00 UTC, a dry run:
   ```
   reliquary ledger pay --collector-coldkey <collector coldkey> --collector-hotkey <collector hotkey>
   ```
   It prints the pass's total in rao and its plan digest, what is due, and what waits (under the floor, unfunded, ownerless) or is held for a person (a rate not frozen, an amount changed since first sent, the subnet owner's coldkey, a coldkey the chain refused three times).
2. On the owner's go, the same command with `--execute --confirm-rao <the total printed> --confirm-plan <the digest printed> --wallet-name <collector wallet>`. An execution pays as of now, at the approved terms, from the registry and the bucket's archives.
3. Then `reliquary ledger publish --execute`.

The defaults are the approved ones:
- obligations due J+2 at 12:00 UTC, in alpha at each order's frozen rate, from the rail's first work period;
- paid to the coldkey that owned the hotkey at the end of the work period;
- minimum transfer, a dust floor: the rao count of the chain's `NominatorMinRequiredStake`, read as alpha (0.01 alpha on 2026-11-27), unless `--min-transfer` is given; each receipt records the floor and where it came from. The chain also refuses a transfer worth less than 0.0001 TAO (`DefaultMinTransfer`): such a transfer reverts its batch and is retried alone;
- batches of 32;
- reserve 0.

## When a command stops (exit code 3)

It says why, and sends nothing:

- **A batch or burn sent earlier may still land.** Wait, and run again later.
- **The chain could not be read.** Run again; nothing was settled on the bad read.
- **The collector's nonce is not the one the journal expects.** The collector's key signed something the journal did not (or the journal is not the one that paid). Find what took the nonce, read-only:
  ```
  reliquary ledger find-nonce --collector-coldkey <collector coldkey> --nonce <nonce> --from-block <block> --to-block <block>
  ```
  Once every such extrinsic is accounted for (none paid or burned what the journal still owes), re-align: `reliquary ledger journal-init ... --resync --reason "<what was checked>"`.
- **A batch waits on a person** (its nonce went to another extrinsic, its outcome is unreadable, or it landed not as planned). Read its block on chain, then decide, with no default:
  ```
  reliquary ledger settle <batch id> --landed --evidence "<extrinsic, block, events>"
  reliquary ledger settle <batch id> --not-landed --evidence "<extrinsic, block, events>"
  ```
  `--landed` marks its obligations paid (or its burn counted); `--not-landed` makes them due again. A batch that landed not as planned can only be settled `--landed`, once its destinations are checked. Every decision is kept in the journal.
- **A coldkey held after three refused transfers.** The chain refused it, for example a coldkey reserved by a coldkey-swap announcement. Once that is fixed: `reliquary ledger clear-held --coldkey <coldkey> --evidence "<why it can be paid now>"`.

Never sign anything else with the collector's coldkey: every pass stops until the journal is re-aligned.

## Restoring the journal

A journal restored from a backup does not know the batches sent since: the next pass stops on the collector's nonce. Before re-aligning it:

1. Publish any local receipts not yet published.
2. Compare the published index with the restored journal: `journal-init --resync` refuses while a published or local receipt lists a paid batch the journal does not hold.
3. Account for every collector extrinsic since the journal's last batch with `find-nonce`.

A journal lost with no backup is not replaced by a new one: `journal-init` refuses while receipts list paid batches it does not hold.
