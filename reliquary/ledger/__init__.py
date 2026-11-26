"""Payout ledgers kept beside the native weight rail.

``obligations`` computes what settled work owes each hotkey and when it falls
due; ``reconcile`` compares that with the miner incentive the chain paid;
``report`` assembles the daily report; ``sources`` reads the inputs (period
archives, price ceilings, the chain), read-only; ``payout`` pays what is due
on tasks of the collector rail, once each, from the collector's stake;
``payout_chain`` is its chain and signer; ``cli`` runs them as
``reliquary ledger report`` and ``reliquary ledger pay``.

Everything but ``payout`` and ``payout_chain`` is read-only. The payer is a
dry run unless it is told to execute and handed the collector's wallet.
"""
