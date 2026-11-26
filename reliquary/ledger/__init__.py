"""Payout ledgers kept beside the native weight rail.

``obligations`` computes what settled work owes each hotkey and when it falls
due; ``reconcile`` compares that with the miner incentive the chain paid;
``report`` assembles the daily report; ``sources`` reads the inputs (period
archives, price ceilings, the chain), read-only; ``cli`` runs it as
``reliquary ledger report``.

Read-only: nothing here holds a key, signs, transfers or writes to the bucket
or the chain.
"""
