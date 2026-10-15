"""The open refusals that are the validator's capacity: one set, read by the session issuer (which tells its
listeners) and by the RL yield ledger (which marks the refused precommit). Imports nothing."""

CAPACITY_REFUSALS = frozenset({"sandbox_saturated", "open_queue_full", "task_unplaceable", "environment_unserved"})
"""Told to ``SessionEvents.refused``, except a saturation or a full queue of the refused miner's own operator's
making (``SessionIssuer._self_caused``). A key's full share of the open queue (``open_queue_key_full``) is never
one: the seeds a started group owes below M always have a ticket, so a key-full refusal is the key's own overrun. A
gateway's signed refusal of a token (``no_free_slot`` included) is ours whoever filled the machine: our placement
chose it. For an unused precommit that met one, its sessions that ran (in a group that could not be formed) are
neither useful nor wasted."""

__all__ = ["CAPACITY_REFUSALS"]
