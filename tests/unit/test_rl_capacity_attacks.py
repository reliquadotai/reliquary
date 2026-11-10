"""Attacks on the RL session quota, the open queue, the capacity attribution, the allocation route and envs added
to the running run: each runs against the real ledger, quota, queue and issuer (fakes only for the fleet and the
machines)."""
import asyncio
import time
from types import SimpleNamespace

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.infrastructure.sandbox_store import MemorySessionStore  # noqa: E402
from reliquary.sandbox.fleet import Placement  # noqa: E402
from reliquary.sandbox.rl_capacity import (  # noqa: E402
    LOW_YIELD, MAIN, NEWCOMER, YieldStats, entity_sizes, sessions_of,
)
from reliquary.sandbox.sessions import (  # noqa: E402
    LAPSED, LIVE, SUBMITTED, FanOutEvents, Grant, SandboxPolicy, SessionBook, SessionRecord,
)
from tests.unit.sandbox_fixtures import ADDRESS, MACHINE, NOW  # noqa: E402
from tests.unit.test_rl_engagements import HOTKEY, POLICY, build, close, engagement, open_  # noqa: E402

M = POLICY.pool_seeds // 2
POOL = POLICY.pool_seeds
WASTED_ROUNDS = -(-200 // POOL)                      # enough wasted precommits to fill a 200-session yield
# The split-invariance stack has its own pool of 32 seeds, independent of the fixture's:
P32 = 32
CAPACITY = 2_160                                     # N = 216, L = 64 (the production constants)


def sha(n):
    return f"{n:064x}"


def want(quota, hotkey, n, first, *, now, pool_seeds):
    """``hotkey`` asks for ``n`` sessions now: its current demand (the allocation is recomputed after)."""
    for i in range(n):
        quota.refusal(hotkey, now, precommit_sha256=sha(first + i // pool_seeds), seed_index=i % pool_seeds)
    quota._alloc_key = None


# --- a capacity verdict needs others' saturation, sessions that ran and a group that could not form

def refused_open(env, seed):
    env.issuer._fleet.placement = None
    outcome = open_(env, seed)
    assert outcome.reason == "sandbox_saturated"


# --- the reference miner's rule against the real quota, production regime, an idle fleet -----------------------

# --- the reference miner's rule against the real quota, under CONTENTION ---------------------------------------
# A fleet smaller than what the miners would run: every MAIN key's ceiling is above its weighted share. The quota and
# the miner's rule (``groups_to_run``) are the real ones; the ledger is a fixed snapshot (the standings do not move in
# an hour). A group that has not started sends one seed at a time until its first grant, then the rest at once
# (the reference miner); a refused seed is asked again every ``retry_s``.

CONTENDED = {"5B": (300.0, 40), "5C": (150.0, 40)}         # ceilings of 1 280 sessions each, far above any share
NEWCOMERS = {f"5N{n}": (None, 2) for n in range(3)}        # fresh keys at their member maximum (2 groups)


def weighted_shares(keys):
    """The MAIN keys' weighted shares of what the newcomers leave (their 3 x 2 groups fit their guarantee N)."""
    main = {h: w for h, (w, _c) in keys.items() if w is not None}
    left = CAPACITY - sum(2 * P32 for w, _c in keys.values() if w is None)
    return {h: left * w / sum(main.values()) for h, w in main.items()}


def tail_mean(rows, column=2):
    return sum(row[column] for row in rows) / len(rows)


# --- the signed allocation route: replay, a foreign hotkey, the rate ---------------------------------------------

def _allocation_route():
    pytest.importorskip("bittensor")
    from tests.unit.test_miner_allocation import MINER, allocation_client, body_of

    asked = []

    def allocation_of(hotkey):
        asked.append(hotkey)
        return {"hotkey": hotkey, "groups": {"e": 3}}

    return allocation_client(allocation_of), body_of, MINER, asked


@pytest.mark.parametrize("respell", [str.upper, lambda s: "0x" + s, lambda s: " " + s + " ",
                                     lambda s: s[:10] + "0x" + s[10:]])
def test_a_replayed_allocation_request_with_its_signature_respelt_is_refused(respell):
    client, body_of, miner, asked = _allocation_route()
    first = body_of()
    assert client.post("/rl/sessions/allocation", json=first).status_code == 200
    again = client.post("/rl/sessions/allocation", json={**first, "signature": respell(first["signature"])})
    assert again.status_code == 409 and again.json()["reason"] == "request_reused"   # the same signature bytes
    assert asked == [miner.ss58_address]                          # the eavesdropper learnt nothing


def test_a_foreign_hotkeys_signature_reads_nothing_and_spends_nothing_of_the_victims_minute():
    pytest.importorskip("bittensor")
    import bittensor as bt

    from reliquary.sandbox.rl_routes import ALLOCATION_MAX_PER_MINUTE

    client, body_of, miner, asked = _allocation_route()
    eve = bt.Keypair.create_from_uri("//Eve")
    for n in range(ALLOCATION_MAX_PER_MINUTE + 5):                # Eve floods requests naming Alice
        answer = client.post("/rl/sessions/allocation", json=body_of(eve, hotkey=miner.ss58_address, at=NOW - n % 5))
        assert answer.status_code == 403
    assert client.post("/rl/sessions/allocation", json=body_of()).status_code == 200   # Alice's minute untouched
    assert asked == [miner.ss58_address]


# --- the miner's manifest-digest refusal, on an env added to the running run -------------------------------------

# --- add-request integrity: a refused add leaves every byte of the run as it was ----------------------

def _swapped_after_writing(folder, req):
    import os

    os.chmod(folder / "conformance.json", 0o600)
    (folder / "conformance.json").write_bytes(b'{"gate": true, "passed": true}')   # swapped after the CLI checked it
    return (lambda name, added, attestation: None,                                 # even an accepting check
            "the attestation file is not the one the entry names")


def _named_like_a_launch_env(folder, req):
    from tests.unit.episode_v2_fixtures import ADDED, EPISODE

    req["add"] = {EPISODE: req["add"][ADDED]}                         # the launch episode env's name
    req["active"] = [name for name in req["active"] if name != ADDED]
    req["shares"] = {name: bps for name, bps in req["shares"].items() if name != ADDED}
    return lambda name, added, attestation: None, f"{EPISODE} is already in the schedule"


# --- the automatic switch-off of an added env: its revision can never be pre-empted -------------------

# --- snapshots that may lack an added env: the optional set opens no other hole -------------------------

# --- set_active_envs: a restart never counts an inactive env's machines in the fleet ----------------

# --- capacity attribution by operator: another hotkey of the same operator is the same miner ------------

# --- the open queue's per-key share: a key-full refusal is the key's own overrun --------------------------------

# --- a graded session the miner withdraws keeps its machine's final for the capacity verdict --------------------

