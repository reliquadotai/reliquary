"""A saturation gives a capacity verdict only when others held the capacity."""
import asyncio

import pytest

attest = pytest.importorskip("reliquary_sandbox.attest")

from reliquary.protocol.service_episode import rl_engagement  # noqa: E402
from reliquary.sandbox.sessions import LIVE, SandboxPolicy, SessionBook  # noqa: E402
from tests.unit.sandbox_fixtures import MACHINE, NOW  # noqa: E402
from tests.unit.test_open_queue import Refusals
from tests.unit.test_rl_engagements import HOTKEY, build, engagement, open_  # noqa: E402

RECEIPT = {"document": {"kind": "open_refusal", "machine_id": MACHINE, "key_id": "k1", "session_id": "s",
                        "token_sha256": "a" * 64, "reason": "no_free_slot", "at": 1}, "signature": "sig"}


def refused(env, seed):
    env.issuer._fleet.placement = None                  # no room now; the machine could take it later
    return asyncio.run(env.issuer.open(hotkey=HOTKEY, request_id=f"{seed:032x}",
                                       engagement=engagement(env.precommit, seed)))


@pytest.mark.parametrize("free, listed, load", [(4, 0, 0), (1, 0, 3), (4, 2, 2), (3, 3, 3)])
def test_a_machines_reported_load_is_its_used_slots_or_listed_sessions(tmp_path, free, listed, load):
    from tests.unit.sandbox_fixtures import ADDRESS, capacity_report
    from tests.unit.test_fleet_resources import RESOURCES
    from tests.unit.test_sandbox_fleet import make

    fleet, machine, served, clock = make(tmp_path)
    clock.now = NOW
    served[ADDRESS] = capacity_report(machine, at=NOW, free=free,
                                      resources={**RESOURCES, "sessions": [f"s-{n}" for n in range(listed)]})
    asyncio.run(fleet.poll_once())
    assert fleet.reported_load((MACHINE,)) == load
    assert fleet.reported_load(("other",)) == 0


