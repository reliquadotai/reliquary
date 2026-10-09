# tests/unit/episode_v2_fixtures.py
"""Builders for signed-episode environments in the v2 service RL path (phase 2, plan 2C)."""
from __future__ import annotations

import time

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.runtime import ServiceRuntime, protocol_slot_geometry
from tests.unit.service_v2_fixtures import MATH, contract_v2_dict, qualification_v2

EPISODE = "reliquary_test_episode_v1"
SANDBOX_ENV = "reliquary-swe"            # the fake gateway's env name (test_signed_episode_e2e)
SPLIT = "train:rl"                       # accepted by reliquary_sandbox_service.episodes.testing
ENV_PACKAGE = "reliquary-swe==0.1.0a1+g0123456789abcdef"   # test_corpus_job_signed_sandbox.ENV_PACKAGE
REVISION = "d" * 40                      # contract_v2_dict's checkpoint revision
WINDOW_BEACON = "ab" * 32
POOL = 0.25
TASK = 3
GIB = 1024**3
BUDGETS = {"max_calls": 64, "per_call_timeout_s": 600, "cpu_s": 3600, "wall_s": 3600,
           "memory_bytes": 4 * GIB, "pids": 1024, "disk_bytes": 10 * GIB}


def episode_block(**overrides) -> dict:
    value = {"kind": "signed-sandbox-episode/v1", "sandbox_env": SANDBOX_ENV, "split": SPLIT,
             "env_package": ENV_PACKAGE, "tools": ["bash", "edit"], "max_turns": 8,
             "max_tokens_per_turn": 512, "max_episode_tokens": 4096, "budgets": dict(BUDGETS)}
    value.update(overrides)
    return value


def episode_contract_dict(*, episode=None, **kwargs) -> dict:
    kwargs.setdefault("envs", (MATH, EPISODE))
    value = contract_v2_dict(**kwargs)
    value["environments"][EPISODE]["episode"] = episode_block() if episode is None else episode
    return value


def episode_contract(**kwargs) -> ServiceContract:
    return ServiceContract.from_dict(episode_contract_dict(**kwargs))


def episode_runtime(tmp_path, contract=None, *, window=1) -> ServiceRuntime:
    """A real runtime with ``window`` frozen and announced on the episode order."""
    contract = contract or episode_contract()
    tmp_path.mkdir(parents=True, exist_ok=True)
    rt = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification_v2(contract),
                        now=time.time() - 10, drand_round_at=lambda instant: 1_000)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=REVISION)
    picks, slots = protocol_slot_geometry()
    rt.open_window(window, pools={name: POOL for name in contract.environments}, picks_target=picks,
                   batch_slots=slots, now=time.time() - 5)
    rt.announcement(window=window, randomness=WINDOW_BEACON)
    return rt
