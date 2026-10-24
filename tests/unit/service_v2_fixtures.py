# tests/unit/service_v2_fixtures.py
"""Builders for service-contract/v2 data (next RL run, phase 1)."""
from __future__ import annotations

import hashlib

from reliquary.constants import M_ROLLOUTS
from reliquary.protocol.service_contract import ServiceContract

MATH = "reliquary_dapo_math_v1"
CODE = "reliquary_code_v1"
SCIENCE = "reliquary_science_v1"


_TWO_M = object()


def contract_v2_dict(*, envs=(MATH, CODE), shares=None, exploration=1, pool_seeds=_TWO_M,
                     task_scoped=1, missing_box=None, visibility="task",
                     cooldown_windows=50, rows=1000) -> dict:
    if shares is None:
        base = 10000 // len(envs)
        shares = {env: base for env in envs}
        shares[envs[0]] += 10000 - base * len(envs)
    return {
        "schema": "service-contract/v2",
        "service_kind": "adaptive_training",
        "revision_id": "next-rl-test",
        "checkpoint": {"repo": "models/test", "revision": "d" * 40, "sha256": "b" * 64},
        "generation_contract_sha256": "c" * 64,
        "scoring": {"kind": "environment-reward/v1", "sigma_min_bps": 2400, "weights_bps": {"reward": 10000}},
        "policies": {
            "checkpoint": {"kind": "trainer-driven/v1", "task_scoped": task_scoped},
            "reward": {"kind": "exploration-first-scan/v1", "price_bps": 1500, "cap_bps": 1000,
                       "audit_bps": 1500, "new_hotkey_audit_groups": 100, "ban_seconds": 86400,
                       "max_tokens_per_group": 10_000_000},
            "cooldown_advice": {"kind": "in-zone-rotation/v1", "margin_bps": 8000, "min_windows": 1,
                                "max_windows": 100000, "smoothing_bps": 3000, "hysteresis_windows": 1,
                                "max_change_windows": 50, "min_first_scans": 100},
        },
        "environments": {
            env: {
                "version": hashlib.sha256(env.encode()).hexdigest(),
                "dataset": {"id": f"{env}-train", "rows": rows},
                # The pool is always 2 x M seeds; ``pool_seeds`` exists only to test the refusal.
                "sampling": {"kind": "public-seed-pool/v3", "group_size": M_ROLLOUTS,
                             "pool_seeds": 2 * M_ROLLOUTS if pool_seeds is _TWO_M else pool_seeds,
                             "renewal_windows": 1},
                "exploration": exploration,
                "missing_box": missing_box or ("uncertain" if "math" in env else "graded"),
                "cooldown_windows": cooldown_windows,
                "share_bps": shares[env],
            }
            for env in envs
        },
        "limits": {"max_groups": 10**9, "max_tokens": 10**15, "deadline_seconds": 10**8},
        "visibility": visibility,
    }


def contract_v2(**kwargs) -> ServiceContract:
    return ServiceContract.from_dict(contract_v2_dict(**kwargs))


def qualification_v2(contract: ServiceContract, **overrides) -> dict:
    value = {"schema": "service-runtime-qualification/v2", "qualified": True,
             "qualification_id": "unit-qualified", "contract_sha256": contract.sha256,
             "group_size": M_ROLLOUTS, "forced_seed_report_sha256": "f" * 64}
    value.update(overrides)
    return value
