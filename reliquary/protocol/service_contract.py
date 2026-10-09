"""Immutable service orders, separate from the historical generation contract."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Mapping

from reliquary.protocol.corpus_submission import MAX_TRAJECTORY_TOKENS
from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256

SCHEMA = "service-contract/v1"
MAX_SAFE_INTEGER = 2**53 - 1
SERVICE_KINDS = {"dataset_mapping", "dataset_curation", "adaptive_training"}
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}\Z")
_SHA = re.compile(r"[0-9a-f]{64}\Z")
_POLICIES = {
    "sampling": {
        "legacy/v1": {},
        "public-group-pool/v1": {"group_size": (2, 64), "pool_groups": (2, 1024), "renewal_windows": (1, 1000000)},
        "public-draw-pool/v1": {"group_size": (2, 64), "pool_draws": (3, 65536), "renewal_windows": (1, 1000000)},
    },
    "eligibility": {
        "all/v1": {},
        "dataset-epoch/v1": {"coverage_bps": (1, 10000), "refresh_windows": (1, 1000000), "max_epoch_windows": (1, 1000000)},
    },
    "cooldown": {
        "static/v1": {"windows": (0, 1000000)},
        "adaptive-rotation/v1": {
            "min_windows": (0, 1000000), "max_windows": (1, 1000000),
            "margin_bps": (1, 10000), "min_panel_groups": (1, 1000000),
            "coverage_bps": (1, 10000), "freshness_windows": (1, 1000000),
            "smoothing_bps": (1, 10000), "max_change_windows": (1, 1000000),
            "hysteresis_windows": (0, 1000000),
            "fallback_windows": (0, 1000000),
        },
    },
    "checkpoint": {
        "frozen/v1": {},
        "trainer-driven/v1": {"task_scoped": (0, 1)},
    },
    "reward": {
        "legacy/v1": {},
        "exploration-discount/v1": {
            "divisor": (2, 1000000), "budget_bps": (1, 10000),
            "refresh_windows": (1, 1000000), "max_tokens_per_group": (1, MAX_SAFE_INTEGER),
        },
    },
}


class ServiceContractError(ValueError):
    pass


def _object(value: Any, fields: set[str], name: str) -> dict:
    if not isinstance(value, dict) or set(value) != fields:
        raise ServiceContractError(f"{name}: expected fields {sorted(fields)}")
    return value


def _integer(value: Any, name: str, low: int = 1, high: int = MAX_SAFE_INTEGER) -> int:
    if type(value) is not int or not low <= value <= high:
        raise ServiceContractError(f"{name}: integer in [{low}, {high}] required")
    return value


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _ID.fullmatch(value) or ".." in value:
        raise ServiceContractError(f"{name}: canonical identifier required")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise ServiceContractError(f"{name}: lowercase SHA-256 required")
    return value


def _validate_v1(value: dict) -> None:
    _object(value, {"schema", "service_kind", "revision_id", "dataset", "checkpoint", "environment",
                    "generation_contract_sha256", "scoring", "policies", "limits", "visibility"}, "contract")
    if value["schema"] != SCHEMA or not isinstance(value["service_kind"], str) or value["service_kind"] not in SERVICE_KINDS:
        raise ServiceContractError("unknown service schema or kind")
    _identifier(value["revision_id"], "revision_id")
    _object(value["dataset"], {"id", "sha256"}, "dataset")
    _identifier(value["dataset"]["id"], "dataset.id")
    _sha(value["dataset"]["sha256"], "dataset.sha256")
    _object(value["checkpoint"], {"repo", "revision", "sha256"}, "checkpoint")
    for field in ("repo", "revision"):
        _identifier(value["checkpoint"][field], f"checkpoint.{field}")
    if not re.fullmatch(r"[0-9a-f]{40}", value["checkpoint"]["revision"]):
        raise ServiceContractError("checkpoint.revision must pin an immutable 40-hex commit")
    _sha(value["checkpoint"]["sha256"], "checkpoint.sha256")
    _object(value["environment"], {"id", "version"}, "environment")
    for field in ("id", "version"):
        _identifier(value["environment"][field], f"environment.{field}")
    _sha(value["generation_contract_sha256"], "generation_contract_sha256")
    score = _object(value["scoring"], {"kind", "sigma_min_bps", "weights_bps"}, "scoring")
    if not isinstance(score["kind"], str) or score["kind"] not in {"environment-reward/v1", "weighted-reward/v1"}:
        raise ServiceContractError("unknown scoring policy")
    _integer(score["sigma_min_bps"], "sigma_min_bps", 0, 10000)
    weights = score["weights_bps"]
    if not isinstance(weights, dict) or not 1 <= len(weights) <= 16:
        raise ServiceContractError("weights_bps: 1..16 metrics required")
    for name, weight in weights.items():
        _identifier(name, "metric")
        _integer(weight, f"weight.{name}", 1, 10000)
    if sum(weights.values()) != 10000:
        raise ServiceContractError("weights_bps must sum to 10000")
    if score["kind"] == "environment-reward/v1" and weights != {"reward": 10000}:
        raise ServiceContractError("environment-reward/v1 uses the environment reward unchanged")
    policies = _object(value["policies"], set(_POLICIES), "policies")
    for lane, variants in _POLICIES.items():
        policy = policies[lane]
        if not isinstance(policy, dict) or not isinstance(policy.get("kind"), str) or policy.get("kind") not in variants:
            raise ServiceContractError(f"unknown {lane} policy")
        params = variants[policy["kind"]]
        _object(policy, {"kind", *params}, lane)
        for name, bounds in params.items():
            _integer(policy[name], f"{lane}.{name}", *bounds)
    sampling = policies["sampling"]
    if sampling["kind"] == "public-draw-pool/v1" and sampling["pool_draws"] <= sampling["group_size"]:
        raise ServiceContractError("pool_draws must exceed group_size")
    cooldown = policies["cooldown"]
    if cooldown["kind"] == "adaptive-rotation/v1":
        if cooldown["min_windows"] > cooldown["max_windows"] or not cooldown["min_windows"] <= cooldown["fallback_windows"] <= cooldown["max_windows"]:
            raise ServiceContractError("invalid cooldown bounds/fallback")
        if policies["eligibility"]["kind"] != "dataset-epoch/v1":
            raise ServiceContractError("adaptive cooldown needs dataset-epoch eligibility")
    if value["service_kind"] != "adaptive_training":
        if policies["checkpoint"]["kind"] != "frozen/v1" or policies["reward"]["kind"] != "legacy/v1" or policies["cooldown"]["kind"] != "static/v1":
            raise ServiceContractError("mapping/curation requires frozen checkpoint and no adaptive incentives")
    elif policies["checkpoint"]["kind"] != "trainer-driven/v1":
        raise ServiceContractError("adaptive training needs trainer-driven checkpoints")
    limits = _object(value["limits"], {"max_groups", "max_tokens", "deadline_seconds"}, "limits")
    for name, amount in limits.items():
        _integer(amount, f"limits.{name}")
    if not isinstance(value["visibility"], str) or value["visibility"] not in {"private", "task"}:
        raise ServiceContractError("unknown visibility")


SCHEMA_V2 = "service-contract/v2"
_ENV_ID = re.compile(r"[a-z0-9][a-z0-9_]{0,63}\Z")
_DATASET_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/@_-]{0,255}\Z")
_V2_FIELDS = {"schema", "service_kind", "revision_id", "checkpoint", "generation_contract_sha256",
              "scoring", "policies", "environments", "limits", "visibility"}
_V2_REWARD = {"price_bps": (1, 10000), "cap_bps": (0, 10000), "audit_bps": (0, 10000),
              "new_hotkey_audit_groups": (0, 1_000_000), "ban_seconds": (0, 30 * 86400),
              "max_tokens_per_group": (1, MAX_SAFE_INTEGER)}
_V2_ADVICE = {"margin_bps": (1, 10000), "min_windows": (0, 1_000_000), "max_windows": (1, 1_000_000),
              "smoothing_bps": (1, 10000), "hysteresis_windows": (0, 1_000_000),
              "max_change_windows": (1, 1_000_000), "min_first_scans": (1, 1_000_000)}
# The pool of an env is ``pool_seeds`` public seeds, always exactly 2 x ``group_size``; a miner
# submits any ``group_size`` distinct seeds of it (reliquary.protocol.seed_pool). The former
# fixed-candidate-group policy ``public-group-pool/v1`` (``pool_groups``) is not a v2 policy.
PUBLIC_SEED_POOL = "public-seed-pool/v3"
_V2_SAMPLING = {"legacy/v1": {}, PUBLIC_SEED_POOL: {"group_size": (2, 64), "pool_seeds": (4, 128),
                                                    "renewal_windows": (1, 1000000)}}
SUPPORTED_V2_CAPABILITIES = frozenset({
    "environment-reward/v1", "trainer-driven/v1", "task-scoped/v1", "exploration-first-scan/v1",
    "in-zone-rotation/v1", PUBLIC_SEED_POOL, "legacy/v1",
})

# Phase 2 (plan 2C): an env whose groups are signed-sandbox episodes. The capability is announced only
# for an order that has such an env (``supported_v2_capabilities``), so a single-turn order keeps its bytes.
EPISODE_CAPABILITY = "signed-sandbox-episode/v1"
EPISODE_BUDGET_FIELDS = ("max_calls", "per_call_timeout_s", "cpu_s", "wall_s", "memory_bytes", "pids",
                         "disk_bytes")
EPISODE_TOOLS = frozenset({"bash", "edit"})
# One episode is one corpus trajectory: its tokens stay under the corpus trajectory cap.
MAX_EPISODE_TOKENS = MAX_TRAJECTORY_TOKENS
# Ceilings of each session budget: the smaller of the review's bound and the sandbox service's own
# maximum (reliquary_sandbox_service/settings.py: episode_max_calls le 100_000, episode_max_call_timeout_s
# le 24 h, episode_max_pids le 65536, and the defaults of episode_max_cpu_s 3600, episode_max_wall_s 4 h,
# episode_max_memory_bytes 8 GiB, episode_max_disk_bytes 10 GiB, which have no hard "le").
EPISODE_BUDGET_CEILINGS = {
    "max_calls": 4096, "per_call_timeout_s": 3600, "cpu_s": 3600, "wall_s": 4 * 3600,
    "memory_bytes": 8 * 1024**3, "pids": 65536, "disk_bytes": 10 * 1024**3,
}
_EPISODE_FIELDS = frozenset({"kind", "sandbox_env", "split", "env_package", "tools", "max_turns",
                             "max_tokens_per_turn", "max_episode_tokens", "budgets"})
_SANDBOX_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:+-]{0,63}\Z")
_ENV_PACKAGE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}==[A-Za-z0-9][A-Za-z0-9.+_!-]{0,127}\Z")


def _validate_episode(name: str, env: dict) -> None:
    """The ``episode`` block of one v2 env: what record 0 must name, the limits, the session budgets."""
    episode = _object(env["episode"], set(_EPISODE_FIELDS), f"{name}.episode")
    if episode["kind"] != EPISODE_CAPABILITY:
        raise ServiceContractError(f"{name}.episode.kind must be {EPISODE_CAPABILITY}")
    for field_name in ("sandbox_env", "split"):
        if not isinstance(episode[field_name], str) or not _SANDBOX_NAME.fullmatch(episode[field_name]):
            raise ServiceContractError(f"{name}.episode.{field_name}: sandbox identifier required")
    if not isinstance(episode["env_package"], str) or not _ENV_PACKAGE.fullmatch(episode["env_package"]):
        raise ServiceContractError(f"{name}.episode.env_package must be name==version")
    tools = episode["tools"]
    if (type(tools) is not list or not tools or any(not isinstance(tool, str) for tool in tools)
            or tools != sorted(set(tools)) or not set(tools) <= EPISODE_TOOLS):
        raise ServiceContractError(
            f"{name}.episode.tools: a sorted, unique, non-empty subset of {sorted(EPISODE_TOOLS)}")
    from reliquary.protocol.submission import MAX_SIGNED_EPISODE_TURNS

    _integer(episode["max_turns"], f"{name}.episode.max_turns", 1, MAX_SIGNED_EPISODE_TURNS)
    _integer(episode["max_tokens_per_turn"], f"{name}.episode.max_tokens_per_turn", 1, MAX_EPISODE_TOKENS)
    _integer(episode["max_episode_tokens"], f"{name}.episode.max_episode_tokens", 2, MAX_EPISODE_TOKENS)
    if episode["max_tokens_per_turn"] >= episode["max_episode_tokens"]:
        raise ServiceContractError(f"{name}.episode: a turn must fit inside the episode")
    budgets = _object(episode["budgets"], set(EPISODE_BUDGET_FIELDS), f"{name}.episode.budgets")
    for field_name in EPISODE_BUDGET_FIELDS:
        _integer(budgets[field_name], f"{name}.episode.budgets.{field_name}", 1,
                 EPISODE_BUDGET_CEILINGS[field_name])
    if env["sampling"]["kind"] != PUBLIC_SEED_POOL:
        raise ServiceContractError(f"{name}: an episode environment draws from a public seed pool")
    if env["missing_box"] != "graded":
        raise ServiceContractError(f"{name}: an episode's reward is its final record's; missing_box is graded")


@dataclass(frozen=True, slots=True)
class EpisodeEnvPolicy:
    """The ``episode`` block of one env, typed (plan 2C)."""

    environment: str
    sandbox_env: str
    split: str
    env_package: str
    tools: tuple[str, ...]
    max_turns: int
    max_tokens_per_turn: int
    max_episode_tokens: int
    budgets: tuple[tuple[str, int], ...]
    pool_seeds: int

    def budgets_dict(self) -> dict[str, int]:
        return dict(self.budgets)



def _validate_v2(value: dict) -> None:
    _object(value, _V2_FIELDS, "contract")
    if value["service_kind"] != "adaptive_training":
        raise ServiceContractError("service-contract/v2 orders adaptive_training only")
    _identifier(value["revision_id"], "revision_id")
    _object(value["checkpoint"], {"repo", "revision", "sha256"}, "checkpoint")
    _identifier(value["checkpoint"]["repo"], "checkpoint.repo")
    if not isinstance(value["checkpoint"]["revision"], str) or not re.fullmatch(r"[0-9a-f]{40}", value["checkpoint"]["revision"]):
        raise ServiceContractError("checkpoint.revision must pin an immutable 40-hex commit")
    _sha(value["checkpoint"]["sha256"], "checkpoint.sha256")
    _sha(value["generation_contract_sha256"], "generation_contract_sha256")
    score = _object(value["scoring"], {"kind", "sigma_min_bps", "weights_bps"}, "scoring")
    if score["kind"] != "environment-reward/v1" or score["weights_bps"] != {"reward": 10000}:
        raise ServiceContractError("v2 scoring is the environment reward unchanged")
    _integer(score["sigma_min_bps"], "sigma_min_bps", 0, 10000)
    policies = _object(value["policies"], {"checkpoint", "reward", "cooldown_advice"}, "policies")
    checkpoint = _object(policies["checkpoint"], {"kind", "task_scoped"}, "checkpoint policy")
    if checkpoint["kind"] != "trainer-driven/v1":
        raise ServiceContractError("v2 needs trainer-driven/v1 checkpoints")
    _integer(checkpoint["task_scoped"], "task_scoped", 0, 1)
    reward = _object(policies["reward"], {"kind", *_V2_REWARD}, "reward policy")
    if reward["kind"] != "exploration-first-scan/v1":
        raise ServiceContractError("unknown v2 reward policy")
    for name, bounds in _V2_REWARD.items():
        _integer(reward[name], name, *bounds)
    advice = _object(policies["cooldown_advice"], {"kind", *_V2_ADVICE}, "cooldown_advice")
    if advice["kind"] != "in-zone-rotation/v1":
        raise ServiceContractError("unknown cooldown advice policy")
    for name, bounds in _V2_ADVICE.items():
        _integer(advice[name], name, *bounds)
    if advice["min_windows"] > advice["max_windows"]:
        raise ServiceContractError("cooldown advice bounds are inverted")
    environments = value["environments"]
    if not isinstance(environments, dict) or not 1 <= len(environments) <= 16:
        raise ServiceContractError("environments: 1..16 entries required")
    total = 0
    for name, env in environments.items():
        if not isinstance(name, str) or not _ENV_ID.fullmatch(name):
            raise ServiceContractError(f"invalid environment id {name!r}")
        env_fields = {"version", "dataset", "sampling", "exploration", "missing_box",
                      "cooldown_windows", "share_bps"}
        if isinstance(env, dict) and "episode" in env:
            env_fields = env_fields | {"episode"}
        _object(env, env_fields, f"environment {name}")
        _sha(env["version"], f"{name}.version")
        dataset = _object(env["dataset"], {"id", "rows"}, f"{name}.dataset")
        if not isinstance(dataset["id"], str) or not _DATASET_ID.fullmatch(dataset["id"]) or ".." in dataset["id"]:
            raise ServiceContractError(f"{name}.dataset.id: canonical identifier required")
        _integer(dataset["rows"], f"{name}.dataset.rows", 1, 2**31)
        sampling = env["sampling"]
        if not isinstance(sampling, dict) or not isinstance(sampling.get("kind"), str) or sampling["kind"] not in _V2_SAMPLING:
            raise ServiceContractError(f"{name}: unknown sampling policy")
        params = _V2_SAMPLING[sampling["kind"]]
        _object(sampling, {"kind", *params}, f"{name}.sampling")
        for field_name, bounds in params.items():
            _integer(sampling[field_name], f"{name}.sampling.{field_name}", *bounds)
        if sampling["kind"] == PUBLIC_SEED_POOL and sampling["pool_seeds"] != 2 * sampling["group_size"]:
            raise ServiceContractError(f"{name}.sampling.pool_seeds must be exactly 2 x group_size")
        if sampling["kind"] == PUBLIC_SEED_POOL and sampling["renewal_windows"] != 1:
            raise ServiceContractError(f"{name}.sampling.renewal_windows must be 1: a public seed pool renews every window")
        if type(env["exploration"]) is not int or env["exploration"] not in (0, 1):
            raise ServiceContractError(f"{name}.exploration must be 0 or 1")
        if env["missing_box"] not in ("uncertain", "graded"):
            raise ServiceContractError(f"{name}.missing_box must be uncertain or graded")
        _integer(env["cooldown_windows"], f"{name}.cooldown_windows", 0, 1_000_000)
        total += _integer(env["share_bps"], f"{name}.share_bps", 0, 10000)
        if "episode" in env:
            _validate_episode(name, env)
    if total != 10000:
        raise ServiceContractError("environment share_bps must sum to 10000")
    limits = _object(value["limits"], {"max_groups", "max_tokens", "deadline_seconds"}, "limits")
    for name, amount in limits.items():
        _integer(amount, f"limits.{name}")
    if not isinstance(value["visibility"], str) or value["visibility"] not in {"private", "task"}:
        raise ServiceContractError("unknown visibility")


def validate_service_contract(value: dict) -> None:
    if isinstance(value, dict) and value.get("schema") == SCHEMA_V2:
        _validate_v2(value)
    else:
        _validate_v1(value)


@dataclass(frozen=True, slots=True)
class ServiceContract:
    """Canonical bytes prevent a caller mutating an already ordered contract."""

    canonical: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.canonical, bytes) or len(self.canonical) > 65536:
            raise ServiceContractError("contract must be at most 64 KiB")
        value = _decode(self.canonical)
        validate_service_contract(value)
        if canonical_json_bytes(value) != self.canonical:
            raise ServiceContractError("contract bytes must be canonical")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ServiceContract:
        try:
            return cls(canonical_json_bytes(value))
        except (TypeError, OverflowError) as exc:
            raise ServiceContractError(str(exc)) from exc

    def to_dict(self) -> dict:
        return json.loads(self.canonical)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def version(self) -> int:
        return 2 if self.to_dict()["schema"] == SCHEMA_V2 else 1

    def _v2(self) -> dict:
        value = self.to_dict()
        if value["schema"] != SCHEMA_V2:
            raise ServiceContractError("this accessor needs service-contract/v2")
        return value

    @property
    def environments(self) -> dict:
        return self._v2()["environments"]

    def environment(self, name: str) -> dict:
        environments = self.environments
        if name not in environments:
            raise ServiceContractError(f"environment {name!r} is not in this order")
        return environments[name]

    def episode_policy(self, name: str) -> EpisodeEnvPolicy | None:
        """The env's signed-episode policy, or None for a single-turn env."""
        env = self.environment(name)
        episode = env.get("episode")
        if episode is None:
            return None
        return EpisodeEnvPolicy(
            environment=name, sandbox_env=episode["sandbox_env"], split=episode["split"],
            env_package=episode["env_package"], tools=tuple(episode["tools"]),
            max_turns=episode["max_turns"], max_tokens_per_turn=episode["max_tokens_per_turn"],
            max_episode_tokens=episode["max_episode_tokens"],
            budgets=tuple((field_name, episode["budgets"][field_name]) for field_name in EPISODE_BUDGET_FIELDS),
            pool_seeds=env["sampling"]["pool_seeds"])

    @property
    def episode_environments(self) -> tuple[str, ...]:
        if self.version != 2:
            return ()
        return tuple(sorted(name for name, env in self.environments.items() if "episode" in env))

    @property
    def reward_policy(self) -> dict:
        return self._v2()["policies"]["reward"]

    @property
    def advice_policy(self) -> dict:
        return self._v2()["policies"]["cooldown_advice"]

    @property
    def context_sha256(self) -> str:
        if self.version == 2:
            raise ServiceContractError("v2 observations are run-wide; there is no checkpoint-bound context")
        value = self.to_dict()
        return canonical_sha256({name: value[name] for name in
                                 ("dataset", "checkpoint", "environment", "generation_contract_sha256", "scoring")})

    def require_capabilities(self, supported: set[str]) -> None:
        value = self.to_dict()
        if value["schema"] == SCHEMA_V2:
            policies = value["policies"]
            required = {value["scoring"]["kind"], policies["checkpoint"]["kind"],
                        policies["reward"]["kind"], policies["cooldown_advice"]["kind"],
                        *(env["sampling"]["kind"] for env in value["environments"].values())}
            if policies["checkpoint"]["task_scoped"] == 1:
                required.add("task-scoped/v1")
            required |= {env["episode"]["kind"] for env in value["environments"].values() if "episode" in env}
        else:
            required = {value["scoring"]["kind"], *(p["kind"] for p in value["policies"].values())}
            if value["policies"]["checkpoint"].get("task_scoped") == 1:
                required.add("task-scoped/v1")
        missing = required - supported
        if missing:
            raise ServiceContractError(f"runtime lacks capabilities: {sorted(missing)}")


def _decode(raw: bytes) -> dict:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ServiceContractError(f"duplicate key {key}")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=pairs)
    except (ValueError, UnicodeDecodeError, RecursionError) as exc:
        raise ServiceContractError("invalid contract JSON") from exc


def parse_service_contract(raw: bytes) -> ServiceContract:
    return ServiceContract(raw)


def supported_v2_capabilities(contract: ServiceContract) -> frozenset[str]:
    """What a v2 runtime supports and announces for ``contract``: the v2 set, plus the episode capability
    only when the order has an episode env (a single-turn order's announcement keeps its exact bytes)."""
    if contract.version == 2 and contract.episode_environments:
        return SUPPORTED_V2_CAPABILITIES | {EPISODE_CAPABILITY}
    return SUPPORTED_V2_CAPABILITIES
