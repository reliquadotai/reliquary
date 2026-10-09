"""Public seed pools: 2 x M seeds per (env, prompt, pool epoch), any M of them make a group.

The pool is the same for every miner. A miner over-generates and keeps the subset of
``group_size`` distinct seeds it likes; there are no fixed candidate groups. What the miner
controls is only WHICH seeds it submits, never what a seed draws:

* the pool digest is fixed by (order, environment, prompt, checkpoint, pool epoch, beacon);
* a seed's uniforms depend on (pool digest, seed index, token position) only -- not on the
  rollout's rank in the group, not on the other chosen seeds, not on the hotkey;
* a subset has exactly one encoding (strictly increasing seed indices), so the digest of the
  selection is the group id and is equal for two miners who chose the same subset.

Wire identities (bumped from the fixed-candidate-group design, which this replaces):

====================  ===========================  =====================================
constant              value                        was
====================  ===========================  =====================================
``CAPABILITY``        ``public-seed-pool/v3``      ``public-group-pool/v1``
``POOL_SCHEMA``       ``public-seed-pool/v3``      ``public-group-pool/v2``
``SELECTION_SCHEMA``  ``public-seed-selection/v2`` ``public-group-selection/v1``
``ROLLOUT_SCHEMA``    ``public-seed-rollout/v2``   ``public-group-rollout/v1``
``DRAW_DOMAIN``       ``public-seed-draw/v3``      ``public-group-draw/v2``
====================  ===========================  =====================================

``PROOF_VERSION`` and the commit/envelope signature domains keep their names: the signed bytes
embed the canonical selection / rollout binding, whose schema strings changed, so a signature
made for the old design cannot verify against the new one.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any, Mapping

from reliquary.protocol.release_contract import canonical_json_bytes
from reliquary.protocol.service_contract import PUBLIC_SEED_POOL, ServiceContract

POOL_SCHEMA = "public-seed-pool/v3"
SELECTION_SCHEMA = "public-seed-selection/v2"
ROLLOUT_SCHEMA = "public-seed-rollout/v2"
CAPABILITY = PUBLIC_SEED_POOL
DRAW_DOMAIN = b"public-seed-draw/v3"
PROOF_VERSION = "public-group-proof/v1"
MAX_GROUP_SIZE = 64
MAX_POOL_SEEDS = 2 * MAX_GROUP_SIZE
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class SeedPoolError(ValueError):
    pass


def _integer(value: Any, name: str, low: int, high: int) -> int:
    if type(value) is not int or not low <= value <= high:
        raise SeedPoolError(f"{name}: integer in [{low}, {high}] required")
    return value


def _sha(value: Any, name: str) -> str:
    if not isinstance(value, str) or not _SHA.fullmatch(value):
        raise SeedPoolError(f"{name}: lowercase SHA-256 required")
    return value


def _seed_subset(value: Any, *, pool_seeds: int = MAX_POOL_SEEDS, count: int | None = None) -> tuple[int, ...]:
    """The one encoding of a subset: strictly increasing plain integers in ``[0, pool_seeds)``.

    The length is checked before any element is looked at, so an oversized list costs nothing.
    """
    if type(value) not in (list, tuple):
        raise SeedPoolError("seeds: a list of seed indices required")
    if count is None:
        if not 2 <= len(value) <= MAX_GROUP_SIZE:
            raise SeedPoolError(f"seeds: between 2 and {MAX_GROUP_SIZE} seed indices required")
    elif len(value) != count:
        raise SeedPoolError(f"seeds: exactly {count} seed indices required")
    previous = -1
    for seed in value:
        _integer(seed, "seed index", 0, pool_seeds - 1)
        if seed <= previous:
            raise SeedPoolError("seeds: distinct seed indices in strictly increasing order required")
        previous = seed
    return tuple(value)


@dataclass(frozen=True, slots=True)
class PoolSelection:
    """The subset a miner submits: the pool digest and the chosen seed indices, ascending.

    Rollout ``i`` of the group is the completion drawn from ``seeds[i]``. Bounds that depend
    on the pool (exactly ``group_size`` seeds, each below ``pool_seeds``) are checked by
    ``SeedPool.validate_selection``; this type alone guarantees the canonical encoding.
    """
    pool_sha256: str
    seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        _sha(self.pool_sha256, "pool_sha256")
        object.__setattr__(self, "seeds", _seed_subset(self.seeds))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PoolSelection:
        if (not isinstance(value, dict) or set(value) != {"schema", "pool_sha256", "seeds"}
                or value["schema"] != SELECTION_SCHEMA or type(value["seeds"]) is not list):
            raise SeedPoolError("unknown public seed selection")
        return cls(value["pool_sha256"], value["seeds"])

    def to_dict(self) -> dict:
        return {"schema": SELECTION_SCHEMA, "pool_sha256": self.pool_sha256, "seeds": list(self.seeds)}

    @property
    def sha256(self) -> str:
        """The service group id: equal for every miner who chose this subset of this pool."""
        return hashlib.sha256(canonical_json_bytes(self.to_dict())).hexdigest()

    def rollout_binding(self, rollout_index: int) -> dict:
        _integer(rollout_index, "rollout_index", 0, len(self.seeds) - 1)
        return {"schema": ROLLOUT_SCHEMA, "pool_sha256": self.pool_sha256,
                "seed_index": self.seeds[rollout_index], "rollout_index": rollout_index}


@dataclass(frozen=True, slots=True)
class RolloutSeed:
    """One rollout's signed claim: which seed of which pool it drew, at which rank of its group."""
    pool_sha256: str
    seed_index: int
    rollout_index: int


def parse_rollout_binding(value: Any) -> RolloutSeed:
    if (not isinstance(value, dict) or set(value) != {"schema", "pool_sha256", "seed_index", "rollout_index"}
            or value["schema"] != ROLLOUT_SCHEMA):
        raise SeedPoolError("unknown public seed rollout binding")
    return RolloutSeed(_sha(value["pool_sha256"], "pool_sha256"),
                       _integer(value["seed_index"], "seed_index", 0, MAX_POOL_SEEDS - 1),
                       _integer(value["rollout_index"], "rollout_index", 0, MAX_GROUP_SIZE - 1))


_ANNOUNCEMENT_FIELDS = {"contract", "schedule", "checkpoint", "supported_capabilities", "pool_epoch", "pool_randomness"}


def pool_from_service_policy(value: Any, *, environment: str, prompt_idx: int,
                             checkpoint_hash: str) -> SeedPool | None:
    """Resolve the exact operator announcement for one env, refusing unsupported policies."""
    if value is None:
        return None
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict) or set(value) != _ANNOUNCEMENT_FIELDS:
        raise SeedPoolError("invalid service policy announcement")
    contract = ServiceContract.from_dict(value["contract"])
    if contract.version != 2:
        raise SeedPoolError("only service-contract/v2 announces public pools")
    supported = value["supported_capabilities"]
    if not isinstance(supported, list) or not 1 <= len(supported) <= 32 or any(not isinstance(x, str) or not 1 <= len(x) <= 256 for x in supported):
        raise SeedPoolError("bounded service capabilities required")
    contract.require_capabilities(set(supported))
    policy = contract.environment(environment)["sampling"]
    if policy["kind"] == "legacy/v1":
        return None
    if policy["kind"] != CAPABILITY:
        raise SeedPoolError("sampling policy is not implemented by this miner")
    return SeedPool.from_contract(contract, environment=environment, prompt_idx=prompt_idx,
                                  checkpoint_hash=checkpoint_hash, pool_epoch=value["pool_epoch"],
                                  randomness=value["pool_randomness"])


@dataclass(frozen=True, slots=True)
class SeedPool:
    """An operator-published pool of ``pool_seeds == 2 * group_size`` public seeds.

    The beacon and epoch belong to the authoritative pool announcement. A new
    window does not silently change an existing pool's beacon. Publication and
    renewal are the caller's responsibility; generation accepts this exact
    immutable manifest and validation compares its digest to the active one.
    Nothing a miner sends enters the digest, so a miner cannot grind beyond
    the ``pool_seeds`` streams the pool defines.
    """
    service_contract_sha256: str
    environment: str
    prompt_idx: int
    checkpoint_hash: str
    pool_epoch: int
    randomness: str
    group_size: int
    pool_seeds: int
    renewal_windows: int
    _digest: bytes = field(init=False, repr=False)

    def __post_init__(self) -> None:
        _sha(self.service_contract_sha256, "service_contract_sha256")
        if not isinstance(self.environment, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_]{0,63}", self.environment):
            raise SeedPoolError("environment: canonical environment id required")
        _integer(self.prompt_idx, "prompt_idx", 0, 2**53 - 1)
        if not isinstance(self.checkpoint_hash, str) or not 1 <= len(self.checkpoint_hash.encode()) <= 256:
            raise SeedPoolError("checkpoint_hash: nonempty bounded identity required")
        _integer(self.pool_epoch, "pool_epoch", 0, 2**53 - 1)
        _sha(self.randomness, "randomness")
        _integer(self.group_size, "group_size", 2, MAX_GROUP_SIZE)
        _integer(self.pool_seeds, "pool_seeds", 4, MAX_POOL_SEEDS)
        if self.pool_seeds != 2 * self.group_size:
            raise SeedPoolError("pool_seeds: the pool is exactly 2 x group_size seeds")
        _integer(self.renewal_windows, "renewal_windows", 1, 1000000)
        object.__setattr__(self, "_digest", hashlib.sha256(canonical_json_bytes(self.to_dict())).digest())

    @classmethod
    def from_contract(cls, contract: ServiceContract, *, environment: str, prompt_idx: int,
                      checkpoint_hash: str, pool_epoch: int, randomness: str) -> SeedPool:
        policy = contract.environment(environment)["sampling"]
        if policy["kind"] != CAPABILITY:
            raise SeedPoolError(f"only {CAPABILITY} is supported")
        return cls(contract.sha256, environment, prompt_idx, checkpoint_hash, pool_epoch, randomness,
                   policy["group_size"], policy["pool_seeds"], policy["renewal_windows"])

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SeedPool:
        names = {"service_contract_sha256", "environment", "prompt_idx", "checkpoint_hash",
                 "pool_epoch", "randomness", "group_size", "pool_seeds", "renewal_windows"}
        if not isinstance(value, dict) or set(value) != {"schema", *names} or value["schema"] != POOL_SCHEMA:
            raise SeedPoolError("unknown public seed pool")
        return cls(**{key: value[key] for key in names})

    def to_dict(self) -> dict:
        return {"schema": POOL_SCHEMA, "service_contract_sha256": self.service_contract_sha256,
                "environment": self.environment, "prompt_idx": self.prompt_idx,
                "checkpoint_hash": self.checkpoint_hash, "pool_epoch": self.pool_epoch,
                "randomness": self.randomness, "group_size": self.group_size,
                "pool_seeds": self.pool_seeds, "renewal_windows": self.renewal_windows}

    @property
    def sha256(self) -> str:
        return self._digest.hex()

    def selection(self, seeds) -> PoolSelection:
        """The selection of ``seeds``: exactly ``group_size`` distinct pool seeds, ascending."""
        return PoolSelection(self.sha256, _seed_subset(seeds, pool_seeds=self.pool_seeds, count=self.group_size))

    def validate_selection(self, selection: PoolSelection, *, rollout_count: int) -> None:
        if not isinstance(selection, PoolSelection) or selection.pool_sha256 != self.sha256:
            raise SeedPoolError("selection is not bound to the active pool")
        _seed_subset(selection.seeds, pool_seeds=self.pool_seeds, count=self.group_size)
        if type(rollout_count) is not int or rollout_count != self.group_size:
            raise SeedPoolError("a group is exactly one rollout per chosen seed")

    def validate_rollout_binding(self, binding: Any, selection: PoolSelection, rollout_index: int) -> None:
        self.validate_selection(selection, rollout_count=self.group_size)
        _integer(rollout_index, "rollout_index", 0, self.group_size - 1)
        expected = selection.rollout_binding(rollout_index)
        if not isinstance(binding, dict) or binding != expected:
            raise SeedPoolError("rollout identity differs from the selected seed at its position")
        # Dict equality considers True == 1; exact canonical bytes do not.
        if canonical_json_bytes(binding) != canonical_json_bytes(expected):
            raise SeedPoolError("rollout identity must use canonical integer fields")

    def uniform(self, seed_index: int, position: int) -> float:
        """The forced draw of seed ``seed_index`` at completion position ``position``.

        ``sha256(DRAW_DOMAIN || pool digest (32) || seed_index (u32 BE) || position (u32 BE))``,
        top 53 bits of the first 8 bytes, divided by 2**53. No rank, no hotkey, no selection.
        """
        _integer(seed_index, "seed_index", 0, self.pool_seeds - 1)
        _integer(position, "position", 0, 2**32 - 1)
        message = DRAW_DOMAIN + self._digest + seed_index.to_bytes(4, "big") + position.to_bytes(4, "big")
        # 53 bits ensure the floating-point result can never round up to 1.0.
        bits = int.from_bytes(hashlib.sha256(message).digest()[:8], "big") >> 11
        return bits / 2**53


def validate_rollout_selection(pool: SeedPool, selection: PoolSelection,
                               commits: list[dict], *, signed_episodes: bool = False) -> None:
    """Refuse, before any proof work, a group that is not one rollout per chosen seed, in order.

    Rollout ``i`` must carry exactly ``selection.rollout_binding(i)``; since the selection is
    strictly increasing, two rollouts can never claim the same seed. ``signed_episodes`` (plan 2C):
    the env's groups are signed episodes, so every rollout carries one, drawn from its own chosen
    seed; otherwise no rollout may carry any episode.
    """
    pool.validate_selection(selection, rollout_count=len(commits))
    for index, commit in enumerate(commits):
        metadata = commit.get("rollout")
        if not isinstance(metadata, dict):
            raise SeedPoolError("public seed pool requires single-turn rollouts")
        episode = metadata.get("episode")
        if signed_episodes:
            from reliquary.protocol.submission import is_signed_episode

            if not is_signed_episode(metadata):
                raise SeedPoolError("this environment takes signed episodes")
            seed = episode.get("seed_index")
            if type(seed) is not int or seed != selection.seeds[index]:
                raise SeedPoolError("an episode is not the one of its chosen seed")
        elif episode is not None:
            raise SeedPoolError("public seed pool requires single-turn rollouts")
        pool.validate_rollout_binding(metadata.get("seed_pool"), selection, index)
