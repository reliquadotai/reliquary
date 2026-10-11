"""Any verifiers env served through the sandbox's verifiers bridge, as a signed-episode env of an RL run
(no per-env code). The order's launch episode envs and the envs added to a running run are both built here
from their PUBLIC catalogue (package, RL split, gateway options, image lock: the order entry's ``catalogue``,
or an added schedule row's), by the validator and by the miner alike: the task source reads the very
``VerifiersEnv`` a gateway serves (same package, same options), so the prompt, the image and the limits a
session token names are the machine's.

Identity. ``identity`` is what record 0 names (``bridged_env_package_of``: ``name==version+g<hash>.vfb<N>``),
``options_sha256`` what the gateway publishes for the env (``env_options_sha256``), ``lock_sha256`` its
image lock as the gateway publishes it (``env_image_locks``), and the manifest digest
``sha256({identity, options, lock, rl_split, rows})`` is the env's ``version`` in the order.

``reliquary_sandbox_verifiers`` and the env packages are optional: imported inside functions only."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from reliquary.sandbox.tasks import ResolvedTask, SweTaskResolver

# {import module: the identity its files had when this process first built an env from it}. A module stays loaded
# (``sys.modules``) after its files change on disk: a later build that reads another identity from disk would serve
# the old code under the new identity, so it is refused until the validator restarts.
_LOADED_IDENTITIES: dict[str, str] = {}
_LOADED_GUARD = threading.Lock()


def loaded_identity_problem(module_name: str | None, identity: str,
                            loaded: dict[str, str] | None = None) -> str | None:
    """None when ``module_name`` was first built at ``identity`` in this process (recorded now if never built);
    the reason otherwise. ``loaded``: the record (default: this process's)."""
    if module_name is None:
        return None
    record = _LOADED_IDENTITIES if loaded is None else loaded
    with _LOADED_GUARD:
        first = record.setdefault(str(module_name), str(identity))
    if first != identity:
        return (f"module {module_name} is loaded at {first} in this process and is {identity} on disk: "
                f"validator restart required")
    return None


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


@dataclass(frozen=True)
class BridgedEnvConfig:
    environment: str
    package: str                     # name==version, the distribution the gateway serves
    rl_split: str                    # the served split name (the entry's episode.split)
    options: Mapping[str, Any] = field(default_factory=dict)   # the gateway's episode_env_options[env]
    image_lock_sha256: str | None = None   # the catalogue's lock digest: the install must have it

    @property
    def entry(self) -> str:
        return f"verifiers:{self.package}"

    @classmethod
    def from_catalogue(cls, name: str, catalogue: Mapping[str, Any]) -> BridgedEnvConfig:
        """From a catalogue the contract already validated (``service_contract.validate_catalogue``)."""
        return cls(environment=name, package=catalogue["package"], rl_split=catalogue["rl_split"],
                   options=dict(catalogue["options"]), image_lock_sha256=catalogue.get("image_lock_sha256"))


def image_lock_sha256(images: Mapping[str, str]) -> str:
    """The image lock's digest, computed as the sandbox computes it (``ImageLock.sha256``)."""
    return hashlib.sha256(_canonical({"version": 1, "images": dict(sorted(images.items()))})).hexdigest()


def bridged_manifest_sha256(*, identity: str, options_sha256: str, lock_sha256: str, rl_split: str,
                            rows: int) -> str:
    return hashlib.sha256(_canonical({"identity": identity, "options_sha256": options_sha256,
                                      "image_lock_sha256": lock_sha256, "rl_split": rl_split,
                                      "rows": int(rows)})).hexdigest()


class BridgedEpisodeSource:
    """``EpisodeTaskSource`` of one bridged env's RL split. Thread-safe.

    Each task resolves as ``SweTaskResolver`` resolves a bridged task (one rule for the image, the declared
    limits, the CPUs rounded up and the prepare timeout), stamped with this env's options digest."""

    def __init__(self, config: BridgedEnvConfig, *, env: Any = None, identity: str | None = None,
                 cache_size: int = 8192) -> None:
        from reliquary_sandbox_verifiers.loading import options_sha256

        if env is None:
            from reliquary_sandbox_verifiers.loading import VerifiersEnv

            env = VerifiersEnv(config.entry, dict(config.options))
        from_disk = identity is None
        if from_disk:
            from reliquary_sandbox_service.episodes import bridged_env_package_of

            identity = bridged_env_package_of(env.module)
        self.config = config
        self._env = env
        self.identity = identity
        self.options_sha256 = options_sha256(dict(config.options))
        self.lock_sha256 = image_lock_sha256(env.lock.images)
        if config.image_lock_sha256 is not None and config.image_lock_sha256 != self.lock_sha256:
            raise ValueError(f"{config.environment}: the catalogue's image lock {config.image_lock_sha256} is not the "
                             f"installed package's {self.lock_sha256}")
        self.lock_images = frozenset(env.lock.images.values())     # the distinct images the lock pins
        module = getattr(env, "module", None)
        self.module_name = getattr(module, "__name__", module)     # the import module; None in fakes
        if from_disk:
            problem = loaded_identity_problem(self.module_name, identity)
            if problem is not None:
                raise ValueError(f"{config.environment}: {problem}")
        self._resolver = SweTaskResolver(config.rl_split, env.sandbox_task, cache_size=cache_size,
                                         options_sha256=self.options_sha256, lock_sha256=self.lock_sha256)
        self._guard = threading.Lock()
        self._rows: int | None = None

    def __len__(self) -> int:
        with self._guard:
            if self._rows is None:
                taskset = self._env.taskset(self.config.rl_split)
                try:
                    rows = len(taskset)
                except TypeError:                      # a taskset without len: counted once
                    rows = sum(1 for _ in taskset)
                self._rows = int(rows)
            return self._rows

    def prompt(self, index: int) -> str:
        prompt = self._env.task(self.config.rl_split, int(index)).data.prompt
        if not isinstance(prompt, str):
            raise TypeError(f"{self.config.environment}: task {index}'s prompt is not text")
        return prompt

    async def resolve(self, index: int) -> ResolvedTask:
        return await self._resolver.resolve(int(index))

    def task_image(self, index: int) -> str:
        """The image of one task (blocking: for activation's servable floor, in a worker thread)."""
        return self._resolver._cached(int(index)).image

    def installed_identity(self) -> str:
        """The identity of the package installed NOW (its files on disk): activation checks that adding an env
        did not move an active one's."""
        from reliquary_sandbox_service.episodes import bridged_env_package_of

        return bridged_env_package_of(self._env.module)

    def manifest_sha256(self) -> str:
        return bridged_manifest_sha256(identity=self.identity, options_sha256=self.options_sha256,
                                       lock_sha256=self.lock_sha256, rl_split=self.config.rl_split, rows=len(self))


def unbuildable():
    raise TypeError("a bridged signed-episode env is built by bridged_episode.build_environment, not by its spec")


def catalogue_spec(name: str, manifest_sha256: str):
    from reliquary.environment.registry import EnvironmentSpec

    return EnvironmentSpec(
        name=name, factory_path="reliquary.environment.bridged_episode:unbuildable",
        scorer_path="reliquary.environment.signed_episode:signed_episode_scorer",
        validator_authoritative_reward=True, admission_resource_class="sandbox",
        termination_policy="eos_or_cap", final_answer_policy="text", reward_lattice_policy="final-record/v1",
        attainable_rewards=(), contract_version="reliquary/signed-episode/v1", interaction_mode="signed_episode",
        environment_manifest_sha256=manifest_sha256)


def build_environment(config: BridgedEnvConfig, *, env: Any = None, identity: str | None = None):
    from reliquary.environment.signed_episode import SignedEpisodeEnvironment

    return SignedEpisodeEnvironment(config.environment, BridgedEpisodeSource(config, env=env, identity=identity))


def describe(*, name: str, package: str, rl_split: str, options: Mapping[str, Any], episode: Mapping[str, Any],
             cooldown_windows: int, attestation: bytes | None = None, env: Any = None,
             identity: str | None = None) -> dict:
    """The env's v2 entry and its public catalogue, computed from the installed package: the operator reviews
    it, then puts ``{**environment, "catalogue": catalogue}`` in the order (a launch env), or passes the block to
    the schedule CLI (an env added to the running run: ``attestation`` records the conformance report's
    digest). ``episode``: the episode block's run choices (sandbox_env, tools, max_turns, max_tokens_per_turn,
    max_episode_tokens, budgets); kind, split and env_package are filled in. ``env``/``identity``: tests."""
    from reliquary.constants import M_ROLLOUTS
    from reliquary.protocol.service_contract import EPISODE_CAPABILITY, PUBLIC_SEED_POOL

    config = BridgedEnvConfig(environment=name, package=package, rl_split=rl_split, options=dict(options))
    source = BridgedEpisodeSource(config, env=env, identity=identity)
    environment = {
        "version": source.manifest_sha256(),
        "dataset": {"id": f"{package.replace('==', '@')}:{rl_split}", "rows": len(source)},
        "sampling": {"kind": PUBLIC_SEED_POOL, "group_size": M_ROLLOUTS, "pool_seeds": 2 * M_ROLLOUTS,
                     "renewal_windows": 1},
        "exploration": 1, "missing_box": "graded", "cooldown_windows": int(cooldown_windows), "share_bps": 0,
        "episode": {**dict(episode), "kind": EPISODE_CAPABILITY, "split": rl_split, "env_package": source.identity},
    }
    catalogue = {"package": package, "options": dict(options), "rl_split": rl_split,
                 "image_lock_sha256": source.lock_sha256}
    if attestation is not None:
        catalogue["conformance_sha256"] = hashlib.sha256(attestation).hexdigest()
    return {"environment": environment, "catalogue": catalogue}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m reliquary.environment.bridged_episode")
    sub = parser.add_subparsers(dest="command", required=True)
    cmd = sub.add_parser("describe", help="print a bridged env's entry and public catalogue (PUBLIC: the options "
                                          "go in the order as they are; never put a credential in them)")
    cmd.add_argument("--name", required=True)
    cmd.add_argument("--package", required=True)
    cmd.add_argument("--rl-split", required=True)
    cmd.add_argument("--options", required=True, help="JSON file: the gateway's episode_env_options for the env")
    cmd.add_argument("--episode", required=True, help="JSON file: sandbox_env, tools, turn/token limits, budgets")
    cmd.add_argument("--cooldown", type=int, required=True)
    cmd.add_argument("--attestation", help="the conformance report file (its sha256 is recorded; added envs)")
    args = parser.parse_args(argv)
    block = describe(name=args.name, package=args.package, rl_split=args.rl_split,
                     options=json.loads(Path(args.options).read_text()),
                     episode=json.loads(Path(args.episode).read_text()), cooldown_windows=args.cooldown,
                     attestation=Path(args.attestation).read_bytes() if args.attestation else None)
    json.dump(block, sys.stdout, indent=2, sort_keys=True)
    print()


if __name__ == "__main__":
    main()


__all__ = ["BridgedEnvConfig", "BridgedEpisodeSource", "bridged_manifest_sha256", "build_environment",
           "catalogue_spec", "describe", "image_lock_sha256", "loaded_identity_problem"]
