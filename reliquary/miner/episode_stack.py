"""The signed-episode stack of ``mine-async``: the forced engine's parts, the env package check, one
episode played under its forced draw (``play_bound``) and the harness environment options."""
from __future__ import annotations

import asyncio
import contextlib
import gc
import logging
import re
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from reliquary.protocol.sandbox_session import SessionRefused

logger = logging.getLogger(__name__)


def parse_harness_env(values: Sequence[str], *, env_file: str | None = None) -> dict[str, str]:
    """``--harness-env-file`` (KEY=VALUE lines, ``#`` comments; where secrets belong) then
    ``--harness-env KEY=VALUE`` (repeatable; it wins over the file)."""
    parsed = {}
    if env_file:
        with open(env_file, encoding="utf-8") as handle:
            lines = [line.strip() for line in handle]
        for line in lines:
            if not line or line.startswith("#"):
                continue
            key, sep, val = line.removeprefix("export ").partition("=")
            if not sep or not key.strip():
                raise ValueError(f"{env_file}: KEY=VALUE lines only")
            parsed[key.strip()] = val.strip()
    for item in values:
        key, sep, val = item.partition("=")
        if not sep or not key.strip():
            raise ValueError(f"--harness-env takes KEY=VALUE, not {item!r}")
        parsed[key.strip()] = val
    return parsed


def installed_env_identity(package: str, pinned: str) -> str:
    """The `env_package` a gateway serving this install of `package` writes into record 0,
    in the form `pinned` takes: an env served through the verifiers bridge (`pinned` ends
    in `.vfb<bridge version>`) is `bridged_env_package_of`, an entry-point env
    `env_package_of`, both over the module `package` with `-` as `_` (the module the
    gateway imports), computed by reliquary-sandbox's own registry."""
    from reliquary_sandbox_service.episodes import bridged_env_package_of, env_package_of

    module = package.replace("-", "_")
    if re.search(r"\.vfb[0-9]+\Z", pinned):
        return bridged_env_package_of(module)
    return env_package_of(module)


def env_package_refusal(policy, *, identity_of: Callable[[str], str] | None = None,
                        need_bridge: bool = True) -> str | None:
    """Why this install cannot play the order's episodes, or None: the env package installed (record 0
    carries its identity, and the task prompts come from it) must be the order's pinned one, and the
    verifiers bridge must load."""
    import importlib

    package = policy.env_package.split("==", 1)[0]
    if identity_of is None:
        def identity_of(name: str) -> str:
            return installed_env_identity(name, policy.env_package)
    try:
        identity = identity_of(package)
    except Exception as exc:   # an unreadable install cannot be the pinned code
        return f"cannot compute the installed {package} identity ({exc})"
    if identity != policy.env_package:
        return f"{identity} is installed, the order pins {policy.env_package}"
    if need_bridge:
        try:
            importlib.import_module("reliquary_sandbox_verifiers")
        except Exception as exc:
            return f"the verifiers bridge cannot load: {exc}"
    return None


def _make_core(checkpoint_dir: str, **kwargs):
    from reliquary.miner.corpus_generate_server import VllmTurnCore

    return VllmTurnCore(checkpoint_dir, forced=True, **kwargs)


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        logger.exception("emptying the CUDA cache failed")


def _release_core(core) -> None:
    """Best effort: vLLM's in-process engine and its hidden-state capture go, so a new checkpoint fits."""
    capture = getattr(core, "_capture_cm", None)
    if capture is not None:
        with contextlib.suppress(Exception):
            capture.__exit__(None, None, None)
    for name in ("_engine", "_llm"):
        if hasattr(core, name):
            with contextlib.suppress(Exception):
                delattr(core, name)
    _free_gpu()


def _sampling():
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE

    return ACTIVE_PROTOCOL_PROFILE.sampling


def _serve(app, port: int):
    from reliquary.miner.agentic_miner import serve_loopback

    return serve_loopback(app, port)


def _sync_http(url: str):
    import httpx

    return httpx.Client(base_url=url, timeout=60.0)


@dataclass
class EpisodeStackDeps:
    """The episode stack's heavy parts (tests replace them with fakes)."""
    load_renderer: Callable[..., Any] = None              # agentic_swe.load_turn_renderer
    make_core: Callable[..., Any] = _make_core
    release_core: Callable[[Any], None] = _release_core
    sampling: Callable[[], Any] = _sampling
    serve: Callable[[Any, int], Awaitable[tuple[Any, Any]]] = _serve
    sync_http: Callable[[str], Any] = _sync_http
    package_refusal: Callable[[Any], str | None] = env_package_refusal

    def __post_init__(self) -> None:
        if self.load_renderer is None:
            from reliquary.environment.agentic_swe import load_turn_renderer

            self.load_renderer = load_turn_renderer


async def play_bound(runner, engine, draws, index: int, binding, deadline, *, label: str):
    """One episode of rollout ``index`` with its forced draw ``binding`` (a window seed's ``DrawBinding`` or a
    lease's ``LeaseDrawBinding``): ``(result, session)``, or None (no session, crash, deadline). The draw is bound
    to every generate session of the episode before its first turn and dropped at its end."""
    bound: list[str] = []
    taken: str | None = None

    def on_session(session_id: str) -> None:
        # Before the trace's first generate request: the forced engine refuses an unbound session.
        draws.bind(session_id, binding)
        bound.append(session_id)

    try:
        try:
            async with asyncio.timeout(deadline(index) if deadline is not None else None):
                result = await runner.run(index, on_session=on_session)
        except SessionRefused as refused:
            logger.info("seed %d of %s: no session (%s)", index, label, refused.reason)
            return None
        except TimeoutError:
            logger.warning("seed %d of %s passed its deadline", index, label)
            return None
        except Exception:
            logger.exception("seed %d of %s crashed", index, label)
            return None
        # The log is taken only now, at the episode's end (taking it also ends its binding).
        taken = result.session_id
        try:
            session = engine.take_session(taken) if taken else None
        except Exception:
            logger.exception("seed %d: taking its generate session failed", index)
            session = None
        return result, session
    finally:
        for session_id in dict.fromkeys(bound):
            draws.drop(session_id)
            if session_id != taken:
                drop = getattr(engine, "drop_session", None)
                if drop is not None:
                    drop(session_id)
