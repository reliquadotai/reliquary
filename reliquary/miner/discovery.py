"""Validator discovery for miners: the first axon advertised by a hotkey holding ``validator_permit``."""

from __future__ import annotations

import logging
from typing import Any

from reliquary.constants import VALIDATOR_HTTP_PORT

logger = logging.getLogger(__name__)


class NoValidatorFoundError(RuntimeError):
    """No metagraph entry advertises a usable validator endpoint."""


def discover_validator_url(metagraph: Any, port: int = VALIDATOR_HTTP_PORT) -> str:
    """Return the HTTP URL of the first validator advertised on the metagraph (``discover_validator``)."""
    return discover_validator(metagraph, port)[0]


def discover_validator(metagraph: Any, port: int = VALIDATOR_HTTP_PORT) -> tuple[str, str | None]:
    """The HTTP URL and the hotkey of the first validator advertised on the metagraph.

    Picks the first uid with validator_permit=True and an axon IP that isn't
    the unset placeholder. Multi-validator coordination is out of scope; this
    deliberately picks ONE validator. The hotkey is ``metagraph.hotkeys[uid]``
    (None when the metagraph carries none).
    """
    permits = getattr(metagraph, "validator_permit", None)
    axons = getattr(metagraph, "axons", None)
    if permits is None or axons is None:
        raise NoValidatorFoundError(
            "metagraph missing validator_permit or axons attributes"
        )
    hotkeys = list(getattr(metagraph, "hotkeys", None) or ())
    for uid, (permit, axon) in enumerate(zip(permits, axons)):
        if not permit:
            continue
        ip = getattr(axon, "ip", None)
        if not ip or ip in ("0.0.0.0", ""):
            continue
        # Use the validator's own port if it's set; fall back to the protocol default.
        axon_port = getattr(axon, "port", None) or port
        hotkey = hotkeys[uid] if uid < len(hotkeys) else getattr(axon, "hotkey", None)
        return f"http://{ip}:{axon_port}", hotkey
    raise NoValidatorFoundError("no validator with permit and routable axon")


