"""Customer environments: a customer's verifiers v1 package, built into an image that runs only in the
sandboxes of the order using it (docs/design/2026-11-26-customer-environments-design.md).

Nothing here imports a customer's code: validators and the admin host read frozen prompts and receive scores.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

PREFIX = "customer:"
_NAME = r"[a-z0-9][a-z0-9-]{0,62}"
_ACCOUNT = r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}"
_ENV_ID_RE = re.compile(rf"\A{PREFIX}(?P<account>{_ACCOUNT})/(?P<name>{_NAME})@(?P<sha256>[0-9a-f]{{64}})\Z")
NAME_RE = re.compile(rf"\A{_NAME}\Z")
ACCOUNT_RE = re.compile(rf"\A{_ACCOUNT}\Z")

USES = ("rl", "batch", "eval")


@dataclass(frozen=True, slots=True)
class EnvId:
    account: str
    name: str
    sha256: str

    def __str__(self) -> str:
        return f"{PREFIX}{self.account}/{self.name}@{self.sha256}"


def is_customer_env(name: object) -> bool:
    return isinstance(name, str) and name.startswith(PREFIX)


def parse_env_id(text: str) -> EnvId:
    match = _ENV_ID_RE.match(text) if isinstance(text, str) else None
    if match is None:
        raise ValueError(f"{text!r} is not customer:<account>/<name>@<sha256>")
    return EnvId(match["account"], match["name"], match["sha256"])


def env_id(account: str, name: str, sha256: str) -> EnvId:
    return parse_env_id(f"{PREFIX}{account}/{name}@{sha256}")


def check_order_env(env: str, *, account: str, product: str, uses: tuple[str, ...] | list[str]) -> EnvId:
    """The order's customer env, or ValueError: only its own account's, only for a use it qualified for."""
    parsed = parse_env_id(env)
    if parsed.account != account:
        raise ValueError("customer_env_not_yours")
    if product not in USES:
        raise ValueError(f"unknown product {product!r}")
    if product not in uses:
        raise ValueError(f"customer_env_not_qualified_for: {product}")
    return parsed


__all__ = ["ACCOUNT_RE", "NAME_RE", "PREFIX", "USES", "EnvId", "check_order_env", "env_id",
           "is_customer_env", "parse_env_id"]
