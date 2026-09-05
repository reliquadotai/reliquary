"""Read-only Hugging Face quota guard for append-only active training.

Active runs never rewrite or prune Hub history. This guard only prevents the
next upload when it would cross an operator-configured organization ceiling.
Finished-run compaction is a separate, explicit operator command.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Any, Mapping


@dataclass(frozen=True)
class HfStoragePolicy:
    """Optional ceiling for visible organization storage."""

    freeze_bytes: int | None = None

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> "HfStoragePolicy":
        values = os.environ if env is None else env
        raw = str(values.get("RELIQUARY_HF_STORAGE_FREEZE_TB", "")).strip()
        if not raw:
            return cls()
        freeze_tb = float(raw)
        if not math.isfinite(freeze_tb) or freeze_tb <= 0:
            raise ValueError(
                "RELIQUARY_HF_STORAGE_FREEZE_TB must be a finite positive number"
            )
        return cls(freeze_bytes=int(freeze_tb * 1_000_000_000_000))

    @property
    def enabled(self) -> bool:
        return self.freeze_bytes is not None


class HfStorageGuard:
    """Fail closed before an append-only upload would exceed the ceiling."""

    def __init__(
        self,
        *,
        policy: HfStoragePolicy | None = None,
        api: Any | None = None,
    ) -> None:
        self.policy = policy or HfStoragePolicy.from_env()
        if api is None and self.policy.enabled:
            from huggingface_hub import HfApi

            api = HfApi()
        self.api = api

    def organization_storage_bytes(self, namespace: str) -> int:
        if self.api is None:
            return 0
        total = 0
        listings = (
            (self.api.list_models(author=namespace), self.api.model_info),
            (self.api.list_datasets(author=namespace), self.api.dataset_info),
            (self.api.list_spaces(author=namespace), self.api.space_info),
        )
        for listing, info_fn in listings:
            for item in listing:
                used = getattr(item, "used_storage", None)
                if used is None:
                    used = getattr(
                        info_fn(str(getattr(item, "id"))),
                        "used_storage",
                        None,
                    )
                if type(used) is not int or used < 0:
                    raise RuntimeError("Hugging Face storage usage is unavailable or invalid")
                total += used
        return total

    def assert_upload_allowed(
        self,
        *,
        repo_id: str,
        upload_bytes: int,
    ) -> int | None:
        ceiling = self.policy.freeze_bytes
        if ceiling is None:
            return None
        if upload_bytes <= 0:
            raise ValueError("upload_bytes must be positive")
        namespace = repo_id.split("/", 1)[0]
        used = self.organization_storage_bytes(namespace)
        projected = used + int(upload_bytes)
        if projected >= ceiling:
            raise RuntimeError(
                "Hugging Face storage safety ceiling would be exceeded: "
                f"namespace={namespace} used={used} upload={upload_bytes} "
                f"projected={projected} ceiling={ceiling}; active history "
                "was not changed—finish or manually compact a sealed run "
                "before resuming publication"
            )
        return used


__all__ = ["HfStorageGuard", "HfStoragePolicy"]
