"""Version-pinned checkpoint evaluation and dashboard publication.

This package is deliberately independent from the validator event loop.  An
evaluation worker may read published checkpoints and write dashboard evidence,
but it never participates in miner admission, scoring, or training.
"""

from reliquary.eval_dashboard.config import (
    build_effective_config,
    canonical_json_bytes,
    config_hash,
)
from reliquary.eval_dashboard.models import EvalConfig

__all__ = [
    "EvalConfig",
    "build_effective_config",
    "canonical_json_bytes",
    "config_hash",
]
