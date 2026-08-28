#!/usr/bin/env python3
"""Emit stable Episode v1 JSONL for cross-host byte comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from reliquary.environment.agentic.goldens import episode_golden_row  # noqa: E402
from reliquary.environment.agentic.suite import (  # noqa: E402
    BUILTIN_EPISODE_ENVIRONMENTS,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks-per-env", type=int, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.tasks_per_env <= 0:
        parser.error("--tasks-per-env must be positive")
    rendered = "\n".join(
        json.dumps(
            episode_golden_row(environment, index),
            sort_keys=True,
            separators=(",", ":"),
        )
        for environment in BUILTIN_EPISODE_ENVIRONMENTS
        for index in range(args.tasks_per_env)
    ) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
