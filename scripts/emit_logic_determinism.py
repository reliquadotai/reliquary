#!/usr/bin/env python3
"""Emit a tokenizer-neutral ``reliquarylogic_v1`` corpus for cross-host diffing.

CI regenerates this on two Ubuntu releases and requires byte-identical
output. Digests rather than raw prompts keep the artifact small while still
failing on any drift in generation or in the verifier spec.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys

from reliquary.environment.logic_tasks import generate_logic_task
from reliquary.environment.reliquarylogic import ReliquaryLogicEnvironment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=int, default=4096)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)

    environment = ReliquaryLogicEnvironment()
    with open(args.output, "w", encoding="utf-8") as handle:
        for index in range(args.start, args.start + args.tasks):
            problem = environment.get_problem(index)
            task = generate_logic_task(index)
            accepted = json.dumps(
                {"result": task.expected}, separators=(",", ":")
            )
            row = {
                "index": index,
                "id": problem["id"],
                "family": task.family,
                "operation_id": problem["operation_id"],
                "difficulty": problem["difficulty"],
                "prompt_sha256": hashlib.sha256(
                    problem["prompt"].encode("utf-8")
                ).hexdigest(),
                "target_sha256": hashlib.sha256(
                    problem["ground_truth"].encode("utf-8")
                ).hexdigest(),
                # Proves the checker agrees with the generator, not just that
                # the bytes match: a verifier drift would flip this to 0.0.
                "reference_reward": environment.compute_reward(
                    problem, accepted
                ),
            }
            handle.write(
                json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
