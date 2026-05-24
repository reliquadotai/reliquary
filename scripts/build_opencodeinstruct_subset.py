"""Build the deterministic subset of nvidia/OpenCodeInstruct.

Run once offline (typically on a beefy box with disk + network).
Filters in order:
  1. Drop rows whose reference solution did not pass all its own
     tests (average_test_score < 1.0).
  2. Parse the unit_tests column (string-encoded list) — drop on
     parse failure.
  3. Drop rows containing any test that imports/uses a non-
     deterministic stdlib module (random, time, socket, ...).
  4. Run a double-execution check on what remains (twice with
     different PYTHONHASHSEED) — drop on mismatch.
  5. Push the resulting subset to HF Hub as
     reliquadotai/opencodeinstruct-deterministic-subset.

Designed so the per-row filter functions are pure-Python and
testable without HuggingFace, network, or subprocess.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import subprocess
import sys
from typing import Optional

logger = logging.getLogger(__name__)


# Conservative regex: any of these tokens anywhere in the test code
# disqualifies the row. False positives are fine — we have 5M rows
# and only need ~2-3M deterministic ones.
_NONDET_PATTERNS = re.compile(
    r"\b(?:import\s+(?:random|time|datetime|socket|urllib|requests|os|"
    r"subprocess|threading|multiprocessing|asyncio|signal|select)\b"
    r"|from\s+(?:random|time|datetime|socket|urllib|requests|os|"
    r"subprocess|threading|multiprocessing|asyncio|signal|select)\s+import"
    r"|\brandom\.|\btime\.|\bdatetime\.|\bsocket\.|\burllib\.|\brequests\."
    r"|\bos\.environ|\bsubprocess\.|\bthreading\.|\bmultiprocessing\.)"
)


def parse_unit_tests(raw: str) -> Optional[list[str]]:
    """Parse the string-encoded list of tests. Return None on failure."""
    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(parsed, list):
        return None
    if not all(isinstance(t, str) for t in parsed):
        return None
    return parsed


def has_nondeterministic_pattern(test_src: str) -> bool:
    return _NONDET_PATTERNS.search(test_src) is not None


def filter_tests(tests: list[str]) -> list[str]:
    """Keep only tests free of non-deterministic patterns."""
    return [t for t in tests if not has_nondeterministic_pattern(t)]


def keep_row(row: dict) -> bool:
    """Stage-1 filter: reference solution must pass all its own tests."""
    return float(row.get("average_test_score", 0.0)) >= 1.0


def double_execute(code: str, tests: list[str]) -> bool:
    """Run (code, tests) twice with different PYTHONHASHSEEDs.

    Returns True iff both runs yield the same passed count.
    """
    runner = (
        "import json,sys\n"
        "data=json.loads(sys.stdin.read())\n"
        "ns={}\n"
        "try: exec(data['code'], ns)\n"
        "except: pass\n"
        "p=0\n"
        "for t in data['tests']:\n"
        "    try: exec(t, dict(ns)); p+=1\n"
        "    except: pass\n"
        "print(p)\n"
    )
    payload = json.dumps({"code": code, "tests": tests})
    out_seed0 = subprocess.run(
        [sys.executable, "-c", runner], input=payload, capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": "0"}, timeout=30,
    )
    out_seed1 = subprocess.run(
        [sys.executable, "-c", runner], input=payload, capture_output=True, text=True,
        env={**os.environ, "PYTHONHASHSEED": "1"}, timeout=30,
    )
    return out_seed0.stdout.strip() == out_seed1.stdout.strip()


def process_row(row: dict) -> Optional[dict]:
    """Apply all filters to one row. Return the kept row (with
    `unit_tests_parsed` added) or None to drop."""
    if not keep_row(row):
        return None
    tests = parse_unit_tests(row.get("unit_tests", ""))
    if tests is None:
        return None
    kept_tests = filter_tests(tests)
    if not kept_tests:
        return None
    if not double_execute(row.get("output", ""), kept_tests):
        return None
    return {
        "input": row["input"],
        "output": row.get("output", ""),
        "unit_tests_parsed": kept_tests,
        "id": row.get("id", ""),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--target-repo", default="reliquadotai/opencodeinstruct-deterministic-subset")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Cap on rows to process — for dry-runs.")
    parser.add_argument("--push", action="store_true",
                        help="Push to HF Hub (requires HF_TOKEN).")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    import datasets as hf
    ds = hf.load_dataset(args.source, split="train", streaming=True)

    kept = []
    i = -1
    for i, row in enumerate(ds):
        if args.max_rows is not None and i >= args.max_rows:
            break
        out = process_row(row)
        if out:
            kept.append(out)
        if i % 1000 == 0:
            logger.info("processed=%d kept=%d", i, len(kept))

    logger.info("final: processed=%d kept=%d", i + 1, len(kept))
    out_ds = hf.Dataset.from_list(kept)

    if args.push:
        token = os.environ.get("HF_TOKEN")
        if not token:
            raise RuntimeError("HF_TOKEN env var is required to push.")
        out_ds.push_to_hub(args.target_repo, token=token, private=False)
        logger.info("pushed %d rows to %s", len(kept), args.target_repo)
    else:
        out_ds.save_to_disk("./opencodeinstruct-subset")
        logger.info("saved locally to ./opencodeinstruct-subset")


if __name__ == "__main__":
    main()
