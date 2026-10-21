#!/usr/bin/env python3
"""Entrypoint for an offline, profile-qualified vision training allocation."""

import argparse
from pathlib import Path

from reliquary.vision.runner import supervise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", type=Path, default=Path("/job"))
    parser.add_argument("--model", type=Path, default=Path("/model"))
    parser.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    supervise(args.job, args.model, args.output)
