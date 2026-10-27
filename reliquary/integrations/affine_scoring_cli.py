"""Prepare and replay a fixed V9 compute unit without score-signing credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from threading import Event

from . import affine_scoring as scoring


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "check",
            "run",
            "verify",
            "export",
            "compare",
            "reconcile",
            "rehearse",
        ),
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--workspace", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument(
        "--fixture",
        action="store_true",
        help="CPU rehearsal only; cannot execute a native job",
    )
    args = parser.parse_args(argv)

    def need(*names):
        if any(getattr(args, name) is None for name in names):
            parser.error("required: " + ", ".join("--" + name for name in names))

    try:
        if args.command == "prepare":
            need("config", "out")
            result = scoring.prepare(args.config, args.out)
        elif args.command == "rehearse":
            from .affine_scoring_fixture import rehearse

            need("out")
            result = rehearse(args.out)
        elif args.command == "reconcile":
            need("workspace")
            result = scoring.reconcile(args.workspace)
        else:
            need("bundle")
            if args.command == "check":
                job, context = scoring.load_job(args.bundle)
                complete = False
                if args.checkpoint:
                    complete = (
                        scoring.inventory(args.checkpoint, max_bytes=256 * 1024**3)
                        == job["checkpointFiles"]
                    )
                    scoring.require(complete, "checkpoint files differ")
                result = {
                    **scoring.summary(job, "metadata_checked"),
                    **context["outputDimensions"],
                    "checkpointBytesVerified": complete,
                }
            elif args.command == "run":
                need("workspace")
                stopped = Event()
                previous = signal.signal(signal.SIGTERM, lambda *_: stopped.set())
                try:
                    result = scoring.run(
                        args.bundle,
                        args.workspace,
                        checkpoint=args.checkpoint,
                        max_seconds=args.max_seconds,
                        fixture=args.fixture,
                        stop_event=stopped,
                    )
                finally:
                    signal.signal(signal.SIGTERM, previous)
            elif args.command == "verify":
                need("workspace")
                result = scoring.verify(args.bundle, args.workspace)
            elif args.command == "compare":
                need("workspace", "reference", "atol", "rtol")
                result = scoring.compare(
                    args.bundle,
                    args.workspace,
                    args.reference,
                    atol=args.atol,
                    rtol=args.rtol,
                )
            else:
                need("workspace", "out")
                result = scoring.export(args.bundle, args.workspace, args.out)
        print(json.dumps(result, sort_keys=True))
        return 1 if result.get("comparisonPassed") is False else 0
    except Exception:
        print(
            json.dumps(
                {
                    "stage": "unavailable",
                    "error": "Scoring operation refused; inspect private inputs and state.",
                }
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
