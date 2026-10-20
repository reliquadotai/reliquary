"""Prepare media, lock local snapshots, or execute a bounded platform job."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Validate and materialize inline PNG JSONL")
    prepare.add_argument("dataset", type=Path)
    prepare.add_argument("output", type=Path)
    lock = commands.add_parser("lock-model", help="Hash an existing model/weights snapshot")
    lock.add_argument("--model", type=Path, required=True)
    lock.add_argument("--model-id", required=True)
    lock.add_argument("--revision", required=True)
    lock.add_argument("--processor-revision", required=True)
    for name in ("run", "_execute"):
        run = commands.add_parser(name)
        run.add_argument("--job", type=Path, default=Path("/job"))
        run.add_argument("--model", type=Path, default=Path("/model"))
        run.add_argument("--output", type=Path, default=Path("/output"))
    args = parser.parse_args()
    from reliquary.vision import runner
    if args.command == "lock-model":
        print(json.dumps(runner.lock_model(args.model, args.model_id, args.revision,
                                           args.processor_revision)))
    elif args.command == "prepare":
        from reliquary.vision.data import prepare_jsonl
        print(prepare_jsonl(args.dataset, args.output))
    elif args.command == "run":
        runner.supervise(args.job, args.model, args.output)
    else:
        runner.execute(args.job, args.model, args.output)


if __name__ == "__main__":
    main()
