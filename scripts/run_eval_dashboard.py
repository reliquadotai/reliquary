#!/usr/bin/env python3
"""Operate the version-pinned Reliquary eval-dashboard worker."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time

from reliquary.eval_dashboard.config import load_config
from reliquary.eval_dashboard.store import FileObjectStore, S3ObjectStore
from reliquary.eval_dashboard.worker import (
    CheckpointTarget,
    EvalWorker,
    check_freshness,
    discover_checkpoint,
    send_alert,
)


def _store(args):
    if args.store_root:
        return FileObjectStore(args.store_root)
    return S3ObjectStore.from_env()


def _worker(args) -> EvalWorker:
    return EvalWorker(
        config=load_config(args.config),
        store=_store(args),
        math_holdout_path=args.math_holdout,
        math_review_path=args.math_review,
        code_holdout_path=args.code_holdout,
        code_review_path=args.code_review,
        state_dir=args.state_dir,
        device=args.device,
        attention_implementation=args.attention_implementation,
    )


def _add_store(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--store-root",
        help="filesystem store for dry-runs; omit to use RELIQUARY_EVAL_R2_*",
    )


def _add_worker(parser: argparse.ArgumentParser) -> None:
    _add_store(parser)
    parser.add_argument("--config", required=True)
    parser.add_argument("--math-holdout", required=True)
    parser.add_argument("--math-review", required=True)
    parser.add_argument("--code-holdout", required=True)
    parser.add_argument("--code-review", required=True)
    parser.add_argument("--state-dir", default="/var/lib/reliquary-eval")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-implementation", default="flash_attention_2")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    subparsers = parser.add_subparsers(dest="command", required=True)

    once = subparsers.add_parser(
        "once", help="evaluate the current validator checkpoint"
    )
    _add_worker(once)
    once.add_argument("--validator-url", required=True)

    target = subparsers.add_parser(
        "target", help="evaluate one explicitly pinned checkpoint"
    )
    _add_worker(target)
    target.add_argument("--model-repo-id", required=True)
    target.add_argument("--revision", required=True)
    target.add_argument("--checkpoint-n", type=int, required=True)
    target.add_argument("--window", type=int, default=0)

    watch = subparsers.add_parser("watch", help="poll and evaluate unseen checkpoints")
    _add_worker(watch)
    watch.add_argument("--validator-url", required=True)
    watch.add_argument("--poll-seconds", type=int, default=300)

    replay = subparsers.add_parser(
        "replay", help="re-run a bounded current-checkpoint slice without publishing"
    )
    _add_worker(replay)
    replay.add_argument("--validator-url", required=True)
    replay.add_argument("--n-prompts", type=int, default=32)
    replay.add_argument("--tolerance", type=float, default=0.02)

    check = subparsers.add_parser("check", help="CPU-only freshness/provenance check")
    _add_store(check)
    check.add_argument("--validator-url", required=True)
    check.add_argument("--expected-repo-id", required=True)
    check.add_argument("--max-age-seconds", type=int, default=21600)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        if args.command == "check":
            target = discover_checkpoint(args.validator_url)
            if target.repo_id != args.expected_repo_id:
                raise RuntimeError(
                    "validator checkpoint repository differs from the configured lineage"
                )
            result = check_freshness(
                _store(args),
                expected_repo_id=args.expected_repo_id,
                expected_revision=target.revision,
                expected_checkpoint_n=target.checkpoint_n,
                max_age_seconds=args.max_age_seconds,
            )
            print(json.dumps(result, sort_keys=True))
            if result["status"] != "fresh":
                send_alert({"event": "eval_dashboard_overdue", **result})
                return 2
            return 0

        worker = _worker(args)
        if args.command == "once":
            target = discover_checkpoint(args.validator_url)
            result = worker.run_once(target)
            print(result.model_dump_json())
            return 0
        if args.command == "target":
            result = worker.run_once(
                CheckpointTarget(
                    repo_id=args.model_repo_id,
                    revision=args.revision,
                    checkpoint_n=args.checkpoint_n,
                    observed_window=args.window,
                )
            )
            print(result.model_dump_json())
            return 0
        if args.command == "replay":
            target = discover_checkpoint(args.validator_url)
            report = worker.replay_target(
                target,
                n_prompts=args.n_prompts,
                tolerance=args.tolerance,
            )
            print(json.dumps(report, sort_keys=True))
            return 0 if report["status"] == "passed" else 3
        if args.command == "watch":
            if args.poll_seconds < 60:
                raise ValueError("poll-seconds must be at least 60")
            while True:
                try:
                    target = discover_checkpoint(args.validator_url)
                    worker.run_once(target)
                except KeyboardInterrupt:
                    return 130
                except Exception as exc:
                    logging.exception("eval-dashboard watch cycle failed")
                    send_alert(
                        {
                            "event": "eval_dashboard_watch_failed",
                            "error_type": type(exc).__name__,
                            "message": str(exc)[:2000],
                        }
                    )
                time.sleep(args.poll_seconds)
        return 2
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        logging.exception("eval-dashboard command failed")
        payload = {
            "event": "eval_dashboard_command_failed",
            "command": args.command,
            "error_type": type(exc).__name__,
            "message": str(exc)[:2000],
        }
        send_alert(payload)
        print(json.dumps(payload, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
