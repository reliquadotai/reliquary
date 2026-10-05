"""Small operator entry point; no SN81 inference or validator imports."""
import argparse
import json

from .affine import AffineRunner, delegate_capability, load_config, prepare_runtime, reconcile_state
from .affine_evidence import bootstrap_readiness, inspect_bootstrap, inspect_evidence


def main(argv=None):
    import sys

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "competition":
        from .affine_competition_cli import main as competition_main

        return competition_main(arguments[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "delegate", "prepare", "run", "evidence", "reconcile", "competition"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--epoch")
    parser.add_argument("--miner")
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--out", help="New owner-private capability or prepared config outside Git")
    args = parser.parse_args(arguments)
    try:
        config = load_config(args.config)
        if args.command == "check":
            result = bootstrap_readiness(config, inspect_bootstrap(config, allow_closed=True))
        elif args.command == "delegate":
            if not args.out:
                parser.error("delegate requires --out")
            result = delegate_capability(config, args.out)
        elif args.command == "prepare":
            if not args.out:
                parser.error("prepare requires --out")
            result = prepare_runtime(config, args.out)
        elif args.command == "run":
            result = AffineRunner(config).run(max_seconds=args.max_seconds)
        elif args.command == "reconcile":
            result = reconcile_state(config)
        else:
            if not args.epoch or not args.miner:
                parser.error("evidence requires --epoch and --miner")
            result = inspect_evidence(config, args.epoch, args.miner)
            # Native identities, private paths, envelopes and URLs never go to stdout.
            result = {k: v for k, v in result.items() if k not in
                      ("epoch_id", "bundle_path", "native_artifact_digests")}
        print(json.dumps(result, sort_keys=True))
        return 1 if result.get("stage") in ("failed", "cancelled", "budget_exhausted", "log_limit_exceeded") else 0
    except Exception:
        print(json.dumps({"error": "Affine operation failed; inspect private operator state."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
