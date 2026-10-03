"""Small operator entry point; no SN81 inference or validator imports."""
import argparse
import json

from .affine import AffineRunner, load_config, reconcile_state, verify_checkout
from .affine_evidence import inspect_bootstrap, inspect_evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "run", "evidence", "reconcile"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--epoch")
    parser.add_argument("--miner")
    parser.add_argument("--max-seconds", type=float)
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        if args.command == "check":
            verify_checkout(config)
            snapshot = inspect_bootstrap(config)
            result = {"schema": "affine-readiness/v1", "bootstrap_verified": True,
                      "bindings": snapshot["bindings"], "hardware_qualified": False, "paid": False}
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
