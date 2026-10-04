"""Prepare, enroll, declare and settle one bounded native competition."""
import argparse
import asyncio
from dataclasses import asdict
import json
from pathlib import Path
import time

from . import affine_competition as competition
from .affine import load_config
from .affine_evidence import inspect_bootstrap, inspect_evidence


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "enroll", "declare", "capture", "settle", "status"))
    parser.add_argument("--config", help="Existing private native runtime configuration")
    parser.add_argument("--competition", help="Owner-private immutable competition draft")
    parser.add_argument("--task-id")
    parser.add_argument("--cap", type=float)
    parser.add_argument("--native-key", help="Existing local native identity seed, never transferred")
    parser.add_argument("--wallet-name", default="default")
    parser.add_argument("--wallet-hotkey", default="default")
    parser.add_argument("--wallet-path")
    parser.add_argument("--enrollment", action="append", default=[])
    parser.add_argument("--evidence", action="append", default=[])
    parser.add_argument("--submission-sha256")
    parser.add_argument("--out", help="New owner-private output outside Git")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--fleet-knows-native-affine-points", action="store_true")
    args = parser.parse_args(argv)

    def needed(*names):
        for name in names:
            if getattr(args, name) is None:
                parser.error("requires --" + name.replace("_", "-"))

    try:
        if args.command == "prepare":
            needed("config", "task_id", "cap", "out")
            config = load_config(args.config)
            snapshot = inspect_bootstrap(config, allow_closed=True)
            draft = competition.prepare(snapshot, authority=config.authority,
                                        task_id=args.task_id, cap=args.cap, env_id=config.env_id)
            competition.write_private(args.out, draft)
            result = dict(stage="prepared", declared=False, gpu_qualified=False,
                          contract_digest=competition.digest(draft["contract"]))
        else:
            needed("competition")
            draft = competition.read_private(args.competition)
            manifest = competition.validate_draft(draft)
            enrollments = [competition.read_private(path) for path in args.enrollment]
            if args.command == "enroll":
                needed("native_key", "out")
                from bittensor_wallet import Wallet

                key = Path(args.native_key)
                competition._private_file(key)
                competition.require(key.stat().st_size <= 128, "native_seed_budget")
                options = dict(name=args.wallet_name, hotkey=args.wallet_hotkey)
                if args.wallet_path:
                    options["path"] = args.wallet_path
                enrollment = competition.sign_enrollment(draft, bytes.fromhex(key.read_text().strip()), Wallet(**options))
                competition.write_private(args.out, enrollment)
                result = dict(stage="enrollment_signed", declared=False)
            elif args.command == "capture":
                needed("config", "out")
                competition.require(len(enrollments) == 1, "one_enrollment_for_capture")
                rows = competition.roster(draft, enrollments)
                native = rows[0]["payload"]["native_id"]
                config = load_config(args.config)
                competition.require(config.authority == draft["contract"]["authority"], "capture_authority")
                evidence = inspect_evidence(config, manifest["epoch"], native,
                                            submission_sha256=args.submission_sha256,
                                            expected_bindings=draft["contract"]["bindings"])
                bundle = competition.read_private(evidence["bundle_path"])
                competition.write_private(args.out, bundle)
                result = dict(stage=evidence["stage"], bundle_digest=competition.digest(bundle),
                              native_payment_verified=False)
            elif args.command == "declare":
                entry = competition.task_entry(draft, enrollments)
                if args.dry_run:
                    needed("out")
                    competition.write_private(args.out, asdict(entry))
                    result = dict(stage="declaration_prepared", declared=False)
                else:
                    asyncio.run(competition.declare(draft, enrollments,
                                acknowledged=args.fleet_knows_native_affine_points))
                    result = dict(stage="declared", declared=True)
            else:
                async def run():
                    if args.command == "settle" and args.dry_run:
                        needed("out")
                        entry = competition.task_entry(draft, enrollments)
                        bundles = [competition.read_private(path) for path in args.evidence]
                        archive = competition.reward_archive(entry, draft, enrollments, bundles, now=time.time())
                        competition.write_private(args.out, archive)
                        return dict(stage="settlement_prepared", archive_written=False,
                                    rewarded_miners=len(archive["rewards_by_hotkey"]),
                                    points=sum(archive["raw_unique_observed_points"].values()),
                                    observation_scope=archive["scope"], native_payment_verified=False)
                    from reliquary.infrastructure.task_registry_store import read_registry

                    entries, _ = await read_registry()
                    entry = entries.get(draft["task_id"])
                    competition.require(entry is not None, "task_not_declared")
                    competition.require(entry.contract == competition.task_entry(draft, enrollments).contract,
                                        "declared_roster_binding")
                    if args.command == "status":
                        archive = await competition.read_archive(entry)
                        return dict(stage="settled" if archive else "awaiting_finalized_evidence",
                                    declared=True, settled=archive is not None,
                                    rewarded_miners=len(archive["rewards_by_hotkey"]) if archive else 0,
                                    native_payment_verified=False)
                    bundles = [competition.read_private(path) for path in args.evidence]
                    archive = await competition.settle(entry, draft, enrollments, bundles)
                    return dict(stage="settled", archive_written=True,
                                rewarded_miners=len(archive["rewards_by_hotkey"]),
                                points=sum(archive["raw_unique_observed_points"].values()),
                                observation_scope=archive["scope"], native_payment_verified=False)
                result = asyncio.run(run())
        print(json.dumps(result, sort_keys=True))
        return 0
    except Exception:
        print(json.dumps({"error": "Native competition operation failed; inspect private inputs and state."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
