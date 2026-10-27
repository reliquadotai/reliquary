"""Bounded entry point for original signed native Affine evaluation jobs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import signal
from threading import Event
import time

from . import affine_native_evaluation as evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "check", "preflight", "run", "verify", "export"))
    for name in ("config", "bundle", "workspace", "checkpoint", "out"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--max-seconds", type=float)
    parser.add_argument("--hf-repository")
    parser.add_argument("--hf-revision")
    parser.add_argument("--max-checkpoint-bytes", type=int)
    parser.add_argument("--fixture", action="store_true")
    args = parser.parse_args(argv)
    hf_options = (args.hf_repository, args.hf_revision, args.max_checkpoint_bytes)
    if any(value is not None for value in hf_options) and args.command != "prepare":
        parser.error("HF options require prepare")

    def need(*names):
        if any(getattr(args, name) is None for name in names):
            parser.error("required: " + ", ".join("--" + name.replace("_", "-") for name in names))

    try:
        if args.command == "prepare":
            need("config", "out")
            if any(value is not None for value in hf_options):
                need("hf_repository", "hf_revision", "max_checkpoint_bytes")
                from .affine_hf_checkpoint import materialize

                materialize(args.config, repository=args.hf_repository, revision=args.hf_revision,
                            max_bytes=args.max_checkpoint_bytes)
            result = evaluation.prepare(args.config, args.out)
        else:
            need("bundle")
            if args.command == "preflight":
                need("config")
                from .affine_native_preflight import preflight
                result = preflight(args.bundle, args.config)
            elif args.command == "check":
                job, *_ = evaluation.load_bundle(args.bundle)
                result = {"stage": "metadata_checked", "jobDigest": job["jobDigest"],
                          "assurance": job["assurance"], "expired": job["nativeJobExpiresAt"] <= time.time(),
                          "nativeAdmissionVerified": False, "activated": False}
            elif args.command == "run":
                need("workspace")
                stopped = Event()
                previous = signal.signal(signal.SIGTERM, lambda *_: stopped.set())
                try:
                    result = evaluation.run(args.bundle, args.workspace, checkpoint=args.checkpoint,
                                            fixture=args.fixture, max_seconds=args.max_seconds,
                                            stop_event=stopped, lease_check=lambda: not stopped.is_set())
                    result = {key: result[key] for key in ("stage", "jobDigest", "assurance", "nativeParityVerified")}
                finally:
                    signal.signal(signal.SIGTERM, previous)
            elif args.command == "verify":
                need("workspace")
                checked = evaluation.verify(args.bundle, args.workspace)
                result = {key: checked[key] for key in ("stage", "jobDigest", "assurance", "nativeParityVerified")}
            else:
                need("workspace", "out")
                result = evaluation.export(args.bundle, args.workspace, args.out)
        print(json.dumps(result, sort_keys=True))
        return 2 if args.command == "preflight" and not result["offlineReady"] else 0
    except Exception:
        print(json.dumps({"stage": "unavailable", "error": "Native evaluation refused; inspect private inputs and state."}))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
