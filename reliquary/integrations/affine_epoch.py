"""Execute one authenticated epoch through admitted native miner code.

The first stage reuses the pinned native source admission and asset hydration.
A fresh isolated second stage supplies the original signed manifest to the native
CLI, so a rotating discovery pointer cannot expand the approved allocation.
"""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import re
import runpy
import stat
import sys
import time
from types import ModuleType


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(condition):
    if not condition:
        raise ValueError("authenticated epoch scope is invalid")


def read_snapshot(path, expected):
    require(path.is_absolute() and not any(p.is_symlink() for p in (path, *path.parents)))
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and not info.st_mode & 0o077
                and info.st_size <= 64 * 1024 * 1024)
        raw = stream.read(64 * 1024 * 1024 + 1)
    require(len(raw) <= 64 * 1024 * 1024 and hashlib.sha256(raw).hexdigest() == expected)

    def unique(pairs):
        value = {}
        for key, member in pairs:
            require(key not in value)
            value[key] = member
        return value

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=lambda _: require(False))
    require(isinstance(value, dict))
    return value


def signed(envelope, authority):
    from nacl.signing import VerifyKey

    require(isinstance(envelope, dict) and set(envelope) == {"payload", "signer", "signature"}
            and envelope["signer"] == authority and isinstance(envelope["payload"], dict))
    signature = base64.b64decode(envelope["signature"], validate=True)
    VerifyKey(bytes.fromhex(authority)).verify(canonical(envelope["payload"]), signature)
    return envelope["payload"]


def authenticate(snapshot, authority, verifier):
    current = verifier(snapshot["current_envelope"], authority)
    manifest = verifier(snapshot["manifest_envelope"], authority)
    require(current.get("transport_policy") == manifest.get("transport_policy") == "direct-r2-v1"
            and isinstance(manifest.get("epoch"), str) and current.get("epoch") == manifest["epoch"]
            and type(manifest.get("deadline")) in (int, float) and math.isfinite(manifest["deadline"])
            and manifest["deadline"] > time.time())
    return current, manifest


def execute(args):
    require(re.fullmatch("[0-9a-f]{64}", args.authority) and re.fullmatch("[0-9a-f]{64}", args.snapshot_sha256))
    snapshot = read_snapshot(args.snapshot_file, args.snapshot_sha256)
    if args.source_root is None:
        # Load the exact verified .py bytes; -B alone permits stale ignored .pyc reads.
        bootstrap = args.upstream_checkout / "subnet/source_bootstrap.py"
        native = ModuleType("approved_source_bootstrap")
        native.__file__ = str(bootstrap)
        exec(compile(bootstrap.read_bytes(), str(bootstrap), "exec", dont_inherit=True), native.__dict__)

        current, manifest = authenticate(snapshot, args.authority,
                                        lambda envelope, key: native.signed(native.canonical(envelope), key))
        native.r2_url(current["manifest_url"])
        descriptor = manifest["source_bundle"]
        body = native.download(native.r2_url(descriptor["url"]), native.COMPRESSED_LIMIT)
        source = native.install(body, descriptor, args.source_cache)
        native.hydrate_task_assets(source, manifest, args.source_cache)
        environment = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
        command = [sys.executable, "-I", "-B", str(Path(__file__).resolve()), *sys.argv[1:],
                   "--source-root", str(source)]
        os.chdir(source)
        os.execve(sys.executable, command, environment)

    # No bootstrap package survives the exec boundary or precedes admitted source.
    current, manifest = authenticate(snapshot, args.authority, signed)
    descriptor = manifest["source_bundle"]
    require(args.source_root == args.source_cache / descriptor["sha256"]
            and args.source_root.is_dir()
            and not any(p.is_symlink() for p in (args.source_root, *args.source_root.parents)))
    sys.path.insert(0, str(args.source_root))
    from subnet import client

    def initial_manifest(url, authority):
        require(url == current["manifest_url"] and authority == args.authority)
        return copy.deepcopy(manifest)

    client.fetch_signed = initial_manifest
    command = ["affine-miner", "--gateway", "https://unused.invalid", "--authority", args.authority,
               "--manifest-url", current["manifest_url"], "--source-bundle-sha256", descriptor["sha256"],
               "--state", str(args.state), "--key" if args.key else "--cap-file", str(args.key or args.cap_file),
               "--once", "--search-budget", str(args.search_budget), "--max-batches", str(args.max_batches)]
    if args.env_id is not None:
        command += ["--env-id", args.env_id]
    if args.indices is not None:
        command += ["--indices", *map(str, args.indices)]
    sys.argv = command
    runpy.run_module("subnet.cli", run_name="__main__")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--upstream-checkout", type=Path, required=True)
    parser.add_argument("--snapshot-file", type=Path, required=True)
    parser.add_argument("--snapshot-sha256", required=True)
    parser.add_argument("--authority", required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    credential = parser.add_mutually_exclusive_group(required=True)
    credential.add_argument("--key", type=Path)
    credential.add_argument("--cap-file", type=Path)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--search-budget", type=int, required=True)
    parser.add_argument("--max-batches", type=int, required=True)
    parser.add_argument("--env-id")
    parser.add_argument("--indices", type=int, nargs="+")
    parser.add_argument("--source-root", type=Path)
    try:
        execute(parser.parse_args())
    except Exception:
        print("Authenticated epoch execution failed; inspect private state.", file=sys.stderr)
        raise SystemExit(1)
