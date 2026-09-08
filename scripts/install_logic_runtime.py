#!/usr/bin/env python3
"""Install the reviewed optional Logic runtime in the current image's Python.

Run only while building an isolated candidate image. The wheel is verified
before pip sees it; the installed runtime is hash-verified before import.
This command does not select a profile or start any service.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
PIN = json.loads((ROOT / "docker/logic-release.json").read_text())


def check_wheel(path: Path) -> None:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("Logic wheel exceeds size limit")
    if hashlib.sha256(path.read_bytes()).hexdigest() != PIN["wheel_sha256"]:
        raise ValueError("Logic wheel SHA-256 mismatch; no code installed or imported")


def verify_runtime() -> dict:
    sys.path.insert(0, str(ROOT))
    from reliquary.environment.registry import get_environment_spec
    from reliquary.environment.agentic.adapters.prime_v1 import pinned_verifiers_v1

    pinned_verifiers_v1()
    spec = get_environment_spec("reliquary_logic_v2")
    if spec.environment_manifest_sha256 != PIN["artifact_sha256"]:
        raise ValueError("image Logic pin and runtime catalog disagree")
    environment = spec.create()
    problem = environment.get_problem(0)
    good = environment._backend.reference_completion(0)
    if spec.score_many(problem, [good, "incorrect"]) != [1.0, 0.0]:
        raise ValueError("installed Logic reward smoke failed")
    return {"wheel_sha256": PIN["wheel_sha256"],
            "artifact_sha256": PIN["artifact_sha256"],
            "verifiers_commit": PIN["verifiers_commit"],
            "task_id": problem["id"], "rewards": [1.0, 0.0],
            "profile_activated": False}


def install(wheel_source: Path | None) -> dict:
    with tempfile.TemporaryDirectory(prefix="reliquary-logic-install-") as directory:
        wheel = Path(directory) / PIN["wheel"]
        if wheel_source is not None:
            check_wheel(wheel_source)
            shutil.copyfile(wheel_source, wheel)
        else:
            url = f'{PIN["repository"]}/releases/download/{PIN["tag"]}/{PIN["wheel"]}'
            size = 0
            with urllib.request.urlopen(url, timeout=30) as response, wheel.open("xb") as output:
                while chunk := response.read(64 * 1024):
                    size += len(chunk)
                    if size > 16 * 1024 * 1024:
                        raise ValueError("Logic wheel exceeds size limit")
                    output.write(chunk)
        check_wheel(wheel)
        pip = [sys.executable, "-m", "pip", "install"]
        subprocess.run([*pip, "--require-hashes", "--no-deps", "-r",
                        str(ROOT / "docker/logic-runtime-requirements.lock")], check=True)
        subprocess.run([*pip, "--no-deps",
                        "verifiers @ git+https://github.com/PrimeIntellect-ai/verifiers.git@" + PIN["verifiers_commit"]], check=True)
        subprocess.run([*pip, "--no-deps", str(wheel)], check=True)
        subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    return verify_runtime()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, help="reviewed candidate wheel; otherwise download the pinned release")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify_runtime() if args.verify_only else install(args.wheel), indent=2))
