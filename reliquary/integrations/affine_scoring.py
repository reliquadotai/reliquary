"""Frozen V9 GPU computation, with private inputs and no score signing or chain writes."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path, PurePosixPath
import platform
import re
import secrets
import select
import selectors
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import zipfile

from reliquary.integrations.affine import (
    _directory,
    _lock,
    _private_file,
    _private_open,
    _private_path,
)
from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.shared.strict_json import strict_json_loads

SCHEMA = "affine-v9-job/v1"
SOURCE_REVISION = "8fa5c14caf47cda012928000aedb991e4ee33fd4"
NATIVE_SOURCE = {
    "grad_docs_v9.py": "d9030c4d8408fab8a79312cd2912a66f6fe077311ae430e9d05fb85e93325e1f",
    "v9common.py": "014889c11c97022ee498879fc0abee50c4981379cb39dc599917e4c21facd1c6",
}
NATIVE_ROOT = Path("/root/sn120-scorer-v9")
TASK_PATH = Path("/root/trusted-sampler/data/original-math7496.tasks.json")
HEX = re.compile(r"[0-9a-f]{64}")
DIGEST_FIELDS = (
    "jobDigest",
    "sourceClosureDigest",
    "runtimeDigest",
    "inputInventoryDigest",
    "checkpointInventoryDigest",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def checksum(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path, limit=32 * 1024**2):
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= limit,
        "bounded regular JSON required",
    )
    return strict_json_loads(path.read_bytes())


def write_json(path, value, *, replace=False):
    path = Path(path)
    _private_path(path)
    _directory(path.parent)
    if not replace:
        fd = _private_open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        return
    fd, name = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(canonical_json_bytes(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def inventory(directory, *, max_bytes=256 * 1024**2):
    directory = Path(directory)
    require(
        directory.is_dir() and not directory.is_symlink(),
        "regular input directory required",
    )
    result, size = {}, 0
    for path in sorted(directory.rglob("*")):
        require(not path.is_symlink(), "symlinks are not supported")
        if path.is_dir():
            continue
        require(path.is_file(), "regular input files required")
        size += path.stat().st_size
        require(
            len(result) < 4096 and size <= max_bytes,
            "input inventory exceeds its byte/file limit",
        )
        result[path.relative_to(directory).as_posix()] = {
            "sha256": checksum(path),
            "bytes": path.stat().st_size,
        }
    require(result, "empty input inventory")
    return result


def validate_inventory(files):
    require(isinstance(files, dict) and 0 < len(files) <= 4096, "invalid inventory")
    for name, entry in files.items():
        require(isinstance(name, str), "invalid inventory name")
        p = PurePosixPath(name)
        require(
            isinstance(name, str)
            and p.parts
            and not p.is_absolute()
            and p.as_posix() == name
            and ".." not in p.parts
            and "\\" not in name
            and len(name) <= 256,
            "invalid inventory name",
        )
        require(
            isinstance(entry, dict)
            and set(entry) == {"sha256", "bytes"}
            and isinstance(entry["sha256"], str)
            and HEX.fullmatch(entry["sha256"])
            and type(entry["bytes"]) is int
            and 0 <= entry["bytes"] <= 256 * 1024**3,
            "invalid inventory entry",
        )


def load_job(bundle):
    bundle = Path(bundle)
    _private_path(bundle)
    _private_file(bundle / "job.json")
    job = read_json(bundle / "job.json")
    expected = {
        "schema",
        "operation",
        "assurance",
        "epoch",
        "shard",
        "nshards",
        "layers",
        "baseCheckpoint",
        "sourceFiles",
        "inputFiles",
        "checkpointFiles",
        "runtime",
        "limits",
        *DIGEST_FIELDS,
    }
    require(
        isinstance(job, dict)
        and set(job) == expected
        and job["schema"] == SCHEMA
        and job["operation"] == "gradient-dot",
        "unsupported scoring job",
    )
    require(
        job["assurance"] in {"fixture", "historical-v9-compute"},
        "unsupported assurance",
    )
    require(
        type(job["epoch"]) is int
        and 0 <= job["epoch"] <= 10**9
        and job["shard"] == 0
        and type(job["shard"]) is int
        and job["nshards"] == 1
        and type(job["nshards"]) is int
        and job["layers"] == 4
        and type(job["layers"]) is int,
        "only one independent four-layer shard is supported",
    )
    require(
        isinstance(job["baseCheckpoint"], str) and HEX.fullmatch(job["baseCheckpoint"]),
        "full checkpoint identity required",
    )
    for key in ("sourceFiles", "inputFiles", "checkpointFiles"):
        validate_inventory(job[key])
    runtime = job["runtime"]
    require(
        isinstance(runtime, dict)
        and set(runtime) == {"python", "packages", "gpu", "upstreamRevision"}
        and runtime["upstreamRevision"] == SOURCE_REVISION
        and runtime["python"] == "3.12"
        and isinstance(runtime["packages"], dict),
        "exact reviewed runtime required",
    )
    require(
        all(
            isinstance(k, str)
            and re.fullmatch(r"[a-zA-Z0-9_.-]{1,64}", k)
            and isinstance(v, str)
            and 0 < len(v) <= 128
            for k, v in runtime["packages"].items()
        ),
        "invalid runtime packages",
    )
    require(
        runtime["gpu"]
        == (
            {"name": "none", "sm": [0, 0]}
            if job["assurance"] == "fixture"
            else {"name": "NVIDIA H200", "sm": [9, 0]}
        ),
        "unsupported GPU profile",
    )
    if job["assurance"] != "fixture":
        require(
            set(job["sourceFiles"]) == set(NATIVE_SOURCE)
            and all(
                job["sourceFiles"][k]["sha256"] == v for k, v in NATIVE_SOURCE.items()
            ),
            "reviewed compute-only source differs",
        )
        require(
            {"torch", "numpy", "transformers", "safetensors", "tokenizers"}
            <= set(runtime["packages"]),
            "incomplete runtime inventory",
        )
        require(
            "config.json" in job["checkpointFiles"]
            and any(k.endswith(".safetensors") for k in job["checkpointFiles"])
            and all(
                not k.endswith((".py", ".pyc", ".pkl", ".bin"))
                for k in job["checkpointFiles"]
            ),
            "offline safetensors checkpoint required",
        )
    limits = job["limits"]
    bounds = {
        "maxSeconds": (1, 86400),
        "maxLogBytes": (1024, 64 * 1024**2),
        "maxOutputBytes": (1024, 256 * 1024**2),
        "maxWorkspaceBytes": (1024, 1024**4),
        "finalizationReserveSeconds": (1, 3600),
    }
    require(
        isinstance(limits, dict) and set(limits) == set(bounds),
        "invalid resource limits",
    )
    for key, (lo, hi) in bounds.items():
        require(
            type(limits[key]) is int and lo <= limits[key] <= hi,
            "resource limit outside supported bound",
        )
    require(
        limits["finalizationReserveSeconds"] < limits["maxSeconds"],
        "execution needs a finalization reserve",
    )
    for name, value in (
        ("sourceClosureDigest", job["sourceFiles"]),
        ("runtimeDigest", runtime),
        ("inputInventoryDigest", job["inputFiles"]),
        ("checkpointInventoryDigest", job["checkpointFiles"]),
    ):
        require(job[name] == canonical_sha256(value), "job inventory digest differs")
    require(
        job["jobDigest"]
        == canonical_sha256({k: v for k, v in job.items() if k != "jobDigest"}),
        "job digest differs",
    )
    require(
        inventory(bundle / "source") == job["sourceFiles"]
        and inventory(bundle / "inputs") == job["inputFiles"],
        "frozen input bytes differ",
    )
    from .affine_scoring_contract import validate_inputs

    return job, validate_inputs(bundle / "inputs", job)


def prepare(config_path, destination):
    config_path, destination = Path(config_path), Path(destination)
    _private_file(config_path)
    config = read_json(config_path)
    require(
        set(config)
        == {
            "source",
            "inputs",
            "checkpointInventory",
            "runtime",
            "limits",
            "epoch",
            "baseCheckpoint",
        },
        "invalid preparation config",
    )
    _private_path(destination)
    require(
        destination.is_absolute() and not destination.exists(),
        "new private bundle directory required",
    )
    _directory(destination)
    for key in ("source", "inputs"):
        source = Path(config[key])
        _private_path(source)
        inventory(source)
        shutil.copytree(source, destination / key)
        for path in (destination / key).rglob("*"):
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
    checkpoint = read_json(config["checkpointInventory"])
    validate_inventory(checkpoint)
    job = dict(
        schema=SCHEMA,
        operation="gradient-dot",
        assurance="historical-v9-compute",
        epoch=config["epoch"],
        shard=0,
        nshards=1,
        layers=4,
        baseCheckpoint=config["baseCheckpoint"],
        sourceFiles=inventory(destination / "source"),
        inputFiles=inventory(destination / "inputs"),
        checkpointFiles=checkpoint,
        runtime=config["runtime"],
        limits=config["limits"],
    )
    for name, value in (
        ("sourceClosureDigest", job["sourceFiles"]),
        ("runtimeDigest", job["runtime"]),
        ("inputInventoryDigest", job["inputFiles"]),
        ("checkpointInventoryDigest", checkpoint),
    ):
        job[name] = canonical_sha256(value)
    job["jobDigest"] = canonical_sha256(job)
    write_json(destination / "job.json", job)
    (destination / "source/data" / f"e{job['epoch']}").mkdir(
        mode=0o700, parents=True, exist_ok=True
    )
    load_job(destination)
    return summary(job, "prepared")


def summary(job, stage):
    return {
        "schema": "affine-v9-result/v1",
        "stage": stage,
        "operation": "gradient-dot",
        "assurance": job["assurance"],
        **{k: job[k] for k in DIGEST_FIELDS},
        "gpuQualified": False,
        "operatorAccepted": False,
        "nativeRewardVerified": False,
        "chainTransactions": False,
    }


def native_mounts(bundle, workspace, checkpoint):
    """Read-only bindings for the unchanged upstream paths, plus a private output tree."""
    bundle, workspace, checkpoint = Path(bundle), Path(workspace), Path(checkpoint)
    job, _ = load_job(bundle)
    native_data = NATIVE_ROOT / "data" / f"e{job['epoch']}"
    mounts = [
        (bundle / "source", NATIVE_ROOT, True),
        (workspace / "native", native_data, False),
        (checkpoint, Path("/root/ckpt-cache") / job["baseCheckpoint"], True),
    ]
    for name in job["inputFiles"]:
        target = TASK_PATH if name == "tasks.json" else native_data / name
        mounts.append((bundle / "inputs" / name, target, True))
    return [(str(a), str(b), ro) for a, b, ro in mounts]


def _runtime(job, checkpoint, workspace):
    require(
        platform.system() == "Linux" and Path("/.dockerenv").is_file(),
        "native execution requires the qualified container host",
    )
    require(
        checkpoint == Path("/root/ckpt-cache") / job["baseCheckpoint"],
        "native checkpoint mount differs",
    )
    _private_path(checkpoint)
    require(
        inventory(checkpoint, max_bytes=256 * 1024**3) == job["checkpointFiles"],
        "checkpoint bytes differ",
    )
    require(sys.version_info[:2] == (3, 12), "runtime interpreter differs")
    for name, version in job["runtime"]["packages"].items():
        require(
            importlib.metadata.version(name) == version, "installed runtime differs"
        )
    observed = (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
            text=True,
            timeout=10,
        )
        .strip()
        .splitlines()
    )
    require(
        len(observed) == 1 and "H200" in observed[0] and observed[0].endswith("9.0"),
        "GPU differs from the prepared runtime",
    )
    for name, expected in NATIVE_SOURCE.items():
        require(checksum(NATIVE_ROOT / name) == expected, "mounted source differs")
    native_data = NATIVE_ROOT / "data" / f"e{job['epoch']}"
    require(
        os.path.samefile(workspace / "native", native_data),
        "native output mount differs",
    )
    require(
        inventory(native_data)
        == {k: v for k, v in job["inputFiles"].items() if k != "tasks.json"},
        "native input tree differs",
    )
    for name, item in job["inputFiles"].items():
        path = TASK_PATH if name == "tasks.json" else native_data / name
        require(checksum(path) == item["sha256"], "mounted input differs")
        require(
            os.statvfs(path).f_flag & os.ST_RDONLY, "native inputs must be read-only"
        )
    for path in [NATIVE_ROOT, checkpoint]:
        require(
            os.statvfs(path).f_flag & os.ST_RDONLY,
            "native source, task and checkpoint mounts must be read-only",
        )
    cache = Path("/tmp") / ("v9-cache-" + secrets.token_hex(16))
    require(
        not cache.exists() and not cache.is_symlink(), "fresh bytecode prefix required"
    )
    bootstrap = "import runpy,sys;sys.path.insert(0,'/root/sn120-scorer-v9');runpy.run_path('/root/sn120-scorer-v9/grad_docs_v9.py',run_name='__main__')"
    return [
        sys.executable,
        "-I",
        "-X",
        "pycache_prefix=" + str(cache),
        "-B",
        "-c",
        bootstrap,
        "--epoch",
        str(job["epoch"]),
        "--shard",
        "0",
        "--nshards",
        "1",
        "--layers",
        "4",
    ]


def run(
    bundle,
    workspace,
    *,
    checkpoint=None,
    max_seconds=None,
    fixture=False,
    stop_event=None,
):
    from .affine_scoring_contract import validate_output

    start = time.monotonic()
    bundle, workspace = Path(bundle), Path(workspace)
    job, context = load_job(bundle)
    _private_path(workspace)
    _directory(workspace)
    limit = job["limits"]
    seconds = limit["maxSeconds"] if max_seconds is None else max_seconds
    require(
        type(seconds) in (int, float)
        and math.isfinite(seconds)
        and 0 < seconds <= limit["maxSeconds"],
        "invalid allocation deadline",
    )
    deadline = start + seconds - limit["finalizationReserveSeconds"]
    with _lock(workspace):
        state = workspace / "state.json"
        if state.exists():
            previous = read_json(state)
            require(
                previous.get("jobDigest") == job["jobDigest"],
                "workspace belongs to another job",
            )
            if previous.get("stage") == "compute_completed":
                return verify(bundle, workspace)
            raise ValueError(
                "previous attempt needs reconciliation and a new workspace"
            )
        require(
            not (workspace / "native/dots-0.json").exists()
            and not (workspace / "native/refgrad").exists(),
            "fresh native outputs required",
        )
        (workspace / "native").mkdir(mode=0o700, exist_ok=True)
        require(
            fixture == (job["assurance"] == "fixture"),
            "fixture and native execution cannot be interchanged",
        )
        if fixture:
            command = [
                sys.executable,
                "-B",
                str(Path(__file__).with_name("affine_scoring_fixture.py")),
                str(bundle),
                str(workspace),
            ]
        else:
            require(checkpoint is not None, "native checkpoint required")
            command = _runtime(job, Path(checkpoint), workspace)
        require(time.monotonic() < deadline, "preflight exhausted allocation")
        require(
            stop_event is None or not stop_event.is_set(), "cancelled during preflight"
        )
        env = {
            k: os.environ[k]
            for k in (
                "PATH",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "TMPDIR",
                "LANG",
                "LC_ALL",
            )
            if k in os.environ
        }
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={
                **env,
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "PYTHONNOUSERSITE": "1",
                "PYTHONDONTWRITEBYTECODE": "1",
            },
        )
        record = {**summary(job, "running"), "pid": process.pid}
        reason, total_log, exited = None, 0, False
        exit_watch = None
        try:
            write_json(state, record)
            if not hasattr(os, "waitid"):
                exit_watch = select.kqueue()
                exit_watch.control(
                    [
                        select.kevent(
                            process.pid,
                            filter=select.KQ_FILTER_PROC,
                            flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                            fflags=select.KQ_NOTE_EXIT,
                        )
                    ],
                    0,
                    0,
                )
            os.set_blocking(process.stdout.fileno(), False)
            with (
                (workspace / "runtime.log").open("xb") as log,
                selectors.DefaultSelector() as selector,
            ):
                os.chmod(workspace / "runtime.log", 0o600)
                selector.register(process.stdout, selectors.EVENT_READ)
                while True:
                    if stop_event is not None and stop_event.is_set():
                        reason = "cancelled"
                        break
                    if time.monotonic() >= deadline:
                        reason = "budget_exhausted"
                        break
                    paths = list(workspace.rglob("*"))
                    require(
                        len(paths) <= 8192 and not any(p.is_symlink() for p in paths),
                        "invalid output tree",
                    )
                    if (
                        sum(p.stat().st_size for p in paths if p.is_file())
                        > limit["maxWorkspaceBytes"]
                    ):
                        reason = "storage_limit_exceeded"
                        break
                    for key, _ in selector.select(0.05):
                        block = os.read(key.fd, 65536)
                        if block:
                            remaining = limit["maxLogBytes"] - total_log
                            log.write(block[:remaining])
                            total_log += min(len(block), remaining)
                            if len(block) > remaining:
                                reason = "log_limit_exceeded"
                                break
                    if reason:
                        break
                    # WNOWAIT reserves the leader PID until its entire group is killed.
                    if (
                        exit_watch.control(None, 1, 0)
                        if exit_watch
                        else os.waitid(
                            os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                        )
                    ):
                        exited = True
                        break
        except KeyboardInterrupt:
            reason = "cancelled"
        except BaseException:
            reason = "failed"
            raise
        finally:
            for sig in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, sig)
                except ProcessLookupError:
                    pass
                except PermissionError:
                    # Darwin reports EPERM for a group containing only its zombie leader.
                    if exit_watch and not exited:
                        exited = bool(exit_watch.control(None, 1, 0.1))
                    if not (exit_watch and exited):
                        raise
            process.wait(timeout=10)
            process.stdout.close()
            if exit_watch:
                exit_watch.close()
            record.update(
                stage=reason
                or ("process_exited" if process.returncode == 0 else "failed")
            )
            write_json(state, record, replace=True)
        require(record["stage"] == "process_exited", record["stage"])
        output = workspace / "native" / "dots-0.json"
        require(
            output.is_file() and output.stat().st_size <= limit["maxOutputBytes"],
            "bounded native output required",
        )
        scientific = validate_output(output, job, context)
        result = {
            **summary(job, "compute_completed"),
            **scientific,
            "elapsedSeconds": time.monotonic() - start,
            "nativeResultDigest": checksum(output),
            "outputInventoryDigest": canonical_sha256(
                {
                    "dots-0.json": {
                        "sha256": checksum(output),
                        "bytes": output.stat().st_size,
                    }
                }
            ),
        }
        write_json(workspace / "compute-result.json", result)
        record["stage"] = "compute_completed"
        write_json(state, record, replace=True)
        return result


def verify(bundle, workspace):
    from .affine_scoring_contract import validate_output

    job, context = load_job(bundle)
    workspace = Path(workspace)
    result = read_json(workspace / "compute-result.json")
    require(
        all(result.get(k) == v for k, v in summary(job, "compute_completed").items()),
        "result binding differs",
    )
    output = workspace / "native/dots-0.json"
    require(
        checksum(output) == result.get("nativeResultDigest"),
        "native result bytes differ",
    )
    require(
        output.stat().st_size <= job["limits"]["maxOutputBytes"],
        "output byte limit exceeded",
    )
    require(
        result.get("outputInventoryDigest")
        == canonical_sha256(
            {
                "dots-0.json": {
                    "sha256": checksum(output),
                    "bytes": output.stat().st_size,
                }
            }
        ),
        "output inventory differs",
    )
    scientific = validate_output(output, job, context)
    require(
        all(result.get(k) == v for k, v in scientific.items()), "result metrics differ"
    )
    require(
        type(result.get("elapsedSeconds")) in (int, float)
        and math.isfinite(result["elapsedSeconds"])
        and 0 <= result["elapsedSeconds"] <= job["limits"]["maxSeconds"],
        "result timing differs",
    )
    return result


def export(bundle, workspace, destination):
    verify(bundle, workspace)
    destination, workspace = Path(destination), Path(workspace)
    _private_path(destination)
    _directory(destination.parent)
    fd = _private_open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with (
        os.fdopen(fd, "wb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        archive.write(workspace / "compute-result.json", "compute-result.json")
        archive.write(workspace / "native/dots-0.json", "dots-0.json")
    require(
        destination.stat().st_size <= load_job(bundle)[0]["limits"]["maxOutputBytes"],
        "artifact byte limit exceeded",
    )
    return {
        **summary(load_job(bundle)[0], "exported"),
        "artifactSha256": checksum(destination),
        "artifactBytes": destination.stat().st_size,
    }


def compare(bundle, workspace, reference, *, atol, rtol):
    """Compare validated output to an operator-provided oracle without admitting it."""
    from .affine_scoring_contract import validate_output

    require(
        all(
            type(v) in (int, float) and math.isfinite(v) and v >= 0
            for v in (atol, rtol)
        ),
        "explicit finite tolerances required",
    )
    job, context = load_job(bundle)
    result = verify(bundle, workspace)
    reference = Path(reference)
    _private_file(reference)
    validate_output(reference, job, context)
    expected = read_json(reference, 128 * 1024**2)
    actual = read_json(Path(workspace) / "native/dots-0.json", 128 * 1024**2)
    maximum, mismatches, values = 0, 0, 0
    for observed, oracle in zip(actual["docs"], expected["docs"]):
        for left, right in zip(observed["dots"], oracle["dots"]):
            for a, b in zip(left, right):
                error = abs(a - b)
                require(math.isfinite(error), "comparison overflow")
                maximum = max(maximum, error)
                mismatches += not math.isclose(a, b, abs_tol=atol, rel_tol=rtol)
                values += 1
    return {
        **summary(job, "compared"),
        "nativeResultDigest": result["nativeResultDigest"],
        "referenceSha256": checksum(reference),
        "atol": atol,
        "rtol": rtol,
        "values": values,
        "mismatches": mismatches,
        "maxAbsError": maximum,
        "comparisonPassed": mismatches == 0,
        "independentReferenceVerified": False,
    }


def reconcile(workspace):
    workspace = Path(workspace)
    with _lock(workspace):
        state = read_json(workspace / "state.json")
        require(
            state.get("stage") != "compute_completed",
            "completed output should be verified and exported",
        )
        pid = state.get("pid")
        require(type(pid) is int and pid > 1, "recorded process group required")
        try:
            os.killpg(pid, 0)
        except ProcessLookupError:
            state["stage"] = "reconciled"
            write_json(workspace / "state.json", state, replace=True)
            return {"stage": "reconciled", "rerunAllowed": False}
        raise ValueError("original process group still exists; reconciliation refused")
