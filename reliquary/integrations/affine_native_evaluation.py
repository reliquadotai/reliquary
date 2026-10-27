"""Frozen, evaluate-only native MiMo jobs; no signing, mining or publication."""

from __future__ import annotations

import hashlib
import importlib.metadata
import math
import os
from pathlib import Path
import platform
import re
import select
import shutil
import signal
import socket
import subprocess
import sys
import time
import zipfile

from .affine import _directory, _lock, _private_file, _private_open, _private_path
from .affine_epoch import canonical, signed
from .affine_checkpoint import _member
from .affine_scoring import checksum, inventory, read_json, require, write_json
from reliquary.protocol.release_contract import canonical_sha256

VERSION = "mimo-heldout-evaluation-v1"
DRAW_POLICY = "mimo-evaluation-common-random-draws-v1"
WORKER_POLICY = "mimo-complete-group-worker-v1"
SCHEMA = "affine-native-evaluation-job/v1"
REVIEWED_REVISION = "ebf4f5f67dd34e3a13c6c275e1b9e3fd3c9766cd"
EVALUATOR_SHA256 = "f658b835a2c40563cef157b69c751c98c9548d7f77a8d3589ab93f6f09d01178"
HEX = re.compile(r"[0-9a-f]{64}")
MAX_REPORT_BYTES = 12_000_000
PINS = (
    "jobDigest",
    "sourceClosureDigest",
    "runtimeDigest",
    "inputInventoryDigest",
    "checkpointInventoryDigest",
)
NATIVE_ROOT = Path("/opt/affine-evaluation")


def _sha(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _integer(value, low, high):
    return type(value) is int and low <= value <= high


def _admit(envelope, authority, *, now=None):
    """Cheap local admission; upstream rechecks its full release policy at run time."""
    require(
        isinstance(authority, str) and HEX.fullmatch(authority),
        "public authority required",
    )
    try:
        job = signed(envelope, authority)
        manifest = signed(job["manifest"], authority)
    except Exception:
        raise ValueError("invalid original native job or opening signature") from None
    require(
        job.get("schema") == 1
        and job.get("role") == "evaluate"
        and job.get("worker_policy") == WORKER_POLICY
        and isinstance(job.get("job_id"), str)
        and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", job["job_id"]),
        "original evaluate-only native job required",
    )
    require(
        not any(
            key in job
            for key in (
                "capability",
                "miner_identity_file",
                "miner_id",
                "mining_subset",
                "submissions",
                "training",
                "put_urls",
                "role_config_file",
            )
        ),
        "evaluation cannot carry mining, training or publication authority",
    )
    require(
        all(
            type(job.get(k)) in (int, float) and math.isfinite(job[k])
            for k in ("created_at", "expires_at")
        )
        and 0 < job["expires_at"] - job["created_at"] <= 86400,
        "original bounded native lifetime required",
    )
    if now is not None:
        require(
            job["created_at"] <= now < job["expires_at"], "original native job expired"
        )
    request = job.get("evaluation")
    require(
        not manifest.get("task_assets"),
        "external task assets require a separately qualified frozen intake",
    )
    require(
        isinstance(request, dict)
        and set(request) == {"version", "run_id", "set_name"}
        and request["version"] == VERSION
        and request["run_id"] == manifest.get("run_id")
        and manifest.get("model_runtime_revision") == "qwen3_5-mimo-v1",
        "original MiMo evaluation request required",
    )
    suites = manifest.get("evaluation_sets")
    require(
        isinstance(suites, list) and 1 <= len(suites) <= 16,
        "signed held-out suites required",
    )
    matches = [
        s
        for s in suites
        if isinstance(s, dict) and s.get("name") == request["set_name"]
    ]
    require(len(matches) == 1, "one original signed suite required")
    suite = matches[0]
    require(
        set(suite) == {"name", "env_id", "indices", "seed", "repeats"}
        and isinstance(suite["indices"], list)
        and 1 <= len(suite["indices"]) <= 750
        and all(_integer(i, 0, 2**31 - 1) for i in suite["indices"])
        and len(set(suite["indices"])) == len(suite["indices"])
        and _integer(suite["seed"], 0, 2**63 - 1)
        and _integer(suite["repeats"], 1, 16),
        "bounded original suite required",
    )
    definitions = manifest.get("environments")
    require(
        isinstance(definitions, list), "signed native environment registry required"
    )
    matches = [
        d
        for d in definitions
        if isinstance(d, dict) and d.get("env_id") == suite["env_id"]
    ]
    require(len(matches) == 1, "original suite environment required")
    definition = matches[0]
    spec, h = definition.get("spec"), definition.get("harness")
    require(
        isinstance(spec, dict)
        and spec.get("max_turns") == 1
        and isinstance(h, dict)
        and set(h)
        == {
            "version",
            "policy",
            "max_output_tokens",
            "max_context_tokens",
            "temperature",
            "top_p",
            "top_k",
            "enable_thinking",
        }
        and h["version"] == "text-tools-mimo-complete-group-v1"
        and h["policy"] == "autoregressive"
        and _integer(h["max_output_tokens"], 1, 8192)
        and _integer(h["max_context_tokens"], h["max_output_tokens"] + 1, 32768)
        and type(h["temperature"]) in (int, float)
        and h["temperature"] == 1
        and type(h["top_p"]) in (int, float)
        and h["top_p"] == 0.95
        and type(h["top_k"]) is int
        and h["top_k"] == 20
        and type(h["enable_thinking"]) is bool
        and len(suite["indices"]) * suite["repeats"] * h["max_output_tokens"]
        <= 1_250_000,
        "exact bounded native MiMo harness required",
    )
    require(
        _integer(spec.get("num_samples"), 1, 10**7)
        and all(i < spec["num_samples"] for i in suite["indices"])
        and not any(
            set(suite["indices"])
            & (
                set(d.get("indices", []))
                | set(
                    manifest.get("sample_harness_registry", {})
                    .get(d["env_id"], {})
                    .get("indices", [])
                )
            )
            for d in definitions
            if d.get("env_id") == suite["env_id"] or d.get("spec") == spec
        ),
        "held-out tasks overlap mining or exceed the taskset",
    )
    runtime, geometry = (
        manifest.get("mimo_runtime"),
        manifest.get("group_transport_policy"),
    )
    require(
        isinstance(runtime, dict)
        and isinstance(geometry, dict)
        and _integer(geometry.get("vocab_size"), 1, 2**20)
        and isinstance(runtime.get("eos_token_ids"), list)
        and runtime["eos_token_ids"]
        and all(
            _integer(t, 0, geometry["vocab_size"] - 1) for t in runtime["eos_token_ids"]
        )
        and isinstance(manifest.get("mimo_answer_projection"), dict),
        "original runtime, geometry and final-channel projection required",
    )
    cp = manifest.get("checkpoint")
    require(
        isinstance(cp, dict)
        and isinstance(cp.get("files"), dict)
        and 1 <= len(cp["files"]) <= 256,
        "bounded checkpoint descriptor required",
    )
    require(
        cp.get("id") == _sha(cp["files"])
        and "config.json" in cp["files"]
        and any(n.endswith(".safetensors") for n in cp["files"])
        and all(
            isinstance(n, str)
            and re.fullmatch(r"[A-Za-z0-9_.-]+", n)
            and not n.startswith(".")
            and Path(n).suffix
            in {".json", ".safetensors", ".txt", ".model", ".jinja", ".tiktoken"}
            and isinstance(v, str)
            and HEX.fullmatch(v)
            for n, v in cp["files"].items()
        ),
        "safe immutable checkpoint files required",
    )
    source = job.get("source_files")
    require(
        isinstance(source, dict)
        and 1 <= len(source) <= 4096
        and "subnet/mimo_evaluation.py" in source
        and all(
            isinstance(n, str)
            and n.startswith("subnet/")
            and ".." not in n
            and n.endswith(".py")
            and isinstance(v, str)
            and HEX.fullmatch(v)
            for n, v in source.items()
        )
        and isinstance(job.get("runtime_versions"), dict)
        and set(job["runtime_versions"]) == {"torch", "transformers", "toploc"}
        and all(
            isinstance(v, str) and 0 < len(v) <= 128
            for v in job["runtime_versions"].values()
        ),
        "complete signed source inventory and package versions required",
    )
    frozen = dict(
        version=VERSION,
        suite=suite,
        environment=spec,
        harness=h,
        runtime=runtime,
        answer_projection=manifest["mimo_answer_projection"],
        draw_policy=DRAW_POLICY,
    )
    return job, manifest, {"suite": suite, "harness": h, "datasetId": _sha(frozen)}


def _limits(value):
    bounds = {
        "maxSeconds": (2, 86400),
        "finalizationReserveSeconds": (1, 3600),
        "maxLogBytes": (1024, 64 * 1024**2),
        "maxOutputBytes": (1024, MAX_REPORT_BYTES),
        "maxWorkspaceBytes": (1024, 1024**4),
    }
    require(
        isinstance(value, dict)
        and set(value) == set(bounds)
        and all(_integer(value[k], *v) for k, v in bounds.items())
        and value["finalizationReserveSeconds"] < value["maxSeconds"],
        "invalid bounded allocation limits",
    )


def _load_intake(config_path):
    config_path = Path(config_path)
    require(config_path.is_absolute(), "absolute private paths required")
    _private_file(config_path)
    c = read_json(config_path, 1024**2)
    require(
        isinstance(c, dict)
        and set(c)
        == {
            "schema",
            "assurance",
            "authority",
            "jobEnvelope",
            "sourceDirectory",
            "sourceRevision",
            "checkpointDirectory",
            "limits",
            "review",
        }
        and c["schema"] == "affine-native-evaluation-intake/v1"
        and c["assurance"] in ("fixture", "native-mimo-evaluation")
        and c["sourceRevision"] == REVIEWED_REVISION
        and c["review"]
        == {
            "sourceApproved": True,
            "checkpointApproved": True,
            "heldoutApproved": True,
        },
        "exact reviewed native evaluation intake required",
    )
    _limits(c["limits"])
    for key in ("jobEnvelope", "sourceDirectory", "checkpointDirectory"):
        require(
            isinstance(c[key], str) and Path(c[key]).is_absolute(),
            "absolute private inputs required",
        )
        _private_path(Path(c[key]))
    _private_file(Path(c["jobEnvelope"]))
    envelope = read_json(c["jobEnvelope"], 4_000_000)
    job, manifest, _ = _admit(envelope, c["authority"], now=time.time())
    return c, envelope, job, manifest


def prepare(config_path, destination):
    """Freeze reviewed local inputs without importing or executing native code."""
    destination = Path(destination)
    require(destination.is_absolute(), "absolute private paths required")
    c, envelope, job, manifest = _load_intake(config_path)
    require(
        all(
            destination != Path(c[key]) and Path(c[key]) not in destination.parents
            for key in ("sourceDirectory", "checkpointDirectory")
        ),
        "prepared bundle must be separate from input trees",
    )
    source = inventory(c["sourceDirectory"], max_bytes=256 * 1024**2)
    require(
        {
            n: v["sha256"]
            for n, v in source.items()
            if n.startswith("subnet/") and n.endswith(".py")
        }
        == job["source_files"],
        "installed full native source closure differs",
    )
    require(
        not any(n.endswith((".pyc", ".pyo")) for n in source),
        "bytecode caches are not admitted source",
    )
    if c["assurance"] != "fixture":
        require(
            job["source_files"]["subnet/mimo_evaluation.py"] == EVALUATOR_SHA256,
            "native evaluator differs from the reviewed contract",
        )
    checkpoint = Path(c["checkpointDirectory"])
    _directory(checkpoint)
    require(
        {p.name for p in checkpoint.iterdir()} == set(manifest["checkpoint"]["files"]),
        "local checkpoint inventory differs",
    )
    files = {
        n: _member(checkpoint / n) for n in sorted(manifest["checkpoint"]["files"])
    }
    require(
        all(
            v["sha256"] == manifest["checkpoint"]["files"][n] for n, v in files.items()
        ),
        "local checkpoint bytes differ",
    )
    _private_path(destination)
    _directory(destination.parent)
    require(not destination.exists(), "new native bundle required")
    destination.mkdir(mode=0o700)
    shutil.copytree(c["sourceDirectory"], destination / "source")
    require(
        inventory(destination / "source", max_bytes=256 * 1024**2) == source,
        "source changed while freezing",
    )
    write_json(destination / "native-job.private.json", envelope)
    descriptor = dict(
        schema=SCHEMA,
        operation="mimo-heldout-evaluation",
        assurance=c["assurance"],
        authority=c["authority"],
        sourceRevision=c["sourceRevision"],
        jobSha256=_sha(job),
        manifestSha256=_sha(manifest),
        checkpointId=manifest["checkpoint"]["id"],
        sourceFiles=source,
        sourceInventoryDigest=canonical_sha256(source),
        checkpointFiles=files,
        checkpointInventoryDigest=canonical_sha256(files),
        limits=c["limits"],
    )
    descriptor.update(
        sourceClosureDigest=canonical_sha256(source),
        runtimeDigest=canonical_sha256(
            {
                "versions": job["runtime_versions"],
                "runtime": manifest["mimo_runtime"],
                "backend": manifest["backend_profile"],
            }
        ),
        inputInventoryDigest=canonical_sha256({"envelope": envelope}),
        nativeJobExpiresAt=math.floor(job["expires_at"]),
    )
    descriptor["jobDigest"] = canonical_sha256(descriptor)
    write_json(destination / "job.private.json", descriptor)
    return {
        "schema": SCHEMA,
        "stage": "prepared",
        "jobDigest": descriptor["jobDigest"],
        "assurance": c["assurance"],
        "nativeAdmissionVerified": False,
        "gpuQualificationVerified": False,
        "activated": False,
    }


def load_bundle(bundle):
    bundle = Path(bundle)
    require(bundle.is_absolute(), "absolute private bundle required")
    _private_file(bundle / "job.private.json")
    _private_file(bundle / "native-job.private.json")
    d = read_json(bundle / "job.private.json")
    require(
        isinstance(d, dict)
        and d.get("schema") == SCHEMA
        and d.get("assurance") in ("fixture", "native-mimo-evaluation")
        and d.get("sourceRevision") == REVIEWED_REVISION
        and d.get("operation") == "mimo-heldout-evaluation"
        and d.get("jobDigest")
        == canonical_sha256({k: v for k, v in d.items() if k != "jobDigest"}),
        "native bundle digest differs",
    )
    _limits(d["limits"])
    envelope = read_json(bundle / "native-job.private.json", 4_000_000)
    job, manifest, selection = _admit(envelope, d["authority"])
    require(
        d["jobSha256"] == _sha(job)
        and d["manifestSha256"] == _sha(manifest)
        and d["nativeJobExpiresAt"] == math.floor(job["expires_at"])
        and d["checkpointId"] == manifest["checkpoint"]["id"]
        and d["sourceInventoryDigest"] == canonical_sha256(d["sourceFiles"])
        and d["sourceClosureDigest"] == canonical_sha256(d["sourceFiles"])
        and d["runtimeDigest"]
        == canonical_sha256(
            {
                "versions": job["runtime_versions"],
                "runtime": manifest["mimo_runtime"],
                "backend": manifest["backend_profile"],
            }
        )
        and d["inputInventoryDigest"] == canonical_sha256({"envelope": envelope})
        and d["checkpointInventoryDigest"] == canonical_sha256(d["checkpointFiles"])
        and {n: v["sha256"] for n, v in d["checkpointFiles"].items()}
        == manifest["checkpoint"]["files"]
        and inventory(bundle / "source", max_bytes=256 * 1024**2) == d["sourceFiles"]
        and {
            n: v["sha256"]
            for n, v in d["sourceFiles"].items()
            if n.startswith("subnet/") and n.endswith(".py")
        }
        == job["source_files"],
        "frozen native input binding differs",
    )
    if d["assurance"] != "fixture":
        require(
            job["source_files"]["subnet/mimo_evaluation.py"] == EVALUATOR_SHA256,
            "native evaluator contract differs",
        )
    return d, job, manifest, selection


def verify_report(bundle, report_path):
    """Check complete original native report structure; no recomputation claim."""
    d, job, m, selection = load_bundle(bundle)
    report_path = Path(report_path)
    _private_file(report_path)
    report = read_json(
        report_path, min(MAX_REPORT_BYTES, d["limits"]["maxOutputBytes"])
    )
    expected = dict(
        job_id=job["job_id"],
        job_sha256=d["jobSha256"],
        operator=d["authority"],
        worker_policy=WORKER_POLICY,
        role="evaluate",
        epoch=m["epoch"],
        checkpoint=d["checkpointId"],
        source_files=job["source_files"],
        runtime_versions=job["runtime_versions"],
        backend_profile=m["backend_profile"],
        numerical_policy=m["sampling_contract"]["calibration"],
        chain_transactions=False,
        success=True,
    )
    require(
        isinstance(report, dict)
        and all(canonical(report.get(k)) == canonical(v) for k, v in expected.items()),
        "native report job/source/runtime binding differs",
    )
    completed = report.get("completed_at")
    require(
        type(completed) in (int, float)
        and math.isfinite(completed)
        and job["created_at"] <= completed < job["expires_at"]
        and completed <= time.time() + 5,
        "native report original deadline differs",
    )
    value = report.get("evaluation")
    expected = dict(
        version=VERSION,
        run_id=m["run_id"],
        epoch=m["epoch"],
        checkpoint=d["checkpointId"],
        manifest_sha256=d["manifestSha256"],
        runtime_sha256=_sha(m["mimo_runtime"]),
        dataset_id=selection["datasetId"],
        suite=selection["suite"],
        draw_policy=DRAW_POLICY,
        harness=selection["harness"],
        answer_projection=m["mimo_answer_projection"],
        payable=False,
        weight_submission=False,
        sampling_audit_performed=False,
    )
    require(
        isinstance(value, dict)
        and all(canonical(value.get(k)) == canonical(v) for k, v in expected.items()),
        "native evaluation experiment binding differs",
    )
    rows, suite = value.get("rows"), selection["suite"]
    wanted = [(i, r) for i in suite["indices"] for r in range(suite["repeats"])]
    require(
        isinstance(rows, list) and len(rows) == len(wanted),
        "every original evaluation attempt is required",
    )
    counts = dict.fromkeys(("positive", "negative", "neutral", "indeterminate"), 0)
    stops, horizon, support = (
        set(m["mimo_runtime"]["eos_token_ids"]),
        selection["harness"]["max_output_tokens"],
        m["group_transport_policy"]["vocab_size"],
    )
    for row, (index, repeat) in zip(rows, wanted, strict=True):
        require(
            isinstance(row, dict)
            and type(row.get("index")) is int
            and type(row.get("repeat")) is int
            and (row["index"], row["repeat"]) == (index, repeat),
            "ordered native attempt coverage differs",
        )
        tokens, kind = row.get("output"), row.get("classification")
        require(
            isinstance(tokens, list)
            and len(tokens) <= horizon
            and row.get("output_length") == len(tokens)
            and all(_integer(t, 0, support - 1) for t in tokens)
            and kind in counts,
            "native output token geometry differs",
        )
        if kind == "indeterminate":
            require(
                row.get("reward") is None and isinstance(row.get("error"), dict),
                "infrastructure failure cannot be a wrong answer",
            )
        else:
            require(
                tokens
                and type(row.get("reward")) in (int, float)
                and row["reward"] == float(kind == "positive")
                and row.get("error") is None,
                "native reward semantics differ",
            )
            require(
                all(
                    isinstance(row.get(k), str) and HEX.fullmatch(row[k])
                    for k in ("task_hash", "prompt_sha256")
                ),
                "completed native task and prompt hashes required",
            )
            if row.get("stop_reason") == "budget":
                require(
                    len(tokens) == horizon
                    and not stops.intersection(tokens)
                    and kind == "neutral",
                    "budget-capped outputs remain unresolved",
                )
            else:
                require(
                    row.get("stop_reason") == "eos"
                    and tokens[-1] in stops
                    and not stops.intersection(tokens[:-1]),
                    "native output must stop at its first approved EOS",
                )
        counts[kind] += 1
    summary = dict(
        requested_count=len(rows),
        count=len(rows) - counts["indeterminate"],
        successes=counts["positive"],
        completed_wrong=counts["negative"],
        unresolved=counts["neutral"],
        error_count=counts["indeterminate"],
        mean_reward=counts["positive"] / len(rows)
        if not counts["indeterminate"]
        else None,
        status="error" if counts["indeterminate"] else "complete",
    )
    require(
        all(canonical(value.get(k)) == canonical(v) for k, v in summary.items()),
        "native aggregate must cover all original attempts",
    )
    return dict(
        schema="affine-native-evaluation-result/v1",
        stage="evaluation_completed",
        operation=d["operation"],
        assurance=d["assurance"],
        **{k: d[k] for k in PINS},
        nativeJobExpiresAt=d["nativeJobExpiresAt"],
        jobSha256=d["jobSha256"],
        manifestSha256=d["manifestSha256"],
        checkpointId=d["checkpointId"],
        sourceInventoryDigest=d["sourceInventoryDigest"],
        datasetId=selection["datasetId"],
        nativeResultDigest=checksum(report_path),
        outputInventoryDigest=canonical_sha256(
            {
                "native-report.json": {
                    "sha256": checksum(report_path),
                    "bytes": report_path.stat().st_size,
                }
            }
        ),
        nativeReportSha256=checksum(report_path),
        **summary,
        gpuQualificationVerified=False,
        nativeParityVerified=False,
        operatorAccepted=False,
        chainTransactions=False,
    )


def _native_command(bundle, workspace, checkpoint, d, job):
    require(
        os.getuid() != 0
        and os.geteuid() != 0
        and platform.system() == "Linux"
        and Path("/.dockerenv").is_file(),
        "native execution requires an unprivileged qualified container",
    )
    require(sys.version_info[:2] == (3, 12), "native Python runtime differs")
    require(
        {name for _, name in socket.if_nameindex()} <= {"lo"},
        "native evaluation requires a network-disabled container",
    )
    for name, expected in job["runtime_versions"].items():
        require(
            importlib.metadata.version(name) == expected,
            "native package runtime differs",
        )
    for path in (bundle / "source", checkpoint):
        require(
            os.statvfs(path).f_flag & os.ST_RDONLY,
            "native source and checkpoint mounts must be read-only",
        )
    require(
        {p.name: _member(p) for p in checkpoint.iterdir()} == d["checkpointFiles"],
        "native local checkpoint bytes differ",
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
        "native GPU requires H200/SM90 qualification",
    )
    bootstrap = (
        "import json,pathlib,sys,time;sys.path.insert(0,sys.argv[1]);"
        "from subnet.backend_jobs import execute;"
        "envelope=json.loads(pathlib.Path(sys.argv[2]).read_bytes());"
        "lease=pathlib.Path(sys.argv[5]);expected=lease.read_bytes();"
        "guard=lambda:None if lease.read_bytes()==expected and time.time()<float(sys.argv[6]) else (_ for _ in ()).throw(RuntimeError('lost allocation lease'));"
        "execute(envelope,sys.argv[3],sys.argv[4],cache=sys.argv[7],assert_lease=guard,allow_public_mimo=True)"
    )
    return [
        sys.executable,
        "-I",
        "-B",
        "-c",
        bootstrap,
        str(bundle / "source"),
        str(bundle / "native-job.private.json"),
        d["authority"],
        str(workspace / "native"),
        str(workspace / "lease.private.json"),
        str(job["expires_at"]),
        str(checkpoint),
    ]


def run(
    bundle,
    workspace,
    *,
    checkpoint=None,
    fixture=False,
    stop_event=None,
    lease_check=None,
    max_seconds=None,
):
    """Supervise one allocation; the host still owns its cgroup/quota/deadline."""
    bundle, workspace = Path(bundle), Path(workspace)
    require(
        bundle.is_absolute() and workspace.is_absolute(),
        "absolute private run paths required",
    )
    require(
        bundle != workspace
        and bundle not in workspace.parents
        and workspace not in bundle.parents,
        "native workspace must be separate from its frozen bundle",
    )
    d, job, _, _ = load_bundle(bundle)
    seconds = d["limits"]["maxSeconds"] if max_seconds is None else max_seconds
    require(
        type(seconds) in (int, float)
        and math.isfinite(seconds)
        and 0 < seconds <= d["limits"]["maxSeconds"],
        "allocation may only shorten the prepared deadline",
    )
    _admit(
        read_json(bundle / "native-job.private.json", 4_000_000),
        d["authority"],
        now=time.time(),
    )
    require(
        fixture == (d["assurance"] == "fixture"),
        "fixture and native execution cannot be interchanged",
    )
    require(
        fixture or (checkpoint is not None and callable(lease_check)),
        "native checkpoint and host lease assertion required",
    )
    _private_path(workspace)
    _directory(workspace)
    with _lock(workspace):
        require(
            not (workspace / "state.private.json").exists(),
            "previous native attempt needs reconciliation and a new workspace",
        )
        (workspace / "native").mkdir(mode=0o700)
        write_json(workspace / "lease.private.json", {"jobDigest": d["jobDigest"]})
        if fixture:
            bootstrap = "import sys;sys.path.insert(0,sys.argv[1]);from reliquary.integrations.affine_native_evaluation import _fixture_worker;_fixture_worker(sys.argv[2],sys.argv[3])"
            command = [
                sys.executable,
                "-I",
                "-B",
                "-c",
                bootstrap,
                str(Path(__file__).parents[2]),
                str(bundle),
                str(workspace),
            ]
        else:
            command = _native_command(bundle, workspace, Path(checkpoint), d, job)
        started = time.monotonic()
        deadline = (
            started
            + min(seconds, job["expires_at"] - time.time())
            - d["limits"]["finalizationReserveSeconds"]
        )
        require(
            deadline > started, "original native deadline leaves no execution budget"
        )
        require(
            stop_event is None or not stop_event.is_set(),
            "cancelled before native allocation",
        )
        if lease_check is not None:
            require(lease_check() is not False, "lost host allocation lease")
        env = {
            k: os.environ[k]
            for k in (
                "PATH",
                "LD_LIBRARY_PATH",
                "CUDA_VISIBLE_DEVICES",
                "NVIDIA_VISIBLE_DEVICES",
                "NVIDIA_DRIVER_CAPABILITIES",
                "LANG",
            )
            if k in os.environ
        }
        state = {"schema": SCHEMA, "jobDigest": d["jobDigest"], "stage": "running"}
        reason = "failed"
        with (workspace / "runtime.private.log").open("xb") as log:
            os.chmod(log.name, 0o600)
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                env={
                    **env,
                    "HF_HUB_OFFLINE": "1",
                    "TRANSFORMERS_OFFLINE": "1",
                    "PYTHONNOUSERSITE": "1",
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                },
            )
            state["pid"] = process.pid
            exit_watch, exited = None, False
            try:
                write_json(workspace / "state.private.json", state)
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
                while True:
                    if stop_event is not None and stop_event.is_set():
                        reason = "cancelled"
                        break
                    if lease_check is not None:
                        require(
                            lease_check() is not False, "lost host allocation lease"
                        )
                    if time.monotonic() >= deadline:
                        reason = "budget_exhausted"
                        break
                    paths = list(workspace.rglob("*"))
                    require(
                        len(paths) <= 8192 and not any(p.is_symlink() for p in paths),
                        "invalid native workspace tree",
                    )
                    if (
                        Path(log.name).stat().st_size > d["limits"]["maxLogBytes"]
                        or sum(p.stat().st_size for p in paths if p.is_file())
                        > d["limits"]["maxWorkspaceBytes"]
                    ):
                        reason = "storage_limit_exceeded"
                        break
                    if (
                        exit_watch.control(None, 1, 0)
                        if exit_watch
                        else os.waitid(
                            os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT
                        )
                    ):
                        exited = True
                        reason = "process_exited"
                        break
                    time.sleep(0.05)
            finally:
                (workspace / "lease.private.json").unlink(missing_ok=True)
                for sig in (signal.SIGTERM, signal.SIGKILL):
                    try:
                        os.killpg(process.pid, sig)
                    except ProcessLookupError:
                        pass
                    except PermissionError:
                        if exit_watch and not exited:
                            exited = bool(exit_watch.control(None, 1, 0.1))
                        if not (exit_watch and exited):
                            raise
                process.wait(timeout=10)
                if exit_watch:
                    exit_watch.close()
                if reason == "process_exited" and process.returncode != 0:
                    reason = "failed"
                state["stage"] = reason
                write_json(workspace / "state.private.json", state, replace=True)
        require(reason == "process_exited", "native allocation did not complete")
        result = verify_report(
            bundle, workspace / "native" / "jobs" / job["job_id"] / "report.json"
        )
        result["elapsedSeconds"] = time.monotonic() - started
        result["nativeAdmissionVerified"] = not fixture
        write_json(workspace / "evaluation-result.private.json", result)
        state["stage"] = "report_checked"
        write_json(workspace / "state.private.json", state, replace=True)
        return result


def native_mounts(bundle, workspace, checkpoint):
    load_bundle(bundle)
    paths = [Path(bundle), Path(workspace), Path(checkpoint)]
    for path in paths:
        require(path.is_absolute(), "absolute native mount paths required")
        _private_path(path)
    require(
        all(
            a != b and a not in b.parents
            for i, a in enumerate(paths)
            for j, b in enumerate(paths)
            if i != j
        ),
        "native mount trees must be separate",
    )
    return [
        (str(Path(bundle)), str(NATIVE_ROOT / "bundle"), True),
        (str(Path(workspace)), str(NATIVE_ROOT / "work"), False),
        (str(Path(checkpoint)), str(NATIVE_ROOT / "checkpoint"), True),
    ]


def verify(bundle, workspace):
    d, job, _, _ = load_bundle(bundle)
    workspace = Path(workspace)
    verified = verify_report(
        bundle, workspace / "native" / "jobs" / job["job_id"] / "report.json"
    )
    result = read_json(workspace / "evaluation-result.private.json")
    require(
        all(canonical(result.get(k)) == canonical(v) for k, v in verified.items()),
        "saved evaluation result differs",
    )
    require(
        result.get("nativeAdmissionVerified") is (d["assurance"] != "fixture"),
        "saved evaluation admission assurance differs",
    )
    require(
        type(result.get("elapsedSeconds")) in (int, float)
        and math.isfinite(result["elapsedSeconds"])
        and 0
        <= result["elapsedSeconds"]
        <= load_bundle(bundle)[0]["limits"]["maxSeconds"],
        "evaluation allocation timing differs",
    )
    return result


def export(bundle, workspace, destination):
    d, job, _, _ = load_bundle(bundle)
    workspace, destination = Path(workspace), Path(destination)
    report = workspace / "native" / "jobs" / job["job_id"] / "report.json"
    result = verify(bundle, workspace)
    with os.fdopen(_private_open(report, os.O_RDONLY), "rb") as source:
        report_bytes = source.read(min(MAX_REPORT_BYTES, d["limits"]["maxOutputBytes"]) + 1)
    require(
        hashlib.sha256(report_bytes).hexdigest() == result["nativeReportSha256"]
        and canonical_sha256({"native-report.json": {"sha256": result["nativeReportSha256"], "bytes": len(report_bytes)}})
        == result["outputInventoryDigest"],
        "native report changed before archive delivery",
    )
    _private_path(destination)
    _directory(destination.parent)
    fd = _private_open(destination, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    with (
        os.fdopen(fd, "wb") as stream,
        zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_DEFLATED) as archive,
    ):
        archive.writestr("evaluation-result.json", canonical(result))
        archive.writestr("native-report.json", report_bytes)
    require(
        destination.stat().st_size <= d["limits"]["maxOutputBytes"],
        "native archive byte limit exceeded",
    )
    return {
        "stage": "exported",
        "jobDigest": d["jobDigest"],
        "assurance": d["assurance"],
        "artifactSha256": checksum(destination),
        "artifactBytes": destination.stat().st_size,
    }


def _fixture_worker(bundle, workspace):
    """Synthetic transport fixture, never an inference or native grading result."""
    d, job, m, selection = load_bundle(bundle)
    require(d["assurance"] == "fixture", "fixture bundle required")
    rows = [
        dict(
            index=i,
            repeat=r,
            task_hash=_sha(["fixture-task", i]),
            prompt_sha256=_sha(["fixture-prompt", i]),
            output=[m["mimo_runtime"]["eos_token_ids"][0]],
            output_length=1,
            stop_reason="eos",
            classification="positive",
            reward=1.0,
            error=None,
        )
        for i in selection["suite"]["indices"]
        for r in range(selection["suite"]["repeats"])
    ]
    value = dict(
        version=VERSION,
        run_id=m["run_id"],
        epoch=m["epoch"],
        checkpoint=d["checkpointId"],
        manifest_sha256=d["manifestSha256"],
        runtime_sha256=_sha(m["mimo_runtime"]),
        dataset_id=selection["datasetId"],
        suite=selection["suite"],
        draw_policy=DRAW_POLICY,
        harness=selection["harness"],
        answer_projection=m["mimo_answer_projection"],
        rows=rows,
        requested_count=len(rows),
        count=len(rows),
        successes=len(rows),
        completed_wrong=0,
        unresolved=0,
        error_count=0,
        mean_reward=1.0,
        status="complete",
        payable=False,
        weight_submission=False,
        sampling_audit_performed=False,
    )
    report = dict(
        schema=1,
        job_id=job["job_id"],
        job_sha256=d["jobSha256"],
        operator=d["authority"],
        worker_policy=WORKER_POLICY,
        role="evaluate",
        epoch=m["epoch"],
        checkpoint=d["checkpointId"],
        source_files=job["source_files"],
        runtime_versions=job["runtime_versions"],
        backend_profile=m["backend_profile"],
        numerical_policy=m["sampling_contract"]["calibration"],
        chain_transactions=False,
        success=True,
        completed_at=time.time(),
        evaluation=value,
    )
    output = Path(workspace) / "native" / "jobs" / job["job_id"]
    _directory(output)
    write_json(output / "report.json", report)
