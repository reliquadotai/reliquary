"""Local, source-pinned batch experiments using an external official miner kit."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
from pathlib import Path
import signal
import subprocess
import sys
from urllib.parse import urlsplit

UPSTREAM = "https://github.com/AffineFoundation/affine"
REVISION = "46c3cb6a83abbc881341d374e86d438af91eedb5"
COMPARISON_RULE = "fixed-z-1-public-prior-v1"
APPROXIMATION = "PUBLIC PRIOR APPROXIMATION"
STAGES = ("plan", "generate", "validate", "train", "compare", "simulate")


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for part in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(part)
    return digest.hexdigest()


def _load(path, limit=64 * 1024 * 1024):
    if Path(path).stat().st_size > limit:
        raise ValueError("input_too_large")
    return json.loads(Path(path).read_text(), parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite_json")))


def _write(path, value):
    with open(path, "x", encoding="utf8", opener=lambda p, flags: os.open(p, flags, 0o600)) as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def inspect_kit(path):
    """Pin all miner files to the Git commit, separately recording the published manifest."""
    root = Path(path).resolve()
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    if git("rev-parse", "HEAD").decode().strip() != REVISION:
        raise ValueError("unsupported_kit_revision")
    entries = git("ls-tree", "-rz", "HEAD", "--", "miner").split(b"\0")
    files = {}
    for entry in filter(None, entries):
        metadata, name = entry.split(b"\t", 1)
        mode, kind, expected = metadata.split()
        rel = name.decode()
        target = root / rel
        if mode != b"100644" and mode != b"100755":
            raise ValueError("unsupported_kit_file")
        if kind != b"blob" or target.is_symlink():
            raise ValueError("modified_kit")
        raw = target.read_bytes()
        blob = hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest().encode()
        if blob != expected:
            raise ValueError("modified_kit")
        files[rel] = hashlib.sha256(raw).hexdigest()
    if not files or git("ls-files", "--others", "--", "miner").strip():
        raise ValueError("untracked_kit_files")
    mismatches = []
    for line in (root / "miner/validator/MANIFEST.sha256").read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        expected, rel = line.split()[:2]
        if files.get("miner/" + rel) != expected:
            mismatches.append(rel)
    return {"repository": UPSTREAM, "revision": REVISION,
            "sourceClosureDigest": hashlib.sha256(_canonical(files)).hexdigest(),
            "officialManifestVerified": not mismatches, "officialManifestMismatches": mismatches}


def checkpoint_digest(path):
    """Use the official train_step descriptor's exact top-level file selection."""
    root = Path(path)
    files = {p.name: _sha(p) for p in sorted(root.iterdir())
             if p.is_file() and not p.name.startswith(".") and p.suffix in (".json", ".safetensors", ".jinja")}
    if not files:
        raise ValueError("empty_checkpoint")
    return hashlib.sha256(_canonical(files)).hexdigest()


def _tokenizer_digest(path):
    root = Path(path)
    files = {p.name: _sha(p) for p in sorted(root.iterdir())
             if p.is_file() and (p.name.startswith("tokenizer") or p.suffix == ".jinja" or p.name == "special_tokens_map.json")}
    if not files:
        raise ValueError("missing_tokenizer")
    return hashlib.sha256(_canonical(files)).hexdigest()


def _url(value):
    parsed = urlsplit(value)
    if (parsed.scheme not in ("http", "https") or parsed.hostname not in ("127.0.0.1", "localhost", "::1")
            or parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("inference_url_must_be_loopback_origin")
    return value.rstrip("/")


def _clock(training_result_at, last_green_decided_at):
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x < 0
           for x in (training_result_at, last_green_decided_at)) or training_result_at < last_green_decided_at:
        raise ValueError("invalid_comparison_clock")
    return (training_result_at - last_green_decided_at) / 3600


def _master_digest(path):
    if path is None:
        return None
    root = Path(path)
    if not root.is_dir():
        raise ValueError("fp32_master_must_be_directory")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise ValueError("empty_fp32_master")
    return hashlib.sha256(_canonical({p.name: _sha(p) for p in files})).hexdigest()


def _constants(path, names):
    tree = ast.parse(Path(path).read_text())
    return {node.targets[0].id: ast.literal_eval(node.value) for node in tree.body
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in names}


def _train_policy(kit):
    return _constants(Path(kit) / "miner/validator/batch/ff_format.py",
                      ("TRAIN_BOUNDS", "TRAIN_DEFAULTS", "KNOB_SPEC", "STEP_MULT_ALLOWED", "MAX_OUTPUT", "MAX_CONTEXT", "LEGACY_CAP"))


def _gate_policy(kit):
    cfg = _load(Path(kit) / "miner/validator/gate/gate-config.public.json")["heldout"]["prior_v1"]
    expected = dict(N=500, k_cand=8, head_rollouts=12, head_blocks=3, scoring_mode="separate",
                    floor=0.1, cap=0.1, shares="size", n_init=2, sigma_run_floor=0.003,
                    bar_z=1, z_inclusive=False, cap_max=0.33, cap_rule="task-level",
                    min_scored=100, min_cover=0.95, block_delay=3, weighting="irt-info")
    if any((type(cfg.get(k)) is bool) != (type(v) is bool) or cfg.get(k) != v for k, v in expected.items()) or cfg.get("tau_run_separate") != 0 or cfg.get("tau_run_same") != 0 or cfg.get("z_bar_fixed_from_epoch") != 553:
        raise ValueError("unsupported_public_gate_policy")
    return cfg


def _mining(path, kit):
    value = _load(path, 1024 * 1024)
    caps = value["training"]["batch_cap"]
    tasks, rollouts = caps["max_tasks_per_miner"], caps["max_rollouts_per_miner"]
    if type(tasks) is not int or type(rollouts) is not int or not 1 <= tasks <= 125 or not 1 <= rollouts <= 1000:
        raise ValueError("unsupported_mining_caps")
    if not isinstance(value.get("epoch"), str) or not re.fullmatch(r"[\w.-]{1,128}", value["epoch"], flags=re.ASCII) or not isinstance(value.get("checkpoint"), str) or not re.fullmatch(r"[a-f0-9]{64}", value["checkpoint"]):
        raise ValueError("invalid_mining_identity")
    formats = value["batch_file"]["formats"]
    if not isinstance(formats, list) or len(formats) > 2 or "affine-batch-v1" not in formats or any(f not in ("affine-batch-v1", "affine-batch-v2") for f in formats):
        raise ValueError("unsupported_batch_formats")
    policy = _train_policy(kit)
    fields = value["batch_file"]["train"]["fields"]
    specs = {name: ("number", bounds, policy["TRAIN_DEFAULTS"][name]) for name, bounds in policy["TRAIN_BOUNDS"].items()}
    specs["step_mult"] = ("integer", policy["STEP_MULT_ALLOWED"], policy["TRAIN_DEFAULTS"]["step_mult"])
    specs.update(policy["KNOB_SPEC"])
    if len(specs) != 16 or set(fields) != set(specs):
        raise ValueError("unsupported_mining_train_policy")
    for name, (kind, limits, default) in specs.items():
        field = fields[name]
        if field.get("default") != default or (default is not None and type(field.get("default")) is bool):
            raise ValueError("unsupported_mining_train_policy")
        if kind == "number":
            if any(type(field.get(k)) not in (int, float) for k in ("min", "max")) or (field["min"], field["max"]) != tuple(limits) or field.get("type", "number") != "number":
                raise ValueError("unsupported_mining_train_policy")
        elif (field.get("type") != ("string" if kind == "choice" else kind)
              or field.get("choices") != list(limits)
              or kind == "integer" and (type(field.get("default")) is not int or any(type(x) is not int for x in field["choices"]))):
            raise ValueError("unsupported_mining_train_policy")
    if (policy["MAX_OUTPUT"], policy["MAX_CONTEXT"], policy["LEGACY_CAP"]) != (12288, 16384, 8192):
        raise ValueError("unsupported_token_policy")
    sampling = value["training"]["sampling"]
    harness = sampling["harness"]["affine_math"]
    if any(type(sampling.get(k)) not in (int, float) or sampling.get(k) != v for k, v in dict(temperature=1, top_p=0.95, top_k=20).items()) or harness.get("max_output_tokens") != policy["MAX_OUTPUT"] or harness.get("max_context_tokens") != policy["MAX_CONTEXT"] or harness.get("enable_thinking") is not True:
        raise ValueError("unsupported_sampling_policy")
    cfg = _gate_policy(kit)
    heldout = value["training"]["promotion_gate"]["heldout"]
    settings = heldout["current_round"]["settings"]
    expected = {k: cfg[k] for k in ("N", "k_cand", "head_rollouts", "head_blocks", "scoring_mode", "floor", "cap", "shares", "n_init", "sigma_run_floor", "bar_z", "z_inclusive", "cap_max", "cap_rule", "min_scored", "min_cover", "block_delay", "weighting")}
    expected.update(max_tokens=12288, max_context_tokens=16384, tau_run=0)
    if any(settings.get(k) != v or (type(settings.get(k)) is bool) != (type(v) is bool) for k, v in expected.items()) or heldout["z_bar"].get("kind") != "fixed" or heldout["z_bar"].get("current") != 1 or heldout["z_bar"].get("from_epoch") != 553:
        raise ValueError("unsupported_mining_gate_policy")
    return {"sha256": _sha(path), "epoch": value["epoch"], "checkpoint": value["checkpoint"],
            "maxTasks": tasks, "maxRollouts": rollouts, "formats": formats,
            "tokenLimits": {"maxOutput": 12288, "maxContext": 16384, "legacyCap": 8192},
            "comparisonClockRule": COMPARISON_RULE, "comparisonMode": APPROXIMATION}


def _bindings(args, source):
    return {"sourceClosureDigest": source["sourceClosureDigest"], "miningSha256": _mining(args.mining, args.kit)["sha256"], "headDigest": checkpoint_digest(args.head),
            "referenceDigest": checkpoint_digest(args.ref), "tokenizerDigest": _tokenizer_digest(args.head),
            "batchSha256": _sha(args.batch), "masterDigest": _master_digest(args.master), "priorSha256": _sha(args.prior) if args.prior else None,
            "gateConfigSha256": _sha(Path(args.kit) / "miner/validator/gate/gate-config.public.json"),
            "seed": args.seed, "expectedDraw": args.n}


def _work(value):
    path = Path(value).absolute()
    if path.is_symlink():
        raise ValueError("symlink_output")
    path = path.resolve()
    if any((parent / ".git").exists() for parent in (path, *path.parents)):
        raise ValueError("output_must_be_outside_repositories")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise ValueError("invalid_output_directory")
    if path.stat().st_mode & 0o077 or path.stat().st_uid != os.getuid():
        raise ValueError("output_directory_must_be_owner_private")
    return path


def _unused(*paths):
    if any(os.path.lexists(path) for path in paths):
        raise ValueError("output_exists_use_a_new_work_directory")


def _run(args, argv, label):
    work = _work(args.work)
    stdout, stderr = work / (label + ".stdout.private.log"), work / (label + ".stderr.private.log")
    _unused(stdout, stderr)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
           "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2"}
    project = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = project + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    process = None
    cancelled = False
    cleaning = False
    class Interrupted(Exception):
        pass
    def interrupt(signum, frame):
        nonlocal cancelled
        cancelled = True
        # Defer during Popen's constructor so a spawned child cannot lose its owner.
        if process is not None and not cleaning:
            raise Interrupted()
    old_handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        for sig in old_handlers:
            signal.signal(sig, interrupt)
        with open(stdout, "xb") as out, open(stderr, "xb") as err:
            os.chmod(stdout, 0o600); os.chmod(stderr, 0o600)
            if cancelled:
                raise Interrupted()
            process = subprocess.Popen([args.python, *map(str, argv)], stdout=out, stderr=err,
                                       env=env, start_new_session=True, umask=0o077)
            if cancelled:
                raise Interrupted()
            code = process.wait(timeout=args.timeout)
    except subprocess.TimeoutExpired:
        raise ValueError("stage_timeout") from None
    except Interrupted:
        raise ValueError("stage_interrupted") from None
    finally:
        cleaning = True
        try:
            if process is not None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                # A compiler/helper can remain in the owned group after its leader exits.
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        finally:
            for sig, previous in old_handlers.items():
                signal.signal(sig, previous)
    if code:
        raise ValueError("official_stage_failed_" + label)
    return stdout


def _script(args, name, flags, label):
    return _run(args, [Path(args.kit) / "miner/kit" / name, *flags], label)


def _require(args, *names):
    if any(getattr(args, name) is None for name in names):
        raise ValueError("missing_" + "_".join(names))


def plan(args):
    """Read local pins only; do not import the kit, create outputs or execute model code."""
    _require(args, "batch", "ref", "seed", "training_result_at", "last_green_decided_at")
    source = inspect_kit(args.kit)
    bindings = _bindings(args, source)
    document = _load(args.batch)
    if document.get("version") not in _mining(args.mining, args.kit)["formats"]:
        raise ValueError("batch_version_unavailable")
    train = document.get("train", {})
    clock = {"trainingResultAt": args.training_result_at, "lastGreenDecidedAt": args.last_green_decided_at}
    _clock(args.training_result_at, args.last_green_decided_at)
    common = ["python", "-m", "reliquary.eval.affine_batches"]
    base = ["--kit", "<KIT>", "--mining", "<MINING>", "--head", "<HEAD>", "--hotkey", args.hotkey, "--work", "<WORK>"]
    return {"schema": "affine-batch-plan/v1", "upstream": source,
            "mining": _mining(args.mining, args.kit),
            "batch": {"sha256": bindings["batchSha256"], "bytes": Path(args.batch).stat().st_size,
                      "format": document.get("version"), "hotkey": args.hotkey},
            "head": {"digest": bindings["headDigest"], "provenance": "caller-provided-unverified"},
            "reference": {"digest": bindings["referenceDigest"], "provenance": "caller-provided-unverified"},
            "training": {"master": "provided" if args.master else "bf16-bootstrap",
                         "lr": train.get("lr", 5e-7), "kl": train.get("kl", 0.02),
                         "parameters": train, "parameterValidation": "not_run"},
            "comparison": {"seed": args.seed, "prior": "provided" if args.prior else "env-average", **clock,
                           "barZ": 1.0, "comparisonClockRule": COMPARISON_RULE, "mode": APPROXIMATION,
                           "liveWeighting": "irt-info-private-unavailable", "localWeighting": "public-beta-prior"},
            "bindings": bindings, "structuralOnly": True, "tokenValidation": "not_run",
            "qualification": "not_run", "liveParity": False, "officialAcceptance": False, "chainSubmission": False,
            "commands": [{"stage": stage, "argv": common + [stage] + base + flags} for stage, flags in (
                ("validate", ["--batch", "<BATCH>"]),
                ("train", ["--batch", "<BATCH>", "--ref", "<REFERENCE>", "--device", "cpu", "--token-budget", "512"] + (["--master", "<MASTER>"] if args.master else [])),
                ("compare", ["--batch", "<BATCH>", "--ref", "<REFERENCE>", "--candidate", "<CANDIDATE>",
                             "--head-url", "<HEAD_URL>", "--candidate-url", "<CANDIDATE_URL>", "--seed", str(args.seed),
                             "--training-result-at", str(args.training_result_at), "--last-green-decided-at", str(args.last_green_decided_at), "--n", str(args.n)] + (["--prior", "<PRIOR>"] if args.prior else []) + (["--master", "<MASTER>"] if args.master else [])))]}


def _report(args, stage, source, result, mining):
    if _mining(args.mining, args.kit) != mining:
        raise ValueError("mining_snapshot_changed")
    if inspect_kit(args.kit) != source:
        raise ValueError("kit_changed_during_execution")
    value = {"schema": "affine-batch-stage-report/v1", "stage": stage, "upstream": source, "mining": mining, "result": result,
             "assurance": "source-pinned-local-simulation", "qualification": "not_run", "liveParity": False,
             "officialAcceptance": False, "chainSubmission": False}
    _write(_work(args.work) / (stage + ".private.json"), value)
    return value


def generate(args, source):
    _require(args, "head_url", "seed")
    _url(args.head_url)
    mining = _mining(args.mining, args.kit)
    work = _work(args.work)
    rollouts, batch = work / "rollouts.private.jsonl", work / "batch.private.json"
    _unused(rollouts, batch)
    _script(args, "generate.py", ["--head", args.head, "--url", args.head_url, "--tasks", args.tasks,
            "--k", args.samples, "--seed", args.seed, "--concurrency", args.concurrency, "--out", rollouts], "generate")
    _script(args, "make_batch.py", ["--head", args.head, "--hotkey", args.hotkey, "--rollouts", rollouts, "--out", batch,
            "--select", args.select, "--max-prompts", min(args.max_prompts, mining["maxTasks"]), "--per-prompt", args.per_prompt], "export")
    args.batch = str(batch)
    return _report(args, "generate", source, {"rolloutsSha256": _sha(rollouts), "batchSha256": _sha(batch), "batchPath": str(batch)}, mining)


def validate(args, source):
    _require(args, "batch")
    mining = _mining(args.mining, args.kit)
    tokenizer = _tokenizer_digest(args.head)
    flags = ["-m", "reliquary.eval.affine_batches", "_validate", "--kit", args.kit, "--mining", args.mining,
             "--head", args.head, "--batch", args.batch, "--hotkey", args.hotkey, "--work", args.work]
    output = _run(args, flags, "validate")
    checked = json.loads(output.read_text().strip().splitlines()[-1])
    if checked.get("ok") is not True or checked.get("summary", {}).get("file_sha256") != _sha(args.batch) or tokenizer != _tokenizer_digest(args.head):
        raise ValueError("batch_validation_failed")
    return _report(args, "validate", source, {"batchSha256": _sha(args.batch), "tokenValidation": "passed", **checked}, mining)


def train(args, source):
    _require(args, "batch", "ref", "device")
    if args.device not in ("cpu", "cuda"):
        raise ValueError("device_must_be_explicit_cpu_or_cuda")
    work = _work(args.work)
    candidate, report = work / "candidate", work / "step.private.json"
    _unused(candidate, report)
    before = {"headDigest": checkpoint_digest(args.head), "referenceDigest": checkpoint_digest(args.ref), "batchSha256": _sha(args.batch), "masterDigest": _master_digest(args.master), "tokenizerDigest": _tokenizer_digest(args.head)}
    caps = _mining(args.mining, args.kit)
    if _load(args.batch).get("version") not in caps["formats"]:
        raise ValueError("batch_version_unavailable")
    flags = ["--max-tasks", caps["maxTasks"], "--max-rollouts", caps["maxRollouts"], "--head", args.head, "--ref", args.ref, "--batch", args.batch, "--hotkey", args.hotkey, "--out", candidate,
             "--device", args.device, "--report", report, "--token-budget", args.token_budget]
    if args.master:
        flags += ["--master", args.master]
    _script(args, "train_step.py", flags, "train")
    result = _load(report)
    if result.get("report", {}).get("status") != "updated":
        raise ValueError("training_did_not_update")
    if before != {"headDigest": checkpoint_digest(args.head), "referenceDigest": checkpoint_digest(args.ref), "batchSha256": _sha(args.batch), "masterDigest": _master_digest(args.master), "tokenizerDigest": _tokenizer_digest(args.head)}:
        raise ValueError("training_inputs_changed")
    args.candidate = str(candidate)
    return _report(args, "train", source, {**before, "candidateDigest": checkpoint_digest(candidate),
                   "reportSha256": _sha(report), "candidatePath": str(candidate),
                   "master": "provided" if args.master else "bf16-bootstrap", "training": result["report"]}, caps)


def compare(args, source):
    _require(args, "batch", "ref", "candidate", "head_url", "candidate_url", "seed", "training_result_at", "last_green_decided_at")
    _url(args.head_url); _url(args.candidate_url)
    _clock(args.training_result_at, args.last_green_decided_at)
    mining = _mining(args.mining, args.kit)
    work = _work(args.work)
    output = work / "comparison.private.json"
    _unused(output)
    flags = ["-m", "reliquary.eval.affine_batches", "_compare", "--kit", args.kit, "--mining", args.mining, "--head", args.head,
             "--ref", args.ref, "--batch", args.batch, "--candidate", args.candidate, "--hotkey", args.hotkey,
             "--work", str(work), "--head-url", args.head_url, "--candidate-url", args.candidate_url,
             "--seed", args.seed, "--n", args.n, "--concurrency", args.concurrency,
             "--training-result-at", args.training_result_at, "--last-green-decided-at", args.last_green_decided_at]
    if args.prior:
        flags += ["--prior", args.prior]
    if args.master:
        flags += ["--master", args.master]
    _run(args, flags, "compare")
    return _report(args, "compare", source, _load(output), mining)


def _validate(args):
    inspect_kit(args.kit)
    sys.path.insert(0, str(Path(args.kit) / "miner/kit"))
    import kitpath  # noqa: F401
    import ff_format
    caps = _mining(args.mining, args.kit)
    if _load(args.batch).get("version") not in caps["formats"]:
        raise ValueError("batch_version_unavailable")
    result = ff_format.validate(Path(args.batch).read_bytes(), args.hotkey, ff_format.tokenizer(args.head),
                                caps["maxTasks"], caps["maxRollouts"], v2=True)
    print(json.dumps({"ok": True, "summary": result["summary"]}))


def _compare(args):
    """Compatibility boundary: official sampler, grader, draw and estimator remain unchanged."""
    _require(args, "batch", "ref", "candidate", "head_url", "candidate_url", "seed", "training_result_at", "last_green_decided_at")
    _url(args.head_url); _url(args.candidate_url)
    _work(args.work)
    _unused(Path(args.work) / "comparison.private.json")
    source = inspect_kit(args.kit)
    sys.path.insert(0, str(Path(args.kit) / "miner/kit"))
    import gate_eval as gate
    import baseline_v1 as baseline
    cfg = dict(baseline.DEFAULTS, **_gate_policy(args.kit))
    _clock(args.training_result_at, args.last_green_decided_at)
    bar = cfg["bar_z"]
    cfg["N"] = args.n
    bindings = _bindings(args, source)
    candidate_digest = checkpoint_digest(args.candidate)
    pool = gate.gate_pool()
    prior = {}
    if args.prior:
        with open(args.prior) as stream:
            for line in stream:
                if line.strip():
                    row = json.loads(line)
                    if row["id"] in prior or row["id"] not in pool or not all(isinstance(row[k], (int, float)) and not isinstance(row[k], bool) and math.isfinite(row[k]) for k in ("s", "n")) or not 0 <= row["s"] <= row["n"]:
                        raise ValueError("invalid_prior")
                    prior[row["id"]] = row
    rows = [dict(task_id=t, env=r["env"], s=prior.get(t, {}).get("s", 0), n=prior.get(t, {}).get("n", 0), b=0.0) for t, r in pool.items()]
    probabilities = baseline.inclusion(rows, N=cfg["N"], floor=cfg["floor"], cap=cfg["cap"], shares=cfg["shares"], n_init=cfg["n_init"])
    selected = baseline.draw(probabilities, args.seed)
    gate.grade.init(args.head)
    ids = {t: gate.prompts.gate_ids(gate.grade._G["tok"], pool[t]["problem"]) for t in selected}
    blocks = [gate.run([args.head_url], "m", baseline.plan_seed(args.seed, b), selected,
                      cfg["head_rollouts"] // cfg["head_blocks"], ids, pool, args.concurrency, cfg) for b in range(cfg["head_blocks"])]
    candidate = gate.run([args.candidate_url], "m", baseline.cand_seed(args.seed, 0), selected, cfg["k_cand"], ids, pool, args.concurrency, cfg)
    sizes = {}
    for row in pool.values():
        sizes[row["env"]] = sizes.get(row["env"], 0) + 1
    selection = {t: dict(env=pool[t]["env"], pi=probabilities[t]) for t in selected}
    head_cells = gate.merge(blocks)
    # The current estimator removed cap_envs and implements the task-level cap itself.
    decision = baseline.estimate(selection, head_cells, candidate, sizes,
                                shares=cfg["shares"], tau=cfg["tau_run_separate"], sigma_run_floor=cfg["sigma_run_floor"],
                                bar_z=bar, z_inclusive=cfg["z_inclusive"], cap_max=cfg["cap_max"], min_scored=cfg["min_scored"], min_cover=cfg["min_cover"])
    if bindings != _bindings(args, source) or candidate_digest != checkpoint_digest(args.candidate) or inspect_kit(args.kit) != source:
        raise ValueError("comparison_inputs_changed")
    _write(Path(args.work) / "comparison.private.json", {"bindings": bindings, "candidateDigest": checkpoint_digest(args.candidate),
           "decision": decision, "drawn": len(selected), "barZ": bar,
           "comparisonClockRule": COMPARISON_RULE, "mode": APPROXIMATION,
           "liveWeighting": "irt-info-private-unavailable", "localWeighting": "public-beta-prior",
           "headCells": head_cells, "candidateCells": candidate,
           "drawSelection": {"tasks": selection, "envSizes": sizes, "estimatorSettings": {
               "shares": cfg["shares"], "tau": cfg["tau_run_separate"], "sigma_run_floor": cfg["sigma_run_floor"],
               "bar_z": bar, "z_inclusive": cfg["z_inclusive"], "cap_max": cfg["cap_max"],
               "min_scored": cfg["min_scored"], "min_cover": cfg["min_cover"]}},
           "clock": {"trainingResultAt": args.training_result_at, "lastGreenDecidedAt": args.last_green_decided_at},
           "prior": "provided" if args.prior else "env-average", "baselineReused": False,
           "inferenceCheckpointIdentity": "caller-asserted-unverified", "headProvenance": "caller-provided-unverified"})


def parser():
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("stage", choices=(*STAGES, "_compare", "_validate"))
    for name in ("kit", "mining", "head", "hotkey", "work"):
        value.add_argument("--" + name, required=True)
    for name in ("ref", "batch", "master", "candidate", "head-url", "candidate-url", "prior", "device"):
        value.add_argument("--" + name)
    value.add_argument("--seed", type=int)
    for name in ("training-result-at", "last-green-decided-at"):
        value.add_argument("--" + name, type=float)
    for name, default in (("n", 500), ("samples", 4), ("max-prompts", 125), ("per-prompt", 8), ("token-budget", 512), ("concurrency", 4), ("timeout", 900)):
        value.add_argument("--" + name, type=int, default=default)
    value.add_argument("--tasks", default="0-3")
    value.add_argument("--select", choices=("correct1", "mixed", "all"), default="correct1")
    value.add_argument("--python", default=sys.executable, help="Interpreter containing the kit's CPU dependencies.")
    return value


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", args.hotkey):
            raise ValueError("invalid_hotkey")
        if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
            raise ValueError("invalid_seed")
        if not all(1 <= getattr(args, field) <= maximum for field, maximum in (
                ("n", 19305), ("samples", 8), ("max_prompts", 125), ("per_prompt", 8),
                ("token_budget", 16384), ("concurrency", 512), ("timeout", 86400))):
            raise ValueError("invalid_bounds")
        if args.stage in ("_compare", "_validate"):
            globals()[args.stage](args); return 0
        if args.stage == "plan":
            result = plan(args)
        else:
            source = inspect_kit(args.kit)
            _mining(args.mining, args.kit)
            _unused(Path(args.work) / (args.stage + ".private.json"))
            if args.stage == "simulate":
                _require(args, "ref", "device", "head_url", "candidate_url", "seed", "training_result_at", "last_green_decided_at")
                _url(args.head_url); _url(args.candidate_url)
                _clock(args.training_result_at, args.last_green_decided_at)
                if args.device not in ("cpu", "cuda"):
                    raise ValueError("device_must_be_explicit_cpu_or_cuda")
                if args.batch is None:
                    generate(args, source)
                validate(args, source); train(args, source)
                compared = compare(args, source)
                result = _report(args, "simulate", source, {"comparison": compared["result"],
                    "stageReports": {stage: _sha(Path(args.work) / (stage + ".private.json"))
                                     for stage in ("generate", "validate", "train", "compare")
                                     if (Path(args.work) / (stage + ".private.json")).exists()}}, _mining(args.mining, args.kit))
            else:
                result = globals()[args.stage](args, source)
        print(json.dumps(result, allow_nan=False))
        return 0
    except (OSError, ValueError, KeyError, subprocess.SubprocessError) as error:
        print(json.dumps({"error": str(error) if isinstance(error, ValueError) else "local_stage_failed",
                          "qualification": "not_run", "officialAcceptance": False, "chainSubmission": False}), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
