"""Trust boundaries of the local official-kit wrapper, without model dependencies."""
import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from reliquary.eval import affine_batches as batch


@pytest.fixture
def kit(tmp_path, monkeypatch):
    root = tmp_path / "kit"
    (root / "miner/validator/gate").mkdir(parents=True)
    (root / "miner/kit").mkdir()
    config = root / "miner/validator/gate/gate-config.public.json"
    config.write_text('{"heldout":{"prior_v1":{}}}')
    (root / "miner/kit/check_batch.py").write_text("# fixture\n")
    (root / "miner/validator/batch").mkdir()
    (root / "miner/validator/batch/ff_format.py").write_text("TRAIN_BOUNDS = {\"lr\": (1.25e-7, 1e-6), \"kl\": (0.005, 0.1)}\nTRAIN_DEFAULTS = {\"lr\": 5e-7, \"kl\": 0.02}\n")
    manifest = root / "miner/validator/MANIFEST.sha256"
    manifest.write_text("0" * 64 + "  validator/gate/gate-config.public.json\n")
    def git(*args):
        return subprocess.check_output(["git", "-C", str(root), *args], stderr=subprocess.DEVNULL)
    git("init", "-q")
    git("add", ".")
    git("-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid", "commit", "-qm", "Fixture")
    monkeypatch.setattr(batch, "REVISION", git("rev-parse", "HEAD").decode().strip())
    return root


@pytest.fixture
def args(tmp_path, kit):
    head, ref = tmp_path / "head", tmp_path / "ref"
    for path in (head, ref):
        path.mkdir()
        (path / "config.json").write_text('{"model_type":"fixture"}')
        (path / "tokenizer.json").write_text('{}')
    document = tmp_path / "batch.json"
    document.write_text(json.dumps({"version": "affine-batch-v1", "hotkey": "fixture", "groups": []}))
    mining = tmp_path / "mining.json"
    mining.write_text(json.dumps({"epoch": "fixture", "checkpoint": "a" * 64,
                                  "training": {"batch_cap": {"max_tasks_per_miner": 125, "max_rollouts_per_miner": 1000}},
                                  "batch_file": {"formats": ["affine-batch-v1", "affine-batch-v2"], "train": {"fields": {"lr": {"min": 1.25e-7, "max": 1e-6, "default": 5e-7}, "kl": {"min": 0.005, "max": 0.1, "default": 0.02}}}}}))
    return batch.parser().parse_args(["plan", "--kit", str(kit), "--mining", str(mining), "--head", str(head),
        "--ref", str(ref), "--batch", str(document), "--hotkey", "fixture", "--work", str(tmp_path / "output"),
        "--seed", "42", "--training-result-at", "21600", "--last-green-decided-at", "0"])


def test_source_commit_and_published_manifest_are_independent_pins(kit):
    result = batch.inspect_kit(kit)
    assert result["officialManifestVerified"] is False
    assert result["officialManifestMismatches"] == ["validator/gate/gate-config.public.json"]
    assert len(result["sourceClosureDigest"]) == 64
    (kit / "miner/kit/check_batch.py").write_text("changed")
    with pytest.raises(ValueError, match="modified_kit"):
        batch.inspect_kit(kit)


def test_untracked_or_symlinked_code_is_not_executed(kit):
    target = kit / "miner/extra.py"
    target.write_text("# unreviewed")
    with pytest.raises(ValueError, match="untracked_kit"):
        batch.inspect_kit(kit)
    target.unlink()
    original = kit / "miner/kit/check_batch.py"
    original.unlink()
    original.symlink_to(kit / "miner/validator/MANIFEST.sha256")
    with pytest.raises(ValueError, match="modified_kit"):
        batch.inspect_kit(kit)


def test_plan_is_declarative_and_binds_exact_inputs(args, monkeypatch):
    monkeypatch.setattr(batch, "_run", lambda *a: pytest.fail("plan must not run model code"))
    result = batch.plan(args)
    assert not Path(args.work).exists()
    assert result["qualification"] == result["tokenValidation"] == "not_run"
    assert result["officialAcceptance"] is result["chainSubmission"] is result["liveParity"] is False
    assert result["comparison"]["barZ"] == 0.625
    assert result["training"]["master"] == "bf16-bootstrap"
    assert result["mining"]["sha256"] == hashlib.sha256(Path(args.mining).read_bytes()).hexdigest()
    assert all("--mining" in command["argv"] for command in result["commands"])
    before = result["bindings"]
    Path(args.batch).write_text(Path(args.batch).read_text() + "\n")
    assert batch.plan(args)["bindings"]["batchSha256"] != before["batchSha256"]


def test_clock_is_explicit_and_not_wall_time(args):
    for value, expected in ((0, 1.0), (21600, 0.625), (43200, 0.25), (100000, 0.25)):
        args.training_result_at = value
        assert batch.plan(args)["comparison"]["barZ"] == expected
    args.last_green_decided_at = 100001
    with pytest.raises(ValueError, match="clock"):
        batch.plan(args)
    for value in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="clock"):
            batch._clock(value, 0)


def test_descriptor_matches_official_file_selection(args):
    root = Path(args.head)
    original = batch.checkpoint_digest(root)
    (root / "ignored.txt").write_text("unrelated")
    assert batch.checkpoint_digest(root) == original
    (root / "model.safetensors").write_bytes(b"weights")
    assert batch.checkpoint_digest(root) != original


def test_outputs_cannot_overwrite_or_enter_a_repository(tmp_path, kit):
    with pytest.raises(ValueError, match="outside_repositories"):
        batch._work(kit / "output")
    existing = tmp_path / "existing"
    existing.write_text("keep")
    with pytest.raises(ValueError, match="output_exists"):
        batch._unused(existing)
    link = tmp_path / "link"
    link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink_output"):
        batch._work(link)


def test_inference_urls_are_owned_loopback_origins():
    assert batch._url("http://127.0.0.1:8000/") == "http://127.0.0.1:8000"
    for value in ("https://example.com", "http://127.0.0.1/api", "http://user:pass@localhost", "http://localhost?token=x"):
        with pytest.raises(ValueError, match="loopback"):
            batch._url(value)


def test_training_requires_explicit_device_before_any_process(args, monkeypatch):
    monkeypatch.setattr(batch, "_script", lambda *a: pytest.fail("must not dispatch"))
    with pytest.raises(ValueError, match="missing"):
        batch.train(args, {})
    args.device = "auto"
    with pytest.raises(ValueError, match="explicit"):
        batch.train(args, {})


def test_bound_contract_refuses_unsupported_caps(args):
    value = json.loads(Path(args.mining).read_text())
    value["training"]["batch_cap"]["max_tasks_per_miner"] = True
    Path(args.mining).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="caps"):
        batch.plan(args)


def test_subprocess_timeout_keeps_private_logs_without_retries(args):
    args.timeout = 0.01
    with pytest.raises(ValueError, match="stage_timeout"):
        batch._run(args, ["-c", "import time; time.sleep(10)"], "timeout")
    assert (Path(args.work) / "timeout.stderr.private.log").stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="output_exists"):
        batch._run(args, ["-c", "raise AssertionError('must not run')"], "timeout")


def test_internal_comparison_cannot_bypass_loopback_boundary(args):
    args.head_url = "https://example.com"
    args.candidate_url = "http://localhost:8001"
    args.candidate = args.head
    with pytest.raises(ValueError, match="loopback"):
        batch._compare(args)


def test_master_prior_and_draw_count_survive_the_plan(args, tmp_path):
    master = tmp_path / "master"
    master.mkdir()
    tensor = master / "fp32.safetensors"
    tensor.write_bytes(b"original master")
    prior = tmp_path / "prior.jsonl"
    prior.write_text('{"id":"math:0","s":1,"n":2}\n')
    args.master, args.prior, args.n = str(master), str(prior), 16
    result = batch.plan(args)
    original = result["bindings"]["masterDigest"]
    assert result["training"]["master"] == "provided"
    train, compare = result["commands"][1:]
    assert "--master" in train["argv"]
    assert compare["argv"][-6:] == ["--n", "16", "--prior", "<PRIOR>", "--master", "<MASTER>"]
    tensor.write_bytes(b"changed master")
    assert batch.plan(args)["bindings"]["masterDigest"] != original


def test_mining_train_limits_must_match_the_frozen_official_parser(args):
    value = json.loads(Path(args.mining).read_text())
    value["batch_file"]["train"]["fields"]["lr"]["max"] = 1e-3
    Path(args.mining).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="train_policy"):
        batch.plan(args)


def test_fp32_master_is_an_official_directory_input(tmp_path):
    master = tmp_path / "master.safetensors"
    master.write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="master_must_be_directory"):
        batch._master_digest(master)


def test_batch_format_availability_is_bound_to_the_snapshot(args):
    value = json.loads(Path(args.mining).read_text())
    value["batch_file"]["formats"] = ["affine-batch-v1"]
    Path(args.mining).write_text(json.dumps(value))
    document = json.loads(Path(args.batch).read_text())
    document["version"] = "affine-batch-v2"
    Path(args.batch).write_text(json.dumps(document))
    with pytest.raises(ValueError, match="version_unavailable"):
        batch.plan(args)


def test_raw_outputs_require_an_owner_private_directory(tmp_path):
    work = tmp_path / "shared"
    work.mkdir(mode=0o755)
    work.chmod(0o755)
    with pytest.raises(ValueError, match="owner_private"):
        batch._work(work)


def test_sigterm_wrapper_stops_the_separate_official_child_group(tmp_path):
    import os
    import signal
    import sys
    import time

    marker = tmp_path / "child.pid"
    child = "import os,time; from pathlib import Path; Path(%r).write_text(str(os.getpid())); time.sleep(60)" % str(marker)
    wrapper = ("from types import SimpleNamespace; from reliquary.eval.affine_batches import _run; "
               "_run(SimpleNamespace(work=%r, python=%r, timeout=60), ['-c', %r], 'official')"
               % (str(tmp_path / "private-output"), sys.executable, child))
    process = subprocess.Popen([sys.executable, "-c", wrapper], start_new_session=True,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    pid = None
    def running(pid):
        try:
            status = Path('/proc/%d/stat' % pid)
            if status.exists() and status.read_text().rsplit(')', 1)[1].split()[0] == 'Z':
                return False
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
    try:
        for _ in range(100):
            if marker.exists():
                break
            time.sleep(.05)
        assert marker.exists(), "official child did not start"
        pid = int(marker.read_text())
        os.kill(process.pid, signal.SIGTERM)
        process.wait(timeout=10)
        for _ in range(100):
            if not running(pid):
                break
            time.sleep(.05)
        assert not running(pid), "SIGTERM left the official child running in its separate session"
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
        if pid and running(pid):
            os.killpg(pid, signal.SIGKILL)
