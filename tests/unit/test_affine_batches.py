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
    config.write_text(json.dumps({"heldout": {"prior_v1": {'N': 500, 'a_hi': 2.0, 'a_lo': 0.5, 'a_shrink': 0.5, 'bar_z': 1.0, 'block_delay': 3, 'cap': 0.1, 'cap_max': 0.33, 'cap_rule': 'task-level', 'dapo_offset': 7496, 'floor': 0.1, 'head_blocks': 3, 'head_rollouts': 12, 'k_cand': 8, 'math_offset': 0, 'min_cover': 0.95, 'min_scored': 100, 'n_init': 2.0, 'probe_env': 'affine_gatepool', 'scoring_mode': 'separate', 'shares': 'size', 'sigma_run_floor': 0, 'tau_run_same': 0.0, 'tau_run_separate': 0.0, 'theta_min_rollouts': 200, 'theta_prior_sd': 2.0, 'weighting': 'irt-info', 'z_bar_fixed_from_epoch': 553, 'z_inclusive': False}}}))
    (root / "miner/kit/check_batch.py").write_text("# fixture\n")
    (root / "miner/validator/batch").mkdir()
    (root / "miner/validator/batch/ff_format.py").write_text("TRAIN_DEFAULTS = {'lr': 5e-07, 'kl': 0.02, 'step_mult': 1}\nTRAIN_BOUNDS = {'lr': (1.25e-07, 1e-06), 'kl': (0.005, 0.1)}\nSTEP_MULT_ALLOWED = (1, 2)\nKNOB_SPEC = {'lr_scale_attn': ('number', (0.0, 2.0), 1.0), 'lr_scale_mlp': ('number', (0.0, 2.0), 1.0), 'lr_scale_embed': ('number', (0.0, 2.0), 1.0), 'lr_scale_lm_head': ('number', (0.0, 2.0), 1.0), 'lr_scale_early': ('number', (0.0, 2.0), 1.0), 'lr_scale_late': ('number', (0.0, 2.0), 1.0), 'adam_eps': ('number', (1e-10, 0.001), 1e-08), 'weight_decay': ('number', (0.0, 0.05), 0.01), 'clamp_mode': ('choice', ('per_tensor', 'global_norm', 'both'), 'per_tensor'), 'global_norm_cap': ('number', (0.005, 1.0), 1.0), 'loss_agg': ('choice', ('seq_mean', 'token_mean'), 'seq_mean'), 'adv_center': ('choice', ('mean', 'none', 'group_norm'), 'mean'), 'adv_clip': ('number', (0.1, 100.0), None)}\nMAX_OUTPUT = 12288\nMAX_CONTEXT = 16384\nLEGACY_CAP = 8192\n")
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
    policy = batch._train_policy(kit)
    fields = {k: dict(min=v[0], max=v[1], default=policy["TRAIN_DEFAULTS"][k]) for k, v in policy["TRAIN_BOUNDS"].items()}
    fields["step_mult"] = dict(type="integer", choices=[1, 2], default=1)
    for k, (kind, limits, default) in policy["KNOB_SPEC"].items():
        fields[k] = dict(type="number" if kind == "number" else "string", default=default,
                         **(dict(min=limits[0], max=limits[1]) if kind == "number" else dict(choices=list(limits))))
    cfg = batch._gate_policy(kit)
    settings = dict(cfg, max_tokens=12288, max_context_tokens=16384, tau_run=0)
    mining.write_text(json.dumps({"epoch": "fixture", "checkpoint": "a" * 64,
        "training": {"batch_cap": {"max_tasks_per_miner": 125, "max_rollouts_per_miner": 1000},
                     "sampling": {"temperature": 1, "top_p": 0.95, "top_k": 20,
                                  "harness": {"affine_math": {"max_output_tokens": 12288, "max_context_tokens": 16384, "enable_thinking": True}}},
                     "promotion_gate": {"heldout": {"current_round": {"settings": settings},
                         "z_bar": {"kind": "fixed", "current": 1, "from_epoch": 553}}}},
        "batch_file": {"formats": ["affine-batch-v1", "affine-batch-v2"], "train": {"fields": fields}}}))

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
    assert result["comparison"]["barZ"] == 1.0
    assert result["comparison"]["comparisonClockRule"] == "fixed-z-1-public-prior-v1"
    assert result["comparison"]["mode"] == "PUBLIC PRIOR APPROXIMATION"
    assert result["training"]["master"] == "bf16-bootstrap"
    assert result["mining"]["sha256"] == hashlib.sha256(Path(args.mining).read_bytes()).hexdigest()
    assert all("--mining" in command["argv"] for command in result["commands"])
    before = result["bindings"]
    Path(args.batch).write_text(Path(args.batch).read_text() + "\n")
    assert batch.plan(args)["bindings"]["batchSha256"] != before["batchSha256"]


def test_current_zero_noise_floor_is_pinned_and_old_floor_is_refused(args, kit):
    config = kit / "miner/validator/gate/gate-config.public.json"
    value = json.loads(config.read_text())
    value["heldout"]["prior_v1"]["sigma_run_floor"] = 0
    config.write_text(json.dumps(value))
    assert batch._gate_policy(kit)["sigma_run_floor"] == 0
    value["heldout"]["prior_v1"]["sigma_run_floor"] = 0.003
    config.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="unsupported_public_gate_policy"):
        batch._gate_policy(kit)


def test_generate_reuses_official_sampler_grader_and_export(args, kit, monkeypatch):
    args.head_url = "http://127.0.0.1:8000"
    mining = json.loads(Path(args.mining).read_text())
    mining["checkpoint"] = batch.checkpoint_digest(args.head)
    Path(args.mining).write_text(json.dumps(mining))
    calls = []
    def script(_args, name, flags, label):
        calls.append((name, flags, label))
        output = Path(flags[flags.index("--out") + 1])
        output.write_text('{"fixture":true}\n')
    monkeypatch.setattr(batch, "_script", script)
    result = batch.generate(args, batch.inspect_kit(kit))
    assert [name for name, _, _ in calls] == ["generate.py", "make_batch.py"]
    sampling = calls[0][1]
    assert sampling[sampling.index("--head") + 1] == args.head
    assert sampling[sampling.index("--url") + 1] == args.head_url
    assert sampling[sampling.index("--seed") + 1] == args.seed
    export = calls[1][1]
    assert export[export.index("--rollouts") + 1] == Path(args.work) / "rollouts.private.jsonl"
    assert export[export.index("--hotkey") + 1] == args.hotkey
    assert result["result"]["batchSha256"] == batch._sha(args.batch)
    assert result["officialAcceptance"] is result["liveParity"] is False


def test_generation_refuses_a_different_local_head_before_compute(args, kit, monkeypatch):
    args.head_url = "http://127.0.0.1:8000"
    monkeypatch.setattr(batch, "_script", lambda *a: pytest.fail("must not dispatch a mismatched head"))
    with pytest.raises(ValueError, match="generation_head_mismatch"):
        batch.generate(args, batch.inspect_kit(kit))
    assert not Path(args.work).exists()


def test_clock_is_explicit_and_not_wall_time(args):
    for value, expected in ((0, 1.0), (21600, 1.0), (43200, 1.0), (100000, 1.0)):
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


@pytest.mark.parametrize("field", ["step_mult", "adv_clip", "clamp_mode", "lr_scale_attn"])
def test_all_current_train_fields_are_bound(args, field):
    value = json.loads(Path(args.mining).read_text())
    del value["batch_file"]["train"]["fields"][field]
    Path(args.mining).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="train_policy"):
        batch.plan(args)


@pytest.mark.parametrize("section,key,bad", [
    ("sampling", "top_k", 40), ("sampling", "temperature", True),
    ("gate", "bar_z", 0.25), ("gate", "max_tokens", 8192),
    ("gate", "weighting", "unknown"), ("gate", "cap_max", 0.2),
    ("gate", "sigma_run_floor", 0.003),
])
def test_changed_current_contract_fails_closed(args, section, key, bad):
    value = json.loads(Path(args.mining).read_text())
    target = value["training"]["sampling"] if section == "sampling" else value["training"]["promotion_gate"]["heldout"]["current_round"]["settings"]
    target[key] = bad
    Path(args.mining).write_text(json.dumps(value))
    with pytest.raises(ValueError, match="policy"):
        batch.plan(args)


def test_comparison_uses_fixed_bar_and_reports_public_prior_approximation(args, monkeypatch):
    import sys
    from types import SimpleNamespace

    args.head_url, args.candidate_url, args.candidate = "http://localhost:8000", "http://localhost:8001", args.head
    calls = []
    pool = {"math:0": {"env": "math", "problem": "fixture"}}
    def run(urls, model, seed, selected, k, ids, tasks, concurrency, cfg):
        calls.append((urls, k))
        return {"math:0": {"n": k, "c": 0}}
    def estimate(selection, head, candidate, sizes, **settings):
        assert settings["bar_z"] == 1 and settings["cap_max"] == 0.33
        assert "cap_envs" not in settings
        return {"green": False}
    gate = SimpleNamespace(gate_pool=lambda: pool, grade=SimpleNamespace(init=lambda _: None, _G={"tok": None}),
                           prompts=SimpleNamespace(gate_ids=lambda *a: [1]), run=run, merge=lambda blocks: blocks[0])
    baseline = SimpleNamespace(DEFAULTS={}, inclusion=lambda *a, **kw: {"math:0": 0.1},
                               draw=lambda *a: ["math:0"], plan_seed=lambda *a: 1,
                               cand_seed=lambda *a: 2, estimate=estimate)
    monkeypatch.setitem(sys.modules, "gate_eval", gate)
    monkeypatch.setitem(sys.modules, "baseline_v1", baseline)
    batch._compare(args)
    result = json.loads((Path(args.work) / "comparison.private.json").read_text())
    assert result["comparisonClockRule"] == "fixed-z-1-public-prior-v1"
    assert result["mode"] == "PUBLIC PRIOR APPROXIMATION"
    assert result["liveWeighting"] == "irt-info-private-unavailable"
    assert [k for _, k in calls] == [4, 4, 4, 8]
    assert result["barZ"] == 1 and result["baselineReused"] is False
