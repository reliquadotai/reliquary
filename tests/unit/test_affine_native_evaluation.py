"""Evaluate-only native transport and original report semantics, without GPUs."""

import base64
from contextlib import nullcontext
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import ModuleType, SimpleNamespace
import zipfile

from nacl.signing import SigningKey
import pytest

from reliquary.integrations import affine_native_evaluation as native
from reliquary.integrations.affine import AffineRuntimeError
from reliquary.integrations.affine_epoch import canonical


def save(path, value):
    path.write_bytes(canonical(value))
    path.chmod(0o600)


def sign(value):
    signer = SigningKey(b"\x07" * 32)
    return dict(
        payload=value,
        signer=signer.verify_key.encode().hex(),
        signature=base64.b64encode(signer.sign(canonical(value)).signature).decode(),
    )


def intake(tmp_path):
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    source, checkpoint = tmp_path / "source", tmp_path / "checkpoint"
    (source / "subnet").mkdir(mode=0o700, parents=True)
    checkpoint.mkdir(mode=0o700)
    program = source / "subnet/mimo_evaluation.py"
    program.write_text(
        "raise AssertionError('fixture never executes supplied source')\n"
    )
    program.chmod(0o600)
    for name, body in {
        "config.json": b"{}",
        "model.safetensors": b"synthetic-not-loadable",
    }.items():
        (checkpoint / name).write_bytes(body)
        (checkpoint / name).chmod(0o600)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoint.iterdir()
    }
    h = dict(
        version="text-tools-mimo-complete-group-v1",
        policy="autoregressive",
        max_output_tokens=4,
        max_context_tokens=16,
        temperature=1.0,
        top_p=0.95,
        top_k=20,
        enable_thinking=True,
    )
    m = dict(
        run_id="fixture-only",
        epoch="nonpayable-fixture",
        model_runtime_revision="qwen3_5-mimo-v1",
        checkpoint=dict(id=native._sha(files), files=files),
        evaluation_sets=[
            dict(name="fixed4", env_id="math", indices=[1, 2, 3, 4], seed=42, repeats=1)
        ],
        environments=[
            dict(
                env_id="math",
                indices=[0],
                spec=dict(max_turns=1, num_samples=8),
                harness=h,
            )
        ],
        mimo_runtime=dict(eos_token_ids=[9]),
        group_transport_policy=dict(vocab_size=16),
        mimo_answer_projection=dict(version="mimo-final-channel-math-v1"),
        backend_profile=dict(device="cuda", sm=[9, 0]),
        sampling_contract=dict(calibration={}),
    )
    job = dict(
        schema=1,
        worker_policy=native.WORKER_POLICY,
        role="evaluate",
        job_id="fixture-evaluation",
        created_at=time.time() - 5,
        expires_at=time.time() + 300,
        manifest=sign(m),
        source_files={"subnet/mimo_evaluation.py": native.checksum(program)},
        runtime_versions=dict(
            torch="fixture", transformers="fixture", toploc="fixture"
        ),
        evaluation=dict(version=native.VERSION, run_id=m["run_id"], set_name="fixed4"),
    )
    envelope = sign(job)
    save(tmp_path / "native-job.private.json", envelope)
    config = dict(
        schema="affine-native-evaluation-intake/v1",
        assurance="fixture",
        authority=envelope["signer"],
        jobEnvelope=str(tmp_path / "native-job.private.json"),
        sourceDirectory=str(source),
        sourceRevision=native.REVIEWED_REVISION,
        checkpointDirectory=str(checkpoint),
        limits=dict(
            maxSeconds=30,
            finalizationReserveSeconds=1,
            maxLogBytes=65536,
            maxOutputBytes=1_000_000,
            maxWorkspaceBytes=2_000_000,
        ),
        review=dict(sourceApproved=True, checkpointApproved=True, heldoutApproved=True),
    )
    path = tmp_path / "intake.private.json"
    save(path, config)
    return path, config, job, m


def prepared(tmp_path):
    path, config, job, manifest = intake(tmp_path)
    bundle = tmp_path / "bundle"
    native.prepare(path, bundle)
    return bundle, config, job, manifest


def report_fixture(bundle, tmp_path):
    workspace = tmp_path / "work"
    native.run(bundle, workspace, fixture=True)
    return workspace / "native/jobs/fixture-evaluation/report.json", workspace


def test_frozen_signed_job_runs_fixture_and_exports_owned_original_report(tmp_path):
    bundle, _, job, _ = prepared(tmp_path)
    descriptor, *_ = native.load_bundle(bundle)
    assert descriptor["nativeJobExpiresAt"] <= job["expires_at"]
    assert descriptor["nativeJobExpiresAt"] == int(job["expires_at"])
    assert all(len(descriptor[k]) == 64 for k in native.PINS)
    report, workspace = report_fixture(bundle, tmp_path)
    result = native.verify(bundle, workspace)
    assert result["successes"] == 4 and result["mean_reward"] == 1.0
    assert result["nativeAdmissionVerified"] is False
    assert result["nativeParityVerified"] is False
    assert result["gpuQualificationVerified"] is False
    assert result["chainTransactions"] is False
    assert str(tmp_path) not in json.dumps(result)
    archive = tmp_path / "evaluation.private.zip"
    exported = native.export(bundle, workspace, archive)
    with zipfile.ZipFile(archive) as source:
        assert set(source.namelist()) == {
            "native-report.json",
            "evaluation-result.json",
        }
        assert source.read("native-report.json") == report.read_bytes()
    assert exported["artifactSha256"] == native.checksum(archive)
    assert archive.stat().st_mode & 0o777 == 0o600
    with pytest.raises(ValueError, match="previous native attempt"):
        native.run(bundle, workspace, fixture=True)
    with pytest.raises((ValueError, AffineRuntimeError)):
        native.export(bundle, workspace, archive)


def test_native_cli_dispatch_preserves_fixture_boundary(tmp_path, capsys):
    from reliquary.integrations.affine_cli import main

    bundle, _, _, _ = prepared(tmp_path)
    prefix = ["native-evaluation"]
    assert main(prefix + ["check", "--bundle", str(bundle)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked["expired"] is False and checked["activated"] is False
    workspace = tmp_path / "cli-work"
    assert (
        main(
            prefix
            + [
                "run",
                "--bundle",
                str(bundle),
                "--workspace",
                str(workspace),
                "--fixture",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["assurance"] == "fixture" and result["nativeParityVerified"] is False
    assert (
        main(
            prefix + ["verify", "--bundle", str(bundle), "--workspace", str(workspace)]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out) == result


def test_export_refuses_report_changed_after_verification(tmp_path, monkeypatch):
    bundle, _, _, _ = prepared(tmp_path)
    report, workspace = report_fixture(bundle, tmp_path)
    original = native.verify

    def changed(*args):
        result = original(*args)
        report.write_bytes(report.read_bytes() + b" ")
        return result

    monkeypatch.setattr(native, "verify", changed)
    destination = tmp_path / "changed.private.zip"
    with pytest.raises(ValueError, match="report changed before archive"):
        native.export(bundle, workspace, destination)
    assert not destination.exists()


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "train"),
        ("capability", {}),
        ("miner_identity_file", "/private/seed"),
        ("submissions", []),
        ("training", {}),
    ],
)
def test_non_evaluation_authority_is_refused_before_allocating(tmp_path, field, value):
    path, config, job, _ = intake(tmp_path)
    job[field] = value
    save(Path(config["jobEnvelope"]), sign(job))
    with pytest.raises(ValueError):
        native.prepare(path, tmp_path / "bundle")
    assert not (tmp_path / "bundle").exists()


def test_invalid_signatures_expiry_and_heldout_overlap_are_refused(tmp_path):
    path, config, job, manifest = intake(tmp_path)
    for case in ("signature", "expired", "overlap", "external-assets"):
        changed = deepcopy(job)
        if case == "expired":
            changed.update(created_at=time.time() - 100, expires_at=time.time() - 1)
        if case == "overlap":
            m = deepcopy(manifest)
            m["environments"][0]["indices"] = [1]
            changed["manifest"] = sign(m)
        if case == "external-assets":
            m = deepcopy(manifest)
            m["task_assets"] = {"unprepared": {}}
            changed["manifest"] = sign(m)
        envelope = sign(changed)
        if case == "signature":
            envelope["payload"]["job_id"] = "changed-after-signature"
        save(Path(config["jobEnvelope"]), envelope)
        with pytest.raises(ValueError):
            native.prepare(path, tmp_path / "bundle")


@pytest.mark.parametrize("changed", ["source", "checkpoint", "extra-source", "symlink"])
def test_full_local_source_and_checkpoint_closure_are_required(tmp_path, changed):
    path, config, _, _ = intake(tmp_path)
    if changed == "source":
        Path(config["sourceDirectory"], "subnet/mimo_evaluation.py").write_text(
            "changed"
        )
    elif changed == "extra-source":
        Path(config["sourceDirectory"], "subnet/other.py").write_text("pass")
    elif changed == "checkpoint":
        Path(config["checkpointDirectory"], "model.safetensors").write_bytes(b"changed")
    else:
        member = Path(config["checkpointDirectory"], "model.safetensors")
        member.unlink()
        member.symlink_to(member.parent / "config.json")
    with pytest.raises((ValueError, AffineRuntimeError)):
        native.prepare(path, tmp_path / "bundle")


def test_fixture_cannot_be_promoted_or_run_as_native(tmp_path):
    bundle, config, *_ = prepared(tmp_path)
    with pytest.raises(ValueError, match="cannot be interchanged"):
        native.run(
            bundle, tmp_path / "work", checkpoint=Path(config["checkpointDirectory"])
        )
    config["assurance"] = "native-mimo-evaluation"
    path = tmp_path / "native-intake.private.json"
    save(path, config)
    with pytest.raises(ValueError, match="reviewed contract"):
        native.prepare(path, tmp_path / "native-bundle")


def test_frozen_inputs_cannot_contain_their_output_or_alias_mounts(tmp_path):
    path, config, *_ = intake(tmp_path)
    with pytest.raises(ValueError, match="separate from input"):
        native.prepare(path, Path(config["sourceDirectory"]) / "output")
    bundle = tmp_path / "bundle"
    native.prepare(path, bundle)
    with pytest.raises(ValueError, match="separate from its frozen"):
        native.run(bundle, bundle / "work", fixture=True)
    with pytest.raises(ValueError, match="mount trees must be separate"):
        native.native_mounts(bundle, tmp_path / "work", tmp_path / "work/checkpoint")


def test_native_report_failure_counts_follow_upstream_evaluator_golden(tmp_path):
    """Golden generated from reviewed evaluate/_summary: +, -, budget, infra."""
    bundle, _, _, manifest = prepared(tmp_path)
    report_path, _ = report_fixture(bundle, tmp_path)
    report = json.loads(report_path.read_bytes())
    value = report["evaluation"]
    value["rows"][1].update(classification="negative", reward=0.0)
    value["rows"][2].update(
        output=[1] * 4,
        output_length=4,
        stop_reason="budget",
        classification="neutral",
        reward=0.0,
    )
    value["rows"][3].update(
        output=[],
        output_length=0,
        stop_reason=None,
        classification="indeterminate",
        reward=None,
        error=dict(stage="native-grade", type="OSError"),
    )
    value.update(
        requested_count=4,
        count=3,
        successes=1,
        completed_wrong=1,
        unresolved=1,
        error_count=1,
        mean_reward=None,
        status="error",
    )
    save(report_path, report)
    result = native.verify_report(bundle, report_path)
    assert (
        result["count"],
        result["successes"],
        result["completed_wrong"],
        result["unresolved"],
        result["error_count"],
    ) == (3, 1, 1, 1, 1)
    assert result["status"] == "error" and result["mean_reward"] is None
    assert result["datasetId"] == native._sha(
        dict(
            version=native.VERSION,
            suite=manifest["evaluation_sets"][0],
            environment=manifest["environments"][0]["spec"],
            harness=manifest["environments"][0]["harness"],
            runtime=manifest["mimo_runtime"],
            answer_projection=manifest["mimo_answer_projection"],
            draw_policy=native.DRAW_POLICY,
        )
    )
    for mutate in (
        lambda r: r["evaluation"].update(mean_reward=0.25),
        lambda r: r["evaluation"]["rows"].reverse(),
        lambda r: r["evaluation"]["rows"].pop(),
        lambda r: r.update(chain_transactions=True),
        lambda r: r.update(completed_at=job_expiry(bundle)),
        lambda r: r["evaluation"]["rows"][2].update(classification="negative"),
    ):
        changed = deepcopy(report)
        mutate(changed)
        save(report_path, changed)
        with pytest.raises(ValueError):
            native.verify_report(bundle, report_path)


def job_expiry(bundle):
    return native.load_bundle(bundle)[1]["expires_at"]


def test_lost_host_lease_stops_child_and_preserves_failed_attempt(
    tmp_path, monkeypatch
):
    bundle, *_ = prepared(tmp_path)
    original = subprocess.Popen
    monkeypatch.setattr(
        native.subprocess,
        "Popen",
        lambda *args, **kwargs: original(
            [sys.executable, "-c", "import time;time.sleep(30)"], **kwargs
        ),
    )
    calls = 0

    def lease():
        nonlocal calls
        calls += 1
        if calls > 2:
            raise ValueError("lost host lease")

    workspace = tmp_path / "work"
    with pytest.raises(ValueError, match="lost host lease"):
        native.run(bundle, workspace, fixture=True, lease_check=lease)
    state = json.loads((workspace / "state.private.json").read_bytes())
    assert state["stage"] == "failed"
    assert not (workspace / "lease.private.json").exists()
    with pytest.raises(ProcessLookupError):
        os.kill(state["pid"], 0)
    assert not (workspace / "evaluation-result.private.json").exists()


def test_original_expiry_and_allocation_deadline_are_never_extended(
    tmp_path, monkeypatch
):
    path, config, job, _ = intake(tmp_path)
    save(path, config)
    bundle = tmp_path / "bundle"
    native.prepare(path, bundle)
    original = subprocess.Popen
    monkeypatch.setattr(
        native.subprocess,
        "Popen",
        lambda *args, **kwargs: original(
            [sys.executable, "-c", "import time;time.sleep(30)"], **kwargs
        ),
    )
    with pytest.raises(ValueError, match="did not complete"):
        native.run(bundle, tmp_path / "work", fixture=True, max_seconds=2)
    assert (
        json.loads((tmp_path / "work/state.private.json").read_bytes())["stage"]
        == "budget_exhausted"
    )
    assert job_expiry(bundle) == job["expires_at"]
    assert native.load_bundle(bundle)[0]["limits"]["maxSeconds"] == 30


@pytest.mark.parametrize("seconds", [-1, 31, float("nan")])
def test_requested_deadline_can_only_shorten_the_prepared_allocation(tmp_path, seconds):
    bundle, *_ = prepared(tmp_path)
    with pytest.raises(ValueError, match="only shorten"):
        native.run(bundle, tmp_path / "work", fixture=True, max_seconds=seconds)
    assert not (tmp_path / "work").exists()


def test_original_native_evaluator_report_contract(tmp_path, monkeypatch):
    """Optional offline golden: original evaluator, scripted runtime and grader."""
    source_path = os.environ.get("AFFINE_NATIVE_EVALUATOR_FILE")
    if not source_path:
        pytest.skip("original reviewed native evaluator file not supplied")
    raw = Path(source_path).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == native.EVALUATOR_SHA256
    path, config, job, manifest = intake(tmp_path)
    manifest["environments"][0]["spec"].update(max_output_tokens=4, config={})
    job["manifest"] = sign(manifest)
    save(Path(config["jobEnvelope"]), sign(job))
    bundle = tmp_path / "bundle"
    native.prepare(path, bundle)
    prefix = "native_evaluation_contract_fixture"
    package = ModuleType(prefix)
    package.__path__ = []
    monkeypatch.setitem(sys.modules, prefix, package)

    def module(name, **members):
        value = ModuleType(prefix + "." + name)
        value.__dict__.update(members)
        monkeypatch.setitem(sys.modules, value.__name__, value)
        return value

    torch = ModuleType("torch")
    torch.inference_mode = nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)
    module("storage", canonical=canonical)
    module("protocol", entries=lambda m: m["environments"])
    module("harness", normalize=lambda h: h)
    module(
        "mimo_runtime",
        validate_runtime_spec=lambda _: SimpleNamespace(
            max_context_tokens=16,
            temperature=1.0,
            top_p=0.95,
            top_k=20,
            enable_thinking=True,
        ),
    )
    module(
        "complete_group_distribution",
        MIMO_HARNESS="text-tools-mimo-complete-group-v1",
        distribution=lambda logits, *args: logits.token,
        pick=lambda token, draw: token,
    )
    module(
        "complete_group_sampling",
        validate_geometry=lambda *args: {"vocab_size": 16},
        token_geometry=lambda value: value,
    )

    class Session:
        def reset(self, index, seed):
            self.index = index
            return dict(
                task_hash=native._sha(["native-fixture-task", index]),
                messages=[dict(role="user", content=str(index))],
                tools=[],
            )

        def close(self):
            pass

    def grade(session, *args):
        if session.index == 4:
            raise OSError("synthetic infrastructure failure")
        return (
            None,
            {1: "positive", 2: "negative", 3: "neutral"}[session.index],
            float(session.index == 1),
        )

    module(
        "mimo_answer_projection",
        policy=lambda _: manifest["mimo_answer_projection"],
        grade=grade,
    )
    module(
        "environments",
        EnvironmentSpec=SimpleNamespace(
            from_dict=lambda value: SimpleNamespace(**value)
        ),
        create_session=lambda _: Session(),
    )

    class Logits:
        ndim, shape = 1, (16,)

        def __init__(self, token):
            self.token = token

        def __getitem__(self, key):
            return self

    class Runtime:
        checkpoint_id = manifest["checkpoint"]["id"]
        spec = SimpleNamespace(to_dict=lambda: manifest["mimo_runtime"])
        eos_token_ids, max_context_tokens, vocab_size, tokenizer = [9], 16, 16, None

        def guard(self):
            pass

        def prepare_prompt(self, messages, tools):
            return [int(messages[0]["content"])]

        def next_token_logits(self, inputs, past_key_values=None, **kwargs):
            index = inputs[0] if past_key_values is None else past_key_values[0]
            return Logits(1 if index == 3 else 9), (index, 1)

    evaluator = module("mimo_evaluation")
    exec(compile(raw, "reviewed-native-evaluator", "exec"), evaluator.__dict__)
    selection = evaluator.validate_job(job, manifest)
    evaluated = evaluator.evaluate(manifest, selection, Runtime())
    evaluator.validate_result(evaluated, job, manifest)
    report_path, _ = report_fixture(bundle, tmp_path)
    report = json.loads(report_path.read_bytes())
    report["evaluation"] = evaluated
    save(report_path, report)
    result = native.verify_report(bundle, report_path)
    assert [row["classification"] for row in evaluated["rows"]] == [
        "positive",
        "negative",
        "neutral",
        "indeterminate",
    ]
    assert (
        result["requested_count"],
        result["count"],
        result["successes"],
        result["error_count"],
    ) == (4, 3, 1, 1)
    assert result["status"] == "error" and result["mean_reward"] is None


def test_offline_preflight_prices_ceiling_and_never_qualifies_runtime(
    tmp_path, capsys, monkeypatch
):
    from reliquary.integrations.affine_native_preflight import preflight

    bundle, _, job, _ = prepared(tmp_path)
    wheels = tmp_path / "wheels"
    wheels.mkdir(mode=0o700)
    hashes = {}
    for name in job["runtime_versions"]:
        path = wheels / (name + "-fixture-py3-none-any.whl")
        with zipfile.ZipFile(path, "w") as z:
            z.writestr(
                name + ".dist-info/METADATA", "Name: " + name + "\nVersion: fixture\n"
            )
        path.chmod(0o600)
        hashes[name] = native.checksum(path)
    lock = tmp_path / "requirements.txt"
    lock.write_text(
        "\n".join(n + "==fixture --hash=sha256:" + h for n, h in hashes.items())
    )
    lock.chmod(0o600)
    config = dict(
        schema="affine-native-preflight/v1",
        maxSeconds=29,
        rateMicroUsdPerHour=7_000_000,
        maxCostMicroUsd=1_000_000,
        providerCeilingMicroUsdPerGpuHour=5_080_000,
        gpuCount=1,
        companyCostCeilingMicroUsd=1_000_000,
        image="runtime@sha256:" + "a" * 64,
        requirementsFile=str(lock),
        wheelDirectory=str(wheels),
    )
    path = tmp_path / "preflight.private.json"
    save(path, config)
    result = preflight(bundle, path)
    assert result["blockers"] == ["fixture_only"]
    from reliquary.integrations.affine_native_evaluation_cli import main

    assert main(["preflight", "--bundle", str(bundle), "--config", str(path)]) == 2
    summary = capsys.readouterr().out
    assert '"nativeExecutionReady": false' in summary and str(path) not in summary

    assert result["maxReservedMicroUsd"] == 56389
    assert result["maxProviderComputeMicroUsd"] == 40923
    assert result["runtimeInventory"]["wheelInventoryVerified"] is True
    assert not result["nativeExecutionReady"] and not result["activated"]
    assert not result["runtimeInventory"]["dependencyResolutionVerified"]
    descriptor = native.read_json(bundle / "job.private.json")
    synthetic = {**descriptor, "assurance": "native-mimo-evaluation"}
    synthetic["jobDigest"] = native.canonical_sha256(
        {k: v for k, v in synthetic.items() if k != "jobDigest"}
    )
    with monkeypatch.context() as context:
        context.setattr(
            native, "EVALUATOR_SHA256", job["source_files"]["subnet/mimo_evaluation.py"]
        )
        save(bundle / "job.private.json", synthetic)
        offline = preflight(bundle, path)
        assert offline["offlineReady"] is True
        assert offline["nativeExecutionReady"] is False
        assert offline["totalRentalCostQualified"] is False
    save(bundle / "job.private.json", descriptor)

    assert (
        "original_job_lifetime_insufficient"
        in preflight(bundle, path, now=job["expires_at"] - 1)["blockers"]
    )
    config["gpuCount"] = 2
    save(path, config)
    with pytest.raises(ValueError, match="bounded preflight"):
        preflight(bundle, path)
    config["gpuCount"] = 1
    config.update(maxCostMicroUsd=1, companyCostCeilingMicroUsd=1)
    save(path, config)
    result = preflight(bundle, path, now=job["expires_at"])
    assert set(result["blockers"]) == {
        "fixture_only",
        "original_job_lifetime_insufficient",
        "customer_budget_insufficient",
        "company_compute_budget_insufficient",
    }
    (wheels / "torch-fixture-py3-none-any.whl").unlink()
    assert "runtime_wheels_missing" in preflight(bundle, path)["blockers"]
    lock.write_text("torch==wrong --hash=sha256:" + "a" * 64)
    with pytest.raises(ValueError, match="fingerprint"):
        preflight(bundle, path)
