"""Frozen experiment identities and private checkpoint trust boundaries."""

from __future__ import annotations

import asyncio
import hashlib
import json
import sys
import time
from types import SimpleNamespace

import pytest

from reliquary.eval import protocol as ep
from reliquary.eval.model_process import ModelProcess
from reliquary.eval.platform_client import PlatformClient, PlatformError
from reliquary.eval.runner import VLLMGenerator
from reliquary.eval.sets import build_set
from reliquary.eval.storage import PlatformEvalStore, SetConflict, _create
from tests.unit.test_eval_sets import opener


def request_protocol():
    return {"schema": ep.REQUEST_PROTOCOL, "sets": [{"set_id": "s", "env": "logic", "problems": 1, "samples": 1}],
            "protocol_bindings": {"s": "a" * 64}, "sampling": {"temperature": 1.0},
            "thinking": False, "max_new_tokens": 32, "harness": "single-turn", "max_turns": 1,
            "max_episode_tokens": 65536, "episode_timeout_seconds": 300, "interaction": "single_turn"}


def envelope(body, raw=None):
    raw = raw or ep.canonical(body).decode()
    return {"protocol": body, "protocol_json": raw,
            "protocol_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "protocol_bindings": body["protocol_bindings"], "sampling": body["sampling"],
            "thinking": body["thinking"], "max_new_tokens": body["max_new_tokens"]}


def test_request_hash_uses_exact_producer_json_not_python_number_spelling():
    body = request_protocol()
    raw = json.dumps(body, ensure_ascii=False).replace('1.0', '1e0')
    doc = envelope(body, raw)
    assert ep.validate_request(doc) == doc["protocol_sha256"]
    # Checkpoint changes are independent of the experiment protocol.
    assert ep.validate_request({**doc, "model": {"repo": "private/other", "revision": "b" * 40}}) == doc["protocol_sha256"]
    with pytest.raises(ValueError, match="mismatch"):
        ep.validate_request({**doc, "protocol": {**body, "thinking": True}})
    with pytest.raises(ValueError, match="duplicate"):
        ep.validate_request(envelope(body, raw.replace('"thinking": false', '"thinking": false, "thinking": false')))
    with pytest.raises(ValueError, match="non-finite"):
        ep.validate_request(envelope(body, raw.replace('1e0', 'NaN')))


def test_telecom_request_cannot_disable_its_fixed_thinking_renderer():
    body = {**request_protocol(), "harness": "telecom-episode-v1", "interaction": "episode", "max_turns": 40}
    with pytest.raises(ValueError, match="thinking renderer"):
        ep.validate_request(envelope(body))
    assert ep.validate_request(envelope({**body, "thinking": True}))
    assert ep.validate_request(envelope(request_protocol()))


def test_set_seals_actual_files_and_reviewed_rights_once(tmp_path):
    reviewed = {"data": {"dataset.json": "1" * 64}, "licenses": [{"id": "MIT", "sha256": "2" * 64}],
                "disjointness": {"disjoint": True, "evidence_sha256": "3" * 64}}
    card = build_set("logic", count=2, seed=1, out=tmp_path / "set",
                     open_environment=opener(), protocol=reviewed)
    assert ep.validate_set(card, current_grader=True) == card["protocol_sha256"]
    assert card["protocol"]["source"]["data"] == reviewed["data"]
    assert card["protocol"]["set"]["grading_sha256"] == hashlib.sha256((tmp_path / "set/grading.jsonl").read_bytes()).hexdigest()
    changed = {**card, "grading_sha256": "0" * 64}
    with pytest.raises(ValueError, match="files changed"):
        ep.validate_set(changed)
    with pytest.raises(FileExistsError):
        build_set("logic", count=2, seed=1, out=tmp_path / "set", open_environment=opener())


def test_grader_change_is_not_absorbed_into_an_old_set(tmp_path, monkeypatch):
    card = build_set("logic", count=2, seed=1, out=tmp_path / "set", open_environment=opener())
    monkeypatch.setattr(ep, "grader_sha256", lambda: "f" * 64)
    with pytest.raises(ValueError, match="grader drift"):
        ep.validate_set(card, current_grader=True)
    assert ep.validate_set({k: v for k, v in card.items() if not k.startswith('protocol')}) is None


def test_source_and_execution_implementation_drift_are_refused(tmp_path, monkeypatch):
    card = build_set("logic", count=1, seed=1, out=tmp_path / "set", open_environment=opener())
    for field, value in (("source_kind", "verifiers"), ("environment_manifest_sha256", "0" * 64)):
        with pytest.raises(ValueError, match="source changed"):
            ep.validate_set({**card, field: value})
    envelope = {k: card[k] for k in ("protocol", "protocol_json", "protocol_sha256")}
    bindings = {card["set_id"]: card["protocol_sha256"]}
    ep.validate_execution({card["set_id"]: envelope}, bindings)
    monkeypatch.setattr(ep, "harness_sha256", lambda *_: "f" * 64)
    # Grading retains the frozen execution provenance, but executing changed code refuses.
    assert ep.validate_set(card) == card["protocol_sha256"]
    with pytest.raises(ValueError, match="harness drift"):
        ep.validate_execution({card["set_id"]: envelope}, bindings)


def test_episode_image_identity_is_exact_digest_or_explicit_no_common_image():
    base = {"set_id": "s", "source": "fixture-telecom", "split": "benchmark", "count": 1,
        "prompts_sha256": "a" * 64, "grading_sha256": "b" * 64,
        "interaction": "episode", "source_kind": "episode"}
    episode = {"renderer_id": "fixture", "renderer_sha256": ep.telecom_renderer_sha256(),
               "tools_sha256": "c" * 64, "runtime_image_sha256": None}
    for image in (None, "d" * 64):
        card = ep.seal_set({**base, "episode": {**episode, "runtime_image_sha256": image}})
        ep.validate_execution({"s": {key: card[key] for key in ("protocol", "protocol_json", "protocol_sha256")}},
                              {"s": card["protocol_sha256"]})
    for image in ("unknown", "sha256:" + "d" * 64, False):
        with pytest.raises(ValueError, match="exact renderer/tool/runtime"):
            ep.seal_set({**base, "episode": {**episode, "runtime_image_sha256": image}})
    with pytest.raises(ValueError, match="exact renderer/tool/runtime"):
        ep.seal_set({**base, "episode": {key: value for key, value in episode.items()
                                       if key != "runtime_image_sha256"}})


def test_current_environment_manifest_is_checked_before_scoring(tmp_path, monkeypatch):
    from dataclasses import replace
    from reliquary.eval.grading import _Graders
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    card = build_set("logic", count=1, seed=1, out=tmp_path / "set", open_environment=opener())
    grading = json.loads((tmp_path / "set/grading.jsonl").read_text())
    graders = _Graders(lambda *_: pytest.fail("changed environment must not open"), None)
    changed = replace(ENVIRONMENT_SPECS[card["source"]], environment_manifest_sha256="f" * 64)
    monkeypatch.setattr(graders, "spec", lambda _: changed)
    assert graders.grade(grading, "answer", card)[0:2] == (
        None, "grader_drift: the installed environment manifest changed")


def test_public_taskset_identity_never_exports_private_configuration():
    from reliquary.eval.grading import _taskset_grading

    private = {"id": "fixture", "args": {"source": "fixture-sensitive-option"},
               "package": "fixture", "package_version": "1", "verifiers_version": "1",
               "artifact_sha256": "f" * 64}
    card = ep.seal_set({"set_id": "s", "count": 1, "prompts_sha256": "a" * 64,
        "grading_sha256": "b" * 64, "source": "verifiers:fixture", "source_kind": "verifiers",
        "split": "taskset", "taskset": private})
    public = card["protocol"]["source"]["taskset"]
    assert "args" not in public and public["args_sha256"] == ep.digest(private["args"])
    assert "fixture-sensitive-option" not in card["protocol_json"]
    assert card["taskset"] == private
    ep.validate_set(card)
    graders = SimpleNamespace(taskset_versions={json.dumps(["fixture", private["args"]], sort_keys=True): private})
    assert "fixture-sensitive-option" not in json.dumps(_taskset_grading(card, graders))


def test_frozen_package_hashes_real_wheel_code_and_refuses_editable_metadata(tmp_path, monkeypatch):
    from importlib.metadata import PathDistribution
    from reliquary.eval.verifiers_source import installed_artifacts

    metadata = tmp_path / "fixture-1.dist-info"
    metadata.mkdir()
    code = tmp_path / "fixture.py"
    code.write_text("VALUE = 1\n")
    direct = metadata / "direct_url.json"
    direct.write_text(json.dumps({"url": "https://example.test/source", "vcs_info": {"commit_id": "a" * 40}}))
    record = metadata / "RECORD"
    record.write_text("fixture.py,,\nfixture-1.dist-info/direct_url.json,,\n")
    monkeypatch.setattr("importlib.metadata.distribution", lambda _: PathDistribution(metadata))
    first = installed_artifacts("fixture")
    code.write_text("VALUE = 2\n")
    assert installed_artifacts("fixture") != first
    direct.write_text(json.dumps({"url": "file:///fixture", "dir_info": {"editable": True}}))
    with pytest.raises(ValueError, match="non-editable"):
        installed_artifacts("fixture")
    direct.write_text(json.dumps({"url": "https://example.test/source"}))
    record.write_text("fixture-1.dist-info/direct_url.json,,\n")
    with pytest.raises(ValueError, match="installed artifact"):
        installed_artifacts("fixture")


class AtomicS3:
    def __init__(self):
        import threading

        self.rows = {}
        self.lock = threading.Lock()

    def put_object(self, *, Bucket, Key, Body, IfNoneMatch):
        from botocore.exceptions import ClientError

        assert IfNoneMatch == "*"
        with self.lock:
            if Key in self.rows:
                raise ClientError({"Error": {"Code": "PreconditionFailed"}}, "PutObject")
            self.rows[Key] = Body

    def get_object(self, *, Bucket, Key):
        import io

        return {"Body": io.BytesIO(self.rows[Key])}


def test_publication_never_overwrites_a_concurrent_winner():
    s3 = AtomicS3()
    store = PlatformEvalStore(bucket="test", client=s3)

    async def run():
        assert await _create(store, "set.json", b"first")
        assert not await _create(store, "set.json", b"first")
        with pytest.raises(SetConflict):
            await _create(store, "set.json", b"second")

    asyncio.run(run())
    assert s3.rows["set.json"] == b"first"


def test_two_concurrent_publishers_cannot_both_replace_frozen_bytes():
    s3 = AtomicS3()
    store = PlatformEvalStore(bucket="test", client=s3)

    async def run():
        return await asyncio.gather(_create(store, "set.json", b"a"),
                                    _create(store, "set.json", b"b"), return_exceptions=True)

    result = asyncio.run(run())
    assert result.count(True) == 1 and sum(isinstance(r, SetConflict) for r in result) == 1
    assert s3.rows["set.json"] in {b"a", b"b"}


def _loader(tmp_path, monkeypatch, *, repo="org/private", revision="a" * 40):
    directory = tmp_path / ("models--" + repo.replace("/", "--")) / "snapshots" / revision
    directory.mkdir(parents=True)
    (directory / "config.json").write_text('{"max_position_embeddings":65536}')
    calls = []
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    monkeypatch.setitem(sys.modules, "vllm", SimpleNamespace(__version__="test", LLM=lambda **kw: calls.append(kw)))
    class Tokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            assert kwargs == {"trust_remote_code": False, "local_files_only": True}
            return lambda text, **kw: {"input_ids": [1, 2]}
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(AutoTokenizer=Tokenizer))
    import huggingface_hub
    def download(*args, **kwargs):
        calls.append(kwargs)
        return str(directory)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", download)
    return calls


def test_public_loader_never_uses_ambient_private_token(tmp_path, monkeypatch):
    calls = _loader(tmp_path, monkeypatch)
    monkeypatch.setenv("HF_TOKEN", "private-read-token")
    info = VLLMGenerator().load(repo="org/private", revision="a" * 40, gpu_count=1,
        max_new_tokens=32, rows=[{"text": "q"}], thinking=False)
    assert calls[0]["token"] is False and calls[1]["trust_remote_code"] is False
    assert "private-read-token" not in json.dumps(info)


def test_private_loader_is_pinned_and_does_not_export_secret(tmp_path, monkeypatch):
    calls = _loader(tmp_path, monkeypatch)
    info = VLLMGenerator().load(repo="org/private", revision="a" * 40, gpu_count=1,
        max_new_tokens=32, rows=[{"text": "q"}], thinking=False,
        model_access="private", hf_token="scoped-read-token", max_total_tokens=65536)
    assert calls[0]["token"] == "scoped-read-token" and info["model_sha"] == "a" * 40
    assert "scoped-read-token" not in json.dumps(info)
    with pytest.raises(RuntimeError, match="repository"):
        VLLMGenerator().load(repo="foreign/model", revision="a" * 40, gpu_count=1,
            max_new_tokens=32, rows=[{"text": "q"}], thinking=False)


def test_private_credential_must_match_claimed_model():
    model = {"repo": "org/private", "revision": "a" * 40}
    client = object.__new__(PlatformClient)
    client.task_id = "task"
    client._request = lambda *a, **kw: SimpleNamespace(json=lambda: {"token": "secret", "model": model})
    assert client.model_credential(model) == "secret"
    with pytest.raises(PlatformError, match="binding mismatch"):
        client.model_credential({**model, "revision": "b" * 40})


def test_credential_refusal_stays_lease_fenced_and_redacts_response_text():
    import httpx
    from reliquary.eval.platform_client import LeaseLost

    calls = []
    def handle(request):
        calls.append(request)
        assert request.headers["authorization"] == "Bearer pod-token"
        assert request.headers["x-task-lease"] == "current-lease"
        return httpx.Response(403, text="sensitive-upstream-text")
    client = PlatformClient("http://fixture", "pod-token", executor_id="pod", http=httpx.Client(
        base_url="http://fixture", transport=httpx.MockTransport(handle)))
    client.task_id, client._lease = "task", "current-lease"
    with pytest.raises(LeaseLost, match="credential unavailable") as failure:
        client.model_credential({"repo": "org/private", "revision": "a" * 40})
    assert "sensitive" not in str(failure.value) and len(calls) == 1


def test_leased_private_runner_never_falls_back_to_ambient_token(tmp_path, monkeypatch):
    from reliquary.eval.platform_client import LeaseLost
    from reliquary.eval.runner import EvalRunner
    from tests.unit.test_eval_platform_client import FakePlatform, _task
    from tests.unit.test_eval_runner import FakeGenerator, _prompts

    monkeypatch.setenv("HF_TOKEN", "ambient-private-token")
    platform = FakePlatform(tasks=[{**_task(), "model_access": "private"}], prompts=_prompts())
    client, generator = platform.client(), FakeGenerator()
    def refused(model):
        raise LeaseLost("private model credential unavailable")
    monkeypatch.setattr(client, "model_credential", refused)
    with pytest.raises(LeaseLost):
        EvalRunner(client, generator, work_dir=tmp_path).run()
    assert generator.loads == 0


class HungModel:
    def load(self, **kwargs):
        return {"model_sha": kwargs["revision"]}

    def generate(self, **kwargs):
        time.sleep(30)


def test_hung_generation_is_killed_and_joined_without_a_second_model():
    model = ModelProcess(HungModel, timeout_seconds=0.2)
    assert model.load(revision="a" * 40)["model_sha"] == "a" * 40
    pid = model._process.pid
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        model.generate([], samples=[], seeds=[], sampling={}, max_new_tokens=1)
    assert time.monotonic() - started < 3
    assert not model._process.is_alive() and model._process.pid == pid
    model.close()


def test_pinned_claim_uses_real_claim_shape_and_rejects_prompt_count_drift(tmp_path):
    from reliquary.eval.runner import EvalRunner
    from tests.unit.test_eval_platform_client import FakePlatform, _task
    from tests.unit.test_eval_runner import FakeGenerator

    body = request_protocol()
    body["sets"] = [{"set_id": "s", "env": "logic", "problems": 1, "samples": 1}]
    card = build_set("logic", count=1, seed=1, set_id="s", out=tmp_path / "set", open_environment=opener())
    body["protocol_bindings"] = {"s": card["protocol_sha256"]}
    task = {**_task(), **envelope(body), "envs": [{"set_id": "s", "env": "logic", "samples": 1,
                                                  "problem_ids_count": 1}],
            "set_protocols": {"s": {k: card[k] for k in ("protocol", "protocol_json", "protocol_sha256")}}}
    class Generator(FakeGenerator):
        def load(self, **kwargs):
            assert kwargs.pop("max_total_tokens") == 65536
            return super().load(**kwargs)
    prompt = {"problem_id": "p", "set_id": "s", "env": "logic", "messages": [{"role": "user", "content": "q"}]}
    platform = FakePlatform(tasks=[task], prompts=[prompt])
    result = EvalRunner(platform.client(), Generator(), work_dir=tmp_path).run()
    assert result["protocol_sha256"] == task["protocol_sha256"]
    platform = FakePlatform(tasks=[task], prompts=[prompt, {**prompt, "problem_id": "extra"}])
    with pytest.raises(PlatformError, match="counts differ"):
        EvalRunner(platform.client(), Generator(), work_dir=tmp_path / "bad").run()


def test_pinned_episode_deadlines_do_not_multiply_uploads(tmp_path):
    from reliquary.eval.runner import EvalRunner
    from tests.unit.test_eval_platform_client import FakePlatform, _task
    from tests.unit.test_eval_runner import FakeGenerator

    card = build_set("logic", count=2, seed=1, set_id="s", out=tmp_path / "set", open_environment=opener())
    body = request_protocol()
    body["sets"][0]["problems"] = 2
    body["sets"][0]["samples"] = 2
    body["protocol_bindings"] = {"s": card["protocol_sha256"]}
    task = {**_task(), **envelope(body), "envs": [{"set_id": "s", "env": "logic", "samples": 2,
            "problem_ids_count": 2}], "set_protocols": {"s": {k: card[k] for k in (
                "protocol", "protocol_json", "protocol_sha256")}}}
    class Generator(FakeGenerator):
        episodes = 0
        def load(self, **kwargs):
            kwargs.pop("max_total_tokens")
            return super().load(**kwargs)
        def begin_episode(self):
            self.episodes += 1
    generator = Generator()
    prompts = [json.loads(line) for line in (tmp_path / "set/prompts.jsonl").read_text().splitlines()]
    platform = FakePlatform(tasks=[task], prompts=prompts)
    result = EvalRunner(platform.client(), generator, work_dir=tmp_path / "run", chunk_problems=2).run()
    assert generator.loads == 1 and generator.episodes == 4
    assert list(map(len, generator.generated)) == [1, 1, 1, 1]
    assert result["rows"] == 4
    assert len(platform.uploads) == len(result["completion_keys"]) == 1


def _partial_frame_worker(connection, factory):
    import os
    import struct
    from reliquary.eval.model_process import _receive, _send

    os.setsid()
    _send(connection, {"worker_pid": os.getpid(), "worker_group": os.getpgrp()})
    _receive(connection)
    _send(connection, {"result": {"model_sha": "a" * 40}})
    _receive(connection)
    os.write(connection.fileno(), struct.pack("!i", 1024) + b"{")
    time.sleep(30)


def test_partial_child_frame_cannot_escape_the_deadline(monkeypatch):
    from reliquary.eval import model_process
    import threading

    monkeypatch.setattr(model_process, "_worker", _partial_frame_worker)
    model = ModelProcess(HungModel, timeout_seconds=0.2)
    model.load(revision="a" * 40)
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        model.generate([], samples=[], seeds=[], sampling={}, max_new_tokens=1)
    assert time.monotonic() - started < 3 and not model._process.is_alive()
    assert not any(t.name == "evaluation-model-io" for t in threading.enumerate())


class TimedModel(HungModel):
    def render(self, **kwargs):
        time.sleep(0.2)
        return [1]

    def generate(self, **kwargs):
        from reliquary.eval.runner import Completion

        time.sleep(0.2)
        return [[Completion("answer", 1, "stop")]]


def test_render_and_generation_share_one_episode_deadline():
    model = ModelProcess(TimedModel, timeout_seconds=0.3)
    model.load(revision="a" * 40)
    model.begin_episode()
    with pytest.raises(TimeoutError):
        model.render({"text": "q"}, thinking=False)
        model.generate([[1]], samples=[1], seeds=[0], sampling={}, max_new_tokens=1)
    assert not model._process.is_alive()


def test_lease_loss_stops_the_owned_model_process():
    import threading
    from reliquary.eval.platform_client import LeaseLost

    cancelled = threading.Event()
    model = ModelProcess(HungModel, timeout_seconds=30, cancelled=cancelled.is_set)
    model.load(revision="a" * 40)
    timer = threading.Timer(0.1, cancelled.set)
    timer.start()
    try:
        with pytest.raises(LeaseLost):
            model.generate([], samples=[], seeds=[], sampling={}, max_new_tokens=1)
        assert not model._process.is_alive()
    finally:
        timer.join()


class CrashingModel(HungModel):
    def load(self, **kwargs):
        import subprocess

        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                                 stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                 stderr=subprocess.DEVNULL, close_fds=True)
        return {"descendant": child.pid}

    def generate(self, **kwargs):
        import os

        os._exit(42)


def test_crashed_model_leader_does_not_orphan_its_descendants():
    import os
    import signal
    import subprocess

    model = ModelProcess(CrashingModel, timeout_seconds=2)
    descendant = model.load()["descendant"]
    group = model._process.pid
    try:
        assert os.getpgid(descendant) == group
        with pytest.raises(EOFError):
            model.generate([], samples=[], seeds=[], sampling={}, max_new_tokens=1)
        until = time.monotonic() + 2
        while True:
            state = subprocess.run(["ps", "-o", "stat=", "-p", str(descendant)],
                                   capture_output=True, text=True, check=False).stdout.strip()
            if not state or state.startswith("Z") or time.monotonic() >= until:
                break
            time.sleep(0.02)
        assert not state or state.startswith("Z")
        assert model._process.exitcode == 42
        model.close()
    finally:
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass


def test_executor_sigterm_joins_its_model_group_and_restores_handler(tmp_path):
    import os
    import signal
    import subprocess
    from pathlib import Path

    ready, finished = tmp_path / "ready.json", tmp_path / "finished.json"
    script = tmp_path / "stop_executor.py"
    script.write_text(f'''
import json, os, signal
from pathlib import Path
from reliquary.eval import runner
from reliquary.eval.model_process import ModelProcess
from reliquary.eval.platform_client import LeaseLost
from tests.unit.test_eval_platform_client import FakePlatform, _task
class Model:
    def load(self, **kwargs):
        return {{'model_sha': kwargs['revision'], 'vllm_version': 'fixture', 'gpu': 'CPU fixture'}}
    def render(self, row, **kwargs): return 'fixture'
    def generate(self, **kwargs):
        import time
        Path({str(ready)!r}).write_text(json.dumps({{'worker': os.getpid()}}))
        time.sleep(60)
if __name__ == '__main__':
    client = FakePlatform(tasks=[_task()], prompts=[{{'problem_id': 'p', 'samples': 1}}]).client()
    client.closed = False
    close = client.close
    def closing():
        client.closed = True
        close()
    client.close = closing
    runner.PlatformClient = lambda *args, **kwargs: client
    runner.VLLMGenerator = lambda: ModelProcess(Model, timeout_seconds=60)
    previous = signal.getsignal(signal.SIGTERM)
    try:
        runner.run_evaluation(platform='http://fixture', executor_id='fixture', work_dir={str(tmp_path / 'work')!r})
    except LeaseLost:
        Path({str(finished)!r}).write_text(json.dumps({{'closed': client.closed,
            'restored': signal.getsignal(signal.SIGTERM) == previous}}))
''')
    process, worker = None, None
    with (tmp_path / "executor.log").open("wb") as log:
        try:
            root = str(Path(__file__).parents[2])
            process = subprocess.Popen([sys.executable, "-B", str(script)], cwd=root,
                env={**os.environ, "PYTHONPATH": root}, stdout=log, stderr=subprocess.STDOUT)
            until = time.monotonic() + 60
            while not ready.exists() and process.poll() is None and time.monotonic() < until:
                time.sleep(0.02)
            assert ready.exists(), (tmp_path / "executor.log").read_text()
            worker = json.loads(ready.read_text())["worker"]
            assert os.getpgid(worker) == worker
            process.send_signal(signal.SIGTERM)
            assert process.wait(timeout=15) == 0
            assert json.loads(finished.read_text()) == {"closed": True, "restored": True}
            with pytest.raises(ProcessLookupError):
                os.kill(worker, 0)
        finally:
            if process is not None and process.poll() is None:
                process.kill()
                process.wait(timeout=5)
            if worker is not None:
                try:
                    os.killpg(worker, signal.SIGKILL)
                except ProcessLookupError:
                    pass


def test_shared_worker_protocol_bytes_fixture():
    from pathlib import Path
    path = Path(__file__).parents[1] / "fixtures/evaluation_protocol.json"
    fixture = json.loads(path.read_text())
    ep._envelope({"protocol": fixture["set_protocol"], "protocol_json": fixture["set_protocol_json"],
                  "protocol_sha256": fixture["set_protocol_sha256"]})
    body = fixture["protocol"]
    doc = {k: fixture[k] for k in ("protocol", "protocol_json", "protocol_sha256")}
    doc.update({k: body[k] for k in ("protocol_bindings", "sampling", "thinking", "max_new_tokens")})
    assert ep.validate_request(doc) == fixture["protocol_sha256"]
