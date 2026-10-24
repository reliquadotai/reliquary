"""The actual job and split-process owners finish descendant cleanup."""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
import tempfile
import textwrap

import pytest


def test_cancelled_auditor_write_waits_for_every_peer_finalizer():
    from reliquary.validator.corpus_auditor import CorpusAuditor

    async def exercise():
        started = [asyncio.Event(), asyncio.Event()]
        cleaned = []

        class Records:
            async def write_verdict(self, job_id, sid, verdict):
                index = int(sid)
                started[index].set()
                try:
                    await asyncio.Future()
                finally:
                    await asyncio.sleep(0.01 * (index + 1))
                    cleaned.append(index)

        auditor = CorpusAuditor(job_id="fixture", records=Records(), model=None,
                                tokenizer=None, proof=None)
        owner = asyncio.create_task(auditor._write_all([("0", {}), ("1", {})]))
        await asyncio.gather(*(event.wait() for event in started))
        owner.cancel()
        with pytest.raises(asyncio.CancelledError):
            await owner
        assert sorted(cleaned) == [0, 1]
        assert all(task is asyncio.current_task() for task in asyncio.all_tasks())

    asyncio.run(exercise())


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX signals and Unix sockets")
@pytest.mark.parametrize("stop_signal", [signal.SIGTERM, signal.SIGINT])
def test_actual_split_judge_signal_awaits_services_feed_backfill_and_threads(stop_signal):
    script = textwrap.dedent("""
        import asyncio
        import os
        from pathlib import Path
        import signal
        import sys
        import threading
        import time
        from types import SimpleNamespace
        from reliquary.infrastructure import corpus_job_store, corpus_record_store
        from reliquary.shared import modeling
        from reliquary.validator import corpus_gpu, corpus_judge_threads, corpus_split, corpus_validator
        from reliquary.validator.corpus_judge_process import run_corpus_judges
        from reliquary.validator.corpus_miner_status import MinerBook

        run_dir, stop_signal = sys.argv[1], int(sys.argv[2])
        events = [asyncio.Event() for _ in range(5)]
        codec_release = threading.Event()

        async def held(index):
            events[index].set()
            try:
                if index == 0:
                    await asyncio.gather(*(event.wait() for event in events))
                    while not (Path(run_dir) / "judge.sock").exists():
                        await asyncio.sleep(0.01)
                    os.kill(os.getpid(), stop_signal)
                await asyncio.Future()
            finally:
                await asyncio.sleep(0.01 * (index + 1))
                print(f"cleaned-{index}", flush=True)
                if index == 4:
                    codec_release.set()

        class Records:
            async def read_settlement(self, job_id):
                return {}, None
            async def list_verdict_ids(self, job_id):
                return ["3", "4"]
            async def read_verdict(self, job_id, sid):
                await held(int(sid))

        records = Records()
        job = SimpleNamespace(job_id="fixture", episode=None)
        async def read_job(self, job_id):
            return job, None
        async def read_info(run_dir):
            return {"vocab_size": 100}
        async def settle(*args):
            await held(1)
        def codec_work():
            assert codec_release.wait(20)
            time.sleep(0.04)
            print("codec-finished", flush=True)
        def judge_records(threads, **kwargs):
            threads.codec.submit(codec_work)
            return records
        def wire(wiring, **kwargs):
            feed = kwargs["arrivals_covered"].__self__
            auditor = SimpleNamespace(run=lambda: held(0), rescan_store=lambda: held(2))
            wiring.auditor = auditor
            wiring.settler = SimpleNamespace()
            wiring.miners = MinerBook(job_id="fixture", task_id="fixture", records=records,
                                      thresholds={})
            wiring.miners.start_backfill()
            feed.auditors = {"fixture": auditor}
            feed.receive({"epoch": "fixture", "ids": {}, "as_of": None})

        corpus_job_store.BucketJobStore.read_job = read_job
        corpus_record_store.BucketRecordStore = lambda: records
        modeling.load_tokenizer = lambda directory: None
        corpus_gpu.read_info = read_info
        corpus_judge_threads.judge_record_store = judge_records
        corpus_validator.wire_job_judge = wire
        corpus_validator.settle_forever = settle
        async def judge(spec, index):
            await run_corpus_judges(served=[(SimpleNamespace(task_id="fixture", job_id="fixture"), 0.0)],
                directory=run_dir, run_dir=run_dir, proof=SimpleNamespace(chunk_tokens=32, topk=8),
                socket_path=str(Path(run_dir) / "judge.sock"))
        corpus_split._judge = judge
        spec = SimpleNamespace(run_dir=run_dir, child_init=None)
        corpus_split.child_main("judge", 0, spec)
        print("child-finished", flush=True)
    """)
    with tempfile.TemporaryDirectory(prefix="cs-", dir="/tmp") as run_dir:
        result = subprocess.run([sys.executable, "-u", "-c", script, run_dir, str(int(stop_signal))],
                                capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines()[-1] == "child-finished"
    assert sorted(result.stdout.splitlines()) == [
        "child-finished", "cleaned-0", "cleaned-1", "cleaned-2", "cleaned-3", "cleaned-4",
        "codec-finished"]


def test_supervisor_uses_configured_shared_stop_deadline(monkeypatch):
    from types import SimpleNamespace
    from reliquary.validator.corpus_split import STOP_GRACE_ENV, Supervisor

    monkeypatch.setenv(STOP_GRACE_ENV, "0.25")
    supervisor = Supervisor(SimpleNamespace(groups=[]))
    calls = []

    class Child:
        name = "fixture"
        alive = True

        def is_alive(self):
            return self.alive

        def terminate(self):
            calls.append("terminate")

        def join(self, timeout):
            calls.append(timeout)
            self.alive = False

    supervisor.children["front"].process = Child()
    supervisor.stop()
    assert calls[0] == "terminate" and 0 < calls[1] <= 0.25
    assert supervisor._stopping


@pytest.mark.skipif(sys.platform == "win32", reason="requires POSIX signals")
def test_actual_split_front_signal_drains_native_http_before_child_cleanup(tmp_path):
    script = textwrap.dedent("""
        import asyncio
        import os
        import signal
        from types import SimpleNamespace
        import uvicorn
        from reliquary.validator import corpus_split
        from reliquary.validator.corpus_hot_jobs import CorpusJobSet
        from reliquary.validator.corpus_service import CorpusJobRoutes
        from reliquary.validator.corpus_validator import _run_corpus_services

        async def front(spec):
            ledger_started, ledger_release = asyncio.Event(), asyncio.Event()
            record_started, record_release = asyncio.Event(), asyncio.Event()
            cleaned = []
            async def app(scope, receive, send):
                await receive()
                ledger_started.set()
                await ledger_release.wait()
                print("ledger-durable", flush=True)
                record_started.set()
                await record_release.wait()
                print("record-durable", flush=True)
                await send({"type": "http.response.start", "status": 200, "headers": []})
                await send({"type": "http.response.body", "body": b"ok"})
            server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0,
                lifespan="off", ws="none", log_level="critical"))
            async def worker(index):
                try:
                    await asyncio.Future()
                finally:
                    await asyncio.sleep(0.02 * (index + 1))
                    cleaned.append(index)
                    print(f"cleaned-{index}", flush=True)
            async def probe():
                while not server.started:
                    await asyncio.sleep(0.01)
                port = server.servers[0].sockets[0].getsockname()[1]
                reader, writer = await asyncio.open_connection("127.0.0.1", port)
                writer.write(b"POST / HTTP/1.1\\r\\nHost: localhost\\r\\nContent-Length: 0\\r\\n\\r\\n")
                await writer.drain()
                await ledger_started.wait()
                # An actual external disconnect must not release the native
                # ASGI task while its accepted write is still pending.
                writer.close()
                await writer.wait_closed()
                os.kill(os.getpid(), signal.SIGTERM)
                await asyncio.sleep(0.15)
                assert server.should_exit and not cleaned and server.server_state.tasks
                ledger_release.set()
                await record_started.wait()
                await asyncio.sleep(0.05)
                assert not cleaned and server.server_state.tasks
                record_release.set()
                print("barrier-qualified", flush=True)
                await asyncio.Future()
            jobs = CorpusJobSet(routes=CorpusJobRoutes(), router_for=lambda w: None, wire=None,
                jobs_of=lambda w: [worker(0), worker(1)])
            jobs.adopt(SimpleNamespace(entry=SimpleNamespace(job_id="fixture")))
            await _run_corpus_services(server, [jobs.run(), probe()])
            assert sorted(cleaned) == [0, 1]
        corpus_split._front = front
        corpus_split.child_main("front", 0, SimpleNamespace(run_dir=".", child_init=None))
        print("child-finished", flush=True)
    """)
    result = subprocess.run([sys.executable, "-u", "-c", script],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == ["ledger-durable", "barrier-qualified", "record-durable",
                                          "cleaned-0", "cleaned-1", "child-finished"]
