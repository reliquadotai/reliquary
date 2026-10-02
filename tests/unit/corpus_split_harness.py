"""Real processes for the split validator's tests: a bucket on disk that every
process shares (with R2-like latency and real conditional puts), the stand-ins
every child installs first (``install``), realistic math records, a load
generator for the miner routes, and runners for the split and the single
process. Not a test module itself.

Realistic records: one completion of 16-32k tokens (the user-stated range for
math; the code contract caps code at 8k), token ids of six digits, one 344
character proof per 32-token chunk, the decoded text: 0.5-1 MB of JSON each.
A backlog of tens of thousands is stored as stubs and expanded on read, so the
reading process pays the full decode while the disk holds kilobytes.
"""

from __future__ import annotations

import asyncio
import contextlib
import fcntl
import hashlib
import json
import math
import multiprocessing
import os
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

ROOT_ENV = "CORPUS_HARNESS_ROOT"
LATENCY_ENV = "CORPUS_HARNESS_LATENCY"      # read_lo,read_hi,put_lo,put_hi,list_page
GPU_TPS_ENV = "CORPUS_HARNESS_GPU_TPS"
BEACON_ENV = "CORPUS_HARNESS_BEACON"         # "const" for one randomness for every round
SEED_SLICE_ENV = "CORPUS_HARNESS_SEED_SLICE"
LOG_ENV = "CORPUS_HARNESS_LOG"               # every child appends its output here

CHECKPOINT = "a" * 64
EOS = 151645
PROMPT_SOURCE = "reliquary_stateful_tools_v1"
SYNTH = b"SYNTH:"
PROOF_TEXT = "A" * 344
DEFAULT_LATENCY = "0.05,0.10,0.10,0.20,0.10"
_B58 = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
HOTKEYS = [f"5Hkm{_B58[i]}" for i in range(24)]


def _root() -> Path:
    return Path(os.environ[ROOT_ENV])


def _latency() -> list[float]:
    return [float(x) for x in os.environ.get(LATENCY_ENV, DEFAULT_LATENCY).split(",")]


def _client_error(code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code}}, "S3")


# -- realistic records ------------------------------------------------------

_TEMPLATES: dict = {}


def _template(n: int, forged: bool) -> bytes:
    """A record of ``n`` completion tokens with placeholders, built once per size."""
    key = (n, forged)
    if key not in _TEMPLATES:
        rng = random.Random(n)
        tokens = [rng.randrange(100_000, 200_000) for _ in range(n - 1)]
        if forged:
            from tests.unit.corpus_split_fakes import FORGED

            tokens[len(tokens) // 2] = FORGED
        tokens.append(EOS)
        record = {"schema": "reliquary/corpus-record/v1", "submission_id": "@@SID@@",
                  "job_id": "@@JOB@@", "hotkey": "@@HK@@", "cursor": 0, "prompt_index": 0,
                  "rendered_prompt": "<prompt row-0>", "received_at": "@@TS@@",
                  "token_count": n,
                  "completions": [{"tokens": tokens, "text": "".join(map(str, tokens[:-1])),
                                   "proofs": [PROOF_TEXT] * math.ceil(n / 32)}]}
        _TEMPLATES[key] = json.dumps(record).encode().replace(b'"@@TS@@"', b"@@TS@@")
    return _TEMPLATES[key]


def synth_stub(*, sid: str, job_id: str, hotkey: str, received_at: float, n: int,
               forged: bool = False) -> bytes:
    return SYNTH + json.dumps({"sid": sid, "job": job_id, "hk": hotkey, "ts": received_at,
                               "n": n, "forged": forged}).encode()


def expand(stub: bytes) -> bytes:
    meta = json.loads(stub[len(SYNTH):])
    return (_template(int(meta["n"]), bool(meta["forged"]))
            .replace(b"@@SID@@", meta["sid"].encode()).replace(b"@@JOB@@", meta["job"].encode())
            .replace(b"@@HK@@", meta["hk"].encode())
            .replace(b"@@TS@@", repr(float(meta["ts"])).encode()))


def size_bucket(rng: random.Random, low: int = 16_384, high: int = 32_768) -> int:
    """Completion lengths in eight steps over [low, high]: templates are shared."""
    return low + (high - low) * rng.randrange(8) // 7


# -- the bucket -------------------------------------------------------------


class _Body:
    def __init__(self, data: bytes) -> None:
        self._data = data

    async def read(self) -> bytes:
        return self._data

    def close(self) -> None:
        pass


class FileS3:
    """An aiobotocore client's calls over a directory: each object a file,
    its ETag its inode and mtime; conditional puts under a per-key flock, so
    two processes racing on one key behave as R2 does."""

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or _root())
        self._objects = self.root / "objects"
        self._locks = self.root / "locks"
        self._lat = _latency()
        self._rng = random.Random()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def _delay(self, kind: str) -> None:
        read_lo, read_hi, put_lo, put_hi, page = self._lat
        seconds = {"read": self._rng.uniform(read_lo, read_hi),
                   "put": self._rng.uniform(put_lo, put_hi), "list": page}[kind]
        if seconds > 0:
            await asyncio.sleep(seconds)

    def _path(self, key: str) -> Path:
        if key.startswith("/") or ".." in key.split("/"):
            raise ValueError(key)
        return self._objects / key

    @staticmethod
    def _etag(stat) -> str:
        return f'"{stat.st_ino}-{stat.st_mtime_ns}-{stat.st_size}"'

    async def get_object(self, Bucket, Key, **kw):
        await self._delay("read")
        path = self._path(Key)
        try:
            with open(path, "rb") as handle:
                data = handle.read()
                etag = self._etag(os.fstat(handle.fileno()))
        except FileNotFoundError:
            raise _client_error("NoSuchKey") from None
        if data.startswith(SYNTH):
            data = expand(data)
        return {"Body": _Body(data), "ETag": etag}

    @contextlib.contextmanager
    def _locked(self, key: str):
        self._locks.mkdir(parents=True, exist_ok=True)
        fd = os.open(self._locks / hashlib.sha1(key.encode()).hexdigest(),
                     os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)

    async def put_object(self, Bucket, Key, Body, IfNoneMatch=None, IfMatch=None, **kw):
        await self._delay("put")
        data = Body if isinstance(Body, (bytes, bytearray)) else Body.encode()
        path = self._path(Key)
        with self._locked(Key):
            try:
                current = self._etag(os.stat(path))
            except FileNotFoundError:
                current = None
            if IfMatch is not None and IfMatch != current:
                raise _client_error("PreconditionFailed")
            if IfNoneMatch is not None and current is not None:
                raise _client_error("PreconditionFailed")
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
            tmp.write_bytes(bytes(data))
            os.replace(tmp, path)
            etag = self._etag(os.stat(path))
        return {"ETag": etag}

    async def delete_object(self, Bucket, Key, **kw):
        await self._delay("put")
        self._path(Key).unlink(missing_ok=True)
        return {}

    def keys(self, prefix: str) -> list[str]:
        base = self._objects / (prefix if prefix.endswith("/") else os.path.dirname(prefix))
        if not base.exists():
            return []
        out = []
        for dirpath, _, files in os.walk(base):
            rel = os.path.relpath(dirpath, self._objects)
            for name in files:
                if name.startswith("."):
                    continue
                key = name if rel == "." else f"{rel}/{name}"
                if key.startswith(prefix):
                    out.append(key)
        return sorted(out)

    def get_paginator(self, name):
        client = self

        class _Paginator:
            def paginate(self, Bucket, Prefix="", Delimiter=None, **kw):
                async def pages():
                    keys = client.keys(Prefix)
                    if Delimiter:
                        prefixes = sorted({Prefix + k[len(Prefix):].split(Delimiter)[0] + Delimiter
                                           for k in keys if Delimiter in k[len(Prefix):]})
                        await client._delay("list")
                        yield {"CommonPrefixes": [{"Prefix": p} for p in prefixes],
                               "Contents": [{"Key": k} for k in keys
                                            if Delimiter not in k[len(Prefix):]]}
                        return
                    for i in range(0, max(len(keys), 1), 1000):
                        await client._delay("list")
                        yield {"Contents": [{"Key": k} for k in keys[i:i + 1000]]}

                return pages()

        return _Paginator()


def put_raw(root: Path, key: str, data: bytes) -> None:
    path = Path(root) / "objects" / key
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def get_raw(root: Path, key: str) -> bytes | None:
    try:
        return (Path(root) / "objects" / key).read_bytes()
    except FileNotFoundError:
        return None


class FileArchives:
    """``R2Archives`` on disk; every write is logged, so a test can see a
    window paid twice with different rewards."""

    def __init__(self, *, served=None, **kw) -> None:
        self._dir = _root() / "archives"

    async def other_max(self, task_id: str):
        return None

    async def write(self, task_id: str, window: int, data: dict) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        (self._dir / f"{task_id}-{int(window)}.json").write_text(json.dumps(data, sort_keys=True))
        with open(self._dir / "writes.jsonl", "a") as log:
            log.write(json.dumps({"task": task_id, "window": int(window), "pid": os.getpid(),
                                  "rewards": data["rewards_by_hotkey"]}, sort_keys=True) + "\n")


# -- the stand-ins every child installs ------------------------------------


def _beacon(round_number: int) -> str:
    if os.environ.get(BEACON_ENV) == "const":
        return "c3" * 32
    return hashlib.sha256(f"drand-{round_number}".encode()).hexdigest()


def round_at(t: float) -> int:
    return int(t // 3) + 1


def gpu_score_sequences(model, sequences, *, chunk_tokens, topk, batch_tokens):
    """The forward at the measured rate (it blocks its thread, as a GPU call
    does, without holding the GIL), honest unless a completion is forged."""
    from tests.unit.corpus_split_fakes import score_rows

    tokens = sum(len(t) - n for t, n, _ in sequences)
    seconds = tokens / float(os.environ.get(GPU_TPS_ENV, "4860"))
    time.sleep(seconds)
    return score_rows(sequences), seconds, 0.0


def install(single: bool = False) -> None:
    if os.environ.get(LOG_ENV):
        fd = os.open(os.environ[LOG_ENV], os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    import huggingface_hub

    import reliquary.corpus.encoding as encoding
    import reliquary.infrastructure.corpus_job_store as job_store
    import reliquary.infrastructure.corpus_record_store as record_store
    import reliquary.infrastructure.storage as storage
    import reliquary.infrastructure.task_registry_store as registry
    import reliquary.protocol.profiles as profiles
    import reliquary.protocol.signatures as signatures
    import reliquary.shared.modeling as modeling
    from reliquary.validator import (
        corpus_audit, corpus_auditor, corpus_gpu, corpus_service, corpus_settlement,
        corpus_validator,
    )
    from tests.unit import corpus_split_fakes as fakes
    from tests.unit.test_corpus_service import _Environment, _Renderer, _Spec

    client = lambda **kw: FileS3()  # noqa: E731
    for module in (job_store, record_store, storage):
        module.get_s3_client = client
    corpus_settlement.R2Archives = FileArchives
    corpus_validator.drand_beacon = _beacon
    corpus_validator.LazyRoundAt = lambda: round_at
    modeling.load_tokenizer = lambda path: fakes.Tokenizer()
    real_prompt_job = corpus_service.prompt_job_for_spec
    corpus_service.renderer_for_job = lambda job, encode, **kw: _Renderer()
    corpus_service.prompt_job_for_spec = lambda job, **kw: real_prompt_job(
        job, environments={PROMPT_SOURCE: _Spec(_Environment(rows=1_000_000))})
    signatures.verify_corpus_signature = lambda request: True
    signatures.verify_corpus_skip_signature = lambda request: True

    async def no_registry():
        return {}, None

    registry.read_registry = no_registry
    corpus_gpu._load_model = lambda directory: fakes.Model()
    corpus_audit.score_sequences = gpu_score_sequences
    corpus_auditor.score_sequences = gpu_score_sequences
    if os.environ.get(SEED_SLICE_ENV):
        corpus_auditor.SEED_SLICE_IDS = int(os.environ[SEED_SLICE_ENV])
    if single:
        huggingface_hub.snapshot_download = lambda repo, revision=None, **kw: str(_root())
        encoding.checkpoint_fingerprint = lambda directory: CHECKPOINT
        modeling.load_text_only_model = lambda path, **kw: fakes.Model()
        corpus_validator.startup_refusal = lambda *a, **kw: None
        profiles.toploc_proof = lambda profile: fakes.PROOF


# -- the bucket's initial content -------------------------------------------


def manifest(job_id: str, *, max_new_tokens: int = 32_768) -> dict:
    return {
        "schema": "reliquary/corpus-job/v1", "job_id": job_id,
        "checkpoint_repo": "org/Frozen", "checkpoint_revision": "abc123",
        "checkpoint_sha256": CHECKPOINT, "prompt_source": PROMPT_SOURCE,
        "prompt_count": 1_000_000, "renderer_id": "reliquary-jsonl-tools-v1",
        "eos_token_id": EOS,
        "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 16,
                     "max_new_tokens": max_new_tokens, "n": 1},
        "slots_per_prompt": 8, "filter": None, "prompt_order": "free",
        "deadline_round": 50_000_000,
    }


def entry(task_id: str, job_id: str, cap: float = 0.1, **audit) -> SimpleNamespace:
    params = {"cap": cap, "audit_q": 0.15, "audit_probation_submissions": 5,
              "audit_hold_seconds": 3600.0, "audit_ban_after_failures": 1000, **audit}
    return SimpleNamespace(task_id=task_id, job_id=job_id, mechanism="corpus-generation",
                           params=params, contract=None, status="active", retired_at=None)


def seed_bucket(root: Path, jobs, *, hotkeys=HOTKEYS, audited_passed: int = 1000) -> None:
    from reliquary.corpus.audit_policy import MinerState
    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.infrastructure import corpus_record_store as record_store

    for job_id in jobs:
        put_raw(root, job_store._job_key(job_id), json.dumps(manifest(job_id)).encode())
        miners = {hk: MinerState(audited_passed=audited_passed).to_dict() for hk in hotkeys}
        put_raw(root, record_store._miners_key(job_id), json.dumps(miners).encode())


def seed_backlog(root: Path, job_id: str, n: int, *, now: float, oldest: float,
                 newest: float, seed: int = 0, hotkeys=HOTKEYS) -> list[str]:
    """``n`` stub records received between ``now - oldest`` and ``now - newest``."""
    from reliquary.infrastructure import corpus_record_store as record_store

    rng = random.Random(seed)
    ids = []
    for i in range(n):
        sid = "%064x" % rng.getrandbits(256)
        received = now - rng.uniform(newest, oldest)
        put_raw(root, record_store._key(job_id, "submissions", sid),
                synth_stub(sid=sid, job_id=job_id, hotkey=hotkeys[i % len(hotkeys)],
                           received_at=received, n=size_bucket(rng)))
        ids.append(sid)
    return ids


def listed(root: Path, job_id: str, kind: str) -> dict[str, dict]:
    from reliquary.infrastructure import corpus_record_store as record_store

    base = Path(root) / "objects" / record_store._prefix(job_id, kind)
    if not base.exists():
        return {}
    out = {}
    for path in base.iterdir():
        if path.name.startswith("."):
            continue
        data = path.read_bytes()
        out[path.name.removesuffix(".json")] = (
            json.loads(data) if not data.startswith(SYNTH) else json.loads(data[len(SYNTH):]))
    return out


def settlement(root: Path, job_id: str) -> dict:
    from reliquary.infrastructure import corpus_record_store as record_store

    data = get_raw(root, record_store._settlement_key(job_id))
    return json.loads(data) if data else {}


# -- miners -----------------------------------------------------------------


def submission_body(job_id: str, hotkey: str, prompt_index: int, n: int, *,
                    rng: random.Random, forged: bool = False) -> bytes:
    tokens = [rng.randrange(100_000, 200_000) for _ in range(n - 1)]
    if forged:
        from tests.unit.corpus_split_fakes import FORGED

        tokens[len(tokens) // 2] = FORGED
    tokens.append(EOS)
    return json.dumps({
        "job_id": job_id, "miner_hotkey": hotkey, "cursor": 0, "prompt_index": prompt_index,
        "checkpoint_sha256": CHECKPOINT, "rendered_prompt": f"<prompt row-{prompt_index}>",
        "completions": [{"tokens": tokens, "text": "".join(map(str, tokens[:-1])),
                         "proofs": [PROOF_TEXT] * math.ceil(n / 32)}],
        "signature": "ok",
    }).encode()


async def submit_load(base_url: str, job_id: str, *, seconds: float, rate: float,
                      size=(16_384, 32_768), seed: int = 0, hotkeys=HOTKEYS,
                      first_prompt: int = 0, forged_hotkeys=()) -> list[dict]:
    """Submissions at ``rate`` a second for ``seconds`` (each its own request,
    as miners send them); per request its latency, HTTP status and answer."""
    import httpx

    rng = random.Random(seed)
    results: list[dict] = []
    tasks = []
    prompt = first_prompt

    async def one(http, hotkey, body):
        start = time.monotonic()
        try:
            response = await http.post(f"/corpus/jobs/{job_id}/submit", content=body,
                                       headers={"content-type": "application/json"})
            status = response.status_code
            answer = response.json() if status == 200 else None
        except Exception as exc:  # noqa: BLE001
            status, answer = repr(exc), None
        results.append({"at": start, "seconds": time.monotonic() - start, "status": status,
                        "accepted": bool(answer and answer.get("accepted")),
                        "reason": answer.get("reason") if answer else None, "hotkey": hotkey})

    async with httpx.AsyncClient(base_url=base_url, timeout=300.0) as http:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            hotkey = hotkeys[rng.randrange(len(hotkeys))]
            n = size[0] + rng.randrange(size[1] - size[0] + 1)
            body = submission_body(job_id, hotkey, prompt, n, rng=rng,
                                   forged=hotkey in forged_hotkeys)
            prompt += 1
            tasks.append(asyncio.ensure_future(one(http, hotkey, body)))
            await asyncio.sleep(rng.expovariate(rate))
        await asyncio.gather(*tasks)
    return results


def quantiles(results: list[dict]) -> dict:
    seconds = sorted(r["seconds"] for r in results if r["status"] == 200)
    if not seconds:
        return {"n": 0}

    def q(p):
        return seconds[min(len(seconds) - 1, int(p * len(seconds)))]

    return {"n": len(seconds), "p50": round(statistics.median(seconds), 3),
            "p90": round(q(0.90), 3), "p99": round(q(0.99), 3), "max": round(seconds[-1], 3),
            "errors": sum(1 for r in results if r["status"] != 200),
            "accepted": sum(1 for r in results if r["accepted"])}


async def wait_http(base_url: str, *, timeout: float = 120.0) -> None:
    import httpx

    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(base_url=base_url, timeout=5.0) as http:
        while time.monotonic() < deadline:
            try:
                if (await http.get("/corpus/jobs")).status_code == 200:
                    return
            except Exception:  # noqa: BLE001
                pass
            await asyncio.sleep(0.5)
    raise TimeoutError(f"{base_url} never answered")


def free_port() -> int:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# -- runners ------------------------------------------------------------------


def harness_env(root: Path, **extra) -> dict:
    env = {ROOT_ENV: str(root), LATENCY_ENV: DEFAULT_LATENCY, SEED_SLICE_ENV: "256"}
    env.update({k: str(v) for k, v in extra.items()})
    return env


def short_run_dir() -> str:
    """Unix socket paths are limited to 108 bytes: sockets go under /tmp."""
    import tempfile

    return tempfile.mkdtemp(prefix="rcs-", dir="/tmp")


def split_spec(root: Path, served, groups, *, port: int, settle_every_seconds: float = 60.0,
               auditor_kwargs=None, run_dir: str | None = None):
    from reliquary.validator.corpus_split import SplitSpec
    from tests.unit.corpus_split_fakes import PROOF

    return SplitSpec(served=list(served), directory=str(root), fingerprint=CHECKPOINT,
                     proof=PROOF, run_dir=run_dir or short_run_dir(), groups=groups,
                     http_host="127.0.0.1", http_port=port, registration_gate=False,
                     settle_every_seconds=settle_every_seconds,
                     child_init="tests.unit.corpus_split_harness:install",
                     auditor_kwargs=dict(auditor_kwargs or {}))


def single_main(served, port: int, settle_every_seconds: float, auditor_kwargs) -> None:
    """The single process, as ``reliquary validate`` runs it today."""
    import logging

    logging.basicConfig(level=logging.INFO, force=True,
                        format="%(asctime)s | single | %(name)s | %(levelname)s | %(message)s")
    install(single=True)
    from reliquary.validator.corpus_validator import run_corpus_validator

    asyncio.run(run_corpus_validator(
        jobs=list(served), wallet=None, netuid=81, signer_client=None, http_host="127.0.0.1",
        http_port=port, set_weights=False, registration_gate=False,
        settle_every_seconds=settle_every_seconds, auditor_kwargs=auditor_kwargs))


def start_single(served, *, port: int, settle_every_seconds: float = 60.0,
                 auditor_kwargs=None):
    process = multiprocessing.get_context("spawn").Process(
        target=single_main, args=(list(served), port, settle_every_seconds,
                                  dict(auditor_kwargs or {})), daemon=True)
    process.start()
    return process


@contextlib.contextmanager
def environment(values: dict):
    saved = {k: os.environ.get(k) for k in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


class SupervisorThread:
    """The supervisor's watch loop on a thread of the test process."""

    def __init__(self, spec) -> None:
        import threading

        from reliquary.validator.corpus_split import Supervisor

        self.supervisor = Supervisor(spec, backoff_seconds=0.5, max_backoff_seconds=2.0)
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)

    def _watch(self) -> None:
        while not self._stop.wait(0.3):
            self.supervisor.check()

    def __enter__(self):
        self.supervisor.start()
        self._thread.start()
        return self.supervisor

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(5)
        self.supervisor.stop(grace_seconds=10)
        return False
