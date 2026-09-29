"""GPU-free rehearsal of the corpus ledger route against a test bucket.

Runs the real ``build_corpus_app`` over the real ``BucketJobStore`` and
``BucketRecordStore`` with a stub tokenizer, renderer and prompt source, so
whichever ``reliquary`` is on ``PYTHONPATH`` (a v2 worktree or a pre-v2
archive) can be driven with the same load and compared. Refuses to run unless
``R2_ENDPOINT_URL`` points at a loopback host.

    seed      put a job manifest and a ledger object at the store's keys
    serve     run one route process (optionally killing itself with SIGKILL
              after a segment seal or after a ledger write)
    supervise run ``serve`` phases back to back, restarting after each death
    load      drive hotkeys at a rate, with self-resends and cross-miner copies
    check     compare the stored ledger against the load logs
    timings   summarise the route's "corpus submission timing" lines
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import random
import signal
import statistics
import subprocess
import sys
import time
import types
import urllib.parse

JOB_PREFIX = "reliquary/corpus/jobs/"


def _require_loopback() -> None:
    host = urllib.parse.urlparse(os.environ.get("R2_ENDPOINT_URL", "")).hostname
    if host not in {"127.0.0.1", "localhost", "::1"}:
        sys.exit(f"refusing to run: R2_ENDPOINT_URL host is {host!r}, not loopback")


def _s3():
    from reliquary.infrastructure.storage import get_s3_client

    return get_s3_client()


# --------------------------------------------------------------------------
# seed
# --------------------------------------------------------------------------


async def _seed(args) -> None:
    from botocore.exceptions import ClientError

    bucket = os.environ["R2_BUCKET_ID"]
    manifest = json.load(open(args.manifest))
    job_id = manifest["job_id"]
    async with _s3() as client:
        try:
            await client.head_bucket(Bucket=bucket)
        except ClientError:
            await client.create_bucket(Bucket=bucket)
        paginator = client.get_paginator("list_objects_v2")
        async for page in paginator.paginate(Bucket=bucket, Prefix=JOB_PREFIX):
            for obj in page.get("Contents", []) or []:
                await client.delete_object(Bucket=bucket, Key=obj["Key"])
        await client.put_object(Bucket=bucket, Key=f"{JOB_PREFIX}{job_id}.json",
                                Body=open(args.manifest, "rb").read())
        await client.put_object(Bucket=bucket, Key=f"{JOB_PREFIX}{job_id}/ledgers.json",
                                Body=open(args.ledger, "rb").read())
        # The design leans on conditional writes: prove this server honours them.
        probe = "rehearsal/probe.json"
        await client.delete_object(Bucket=bucket, Key=probe)
        await client.put_object(Bucket=bucket, Key=probe, Body=b"1", IfNoneMatch="*")
        try:
            await client.put_object(Bucket=bucket, Key=probe, Body=b"2", IfNoneMatch="*")
            sys.exit("store ignores IfNoneMatch: rehearsal would prove nothing")
        except ClientError as exc:
            print("create-only probe refused:", exc.response["Error"]["Code"])
        try:
            await client.put_object(Bucket=bucket, Key=probe, Body=b"3", IfMatch='"deadbeef"')
            sys.exit("store ignores IfMatch: rehearsal would prove nothing")
        except ClientError as exc:
            print("stale IfMatch refused:", exc.response["Error"]["Code"])
        await client.delete_object(Bucket=bucket, Key=probe)
    print(f"seeded {bucket}: {job_id}")


# --------------------------------------------------------------------------
# serve
# --------------------------------------------------------------------------


class _Tokenizer:
    def decode(self, ids, **kwargs):
        return "".join(str(i) for i in ids)


class _Renderer:
    def initial_text(self, task):
        return f"<prompt {task.id}>"


class _Prompts:
    def task_for(self, index):
        from reliquary.environment.agentic.types import EpisodeTask

        return EpisodeTask(id=f"row-{index}", prompt=f"question {index}", tools=())


class _CrashingStore:
    """Delegates to the real store; SIGKILLs this process right after the
    ``after``-th successful call of the armed kind."""

    def __init__(self, inner, mode: str | None, after: int) -> None:
        self._inner, self._mode, self._left = inner, mode, after
        self.armed = False

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def _maybe_die(self, kind: str) -> None:
        if self.armed and self._mode == kind:
            self._left -= 1
            if self._left <= 0:
                logging.getLogger("rehearsal").critical("CRASH INJECTED %s", kind)
                for handler in logging.getLogger().handlers:
                    handler.flush()
                os.kill(os.getpid(), signal.SIGKILL)

    async def write_seen_segment(self, job_id, digests):
        result = await self._inner.write_seen_segment(job_id, digests)
        self._maybe_die("after_seal")
        return result

    async def write_ledgers(self, job_id, snapshot, etag):
        result = await self._inner.write_ledgers(job_id, snapshot, etag)
        self._maybe_die("after_ledger_write")
        return result


def _serve(args) -> None:
    import uvicorn

    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import BucketRecordStore
    from reliquary.validator import corpus_service
    from reliquary.validator.corpus_validator import build_corpus_app

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(process)d %(name)s %(levelname)s %(message)s")
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    store = _CrashingStore(BucketJobStore(), args.crash, args.crash_after)

    async def build():
        job, _ = await store.read_job(args.job)
        if job is None:
            sys.exit(f"no job {args.job}")
        extra = {}
        migrate = getattr(corpus_service, "migrate_ledgers_at_startup", None)
        if migrate is not None and not args.no_startup_migration:
            started = time.perf_counter()
            extra["seen_index"] = await migrate(store, job)
            logging.getLogger("rehearsal").info(
                "startup preparation took %.3f s", time.perf_counter() - started)
        return job, extra

    job, extra = asyncio.run(build())
    import reliquary.protocol.signatures  # noqa: F401 - pulls bittensor in now

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(process)d %(name)s %(levelname)s %(message)s"))

    def keep_logging() -> None:
        # Importing bittensor reconfigures logging; the timing lines must survive it.
        for name in ("reliquary", "rehearsal", "reliquary.validator.corpus_service"):
            named = logging.getLogger(name)
            named.handlers[:] = [handler] if name != "reliquary.validator.corpus_service" else []
            named.setLevel(logging.INFO)
            named.propagate = name != "reliquary" and name != "rehearsal"
            named.disabled = False

    keep_logging()
    store.armed = True
    app = build_corpus_app(
        entry=types.SimpleNamespace(job_id=job.job_id, task_id="rehearsal"),
        job=job, store=store, records=BucketRecordStore(), tokenizer=_Tokenizer(),
        renderer=_Renderer(), verify_signature=lambda request: True,
        auditor=types.SimpleNamespace(enqueue=lambda submission_id: None),
        proof_chunk_tokens=None, prompt_job_for=lambda job: _Prompts(),
        registration=None, **extra,
    )
    app.router.on_startup.append(keep_logging)
    print(f"serving {job.job_id} on {args.port} pid={os.getpid()}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


def _supervise(args) -> None:
    """Each phase is 'CRASHMODE:N' or 'none'; the last phase runs until SIGTERM."""
    child = None

    def stop(*_):
        if child is not None and child.poll() is None:
            child.terminate()
        sys.exit(0)

    signal.signal(signal.SIGTERM, stop)
    for phase in args.phases:
        cmd = [sys.executable, os.path.abspath(__file__), "serve", "--job", args.job,
               "--port", str(args.port)]
        if phase != "none":
            mode, after = phase.split(":")
            cmd += ["--crash", mode, "--crash-after", after]
        print(f"[supervise] phase {phase}", flush=True)
        child = subprocess.Popen(cmd)
        code = child.wait()
        print(f"[supervise] phase {phase} exited {code}", flush=True)
    print("[supervise] all phases done", flush=True)


# --------------------------------------------------------------------------
# load
# --------------------------------------------------------------------------


class _Load:
    def __init__(self, args) -> None:
        from reliquary.corpus.walk import walk_index

        self.args = args
        self.urls = args.url
        self.rng = random.Random(args.seed)
        self.manifest = json.load(open(args.manifest))
        self.job_id = self.manifest["job_id"]
        self.count = self.manifest["prompt_count"]
        self.n = self.manifest["sampling"]["n"]
        self.eos = self.manifest["eos_token_id"]
        self.walk = lambda hk, c: walk_index(self.job_id, hk, c, self.count)
        self.hotkeys = [f"5Rehearsal{args.tag}{i:02d}" for i in range(args.hotkeys)]
        self.log = open(args.log, "a")
        self.accepted: list[dict] = []  # bodies accepted so far, for copies
        self.done = 0
        self.interval = 60.0 / args.rate if args.rate > 0 else 0.0
        self.next_slot = time.monotonic()
        self.copiers = None

    def _copier_for(self, prompt_index: int) -> str:
        # Hotkeys whose walk starts on a given prompt, so a copy reaches the
        # duplicate check instead of stopping at the cursor check.
        if self.copiers is None:
            self.copiers = {}
            k = 0
            while len(self.copiers) < self.count and k < 30 * self.count:
                hk = f"5Copier{self.args.tag}{k}"
                self.copiers.setdefault(self.walk(hk, 0), hk)
                k += 1
        return self.copiers.get(prompt_index)

    def completion(self):
        body = [self.rng.randrange(1, 200000) for _ in range(20)]
        return {"tokens": body + [self.eos], "text": "".join(str(t) for t in body)}

    def body(self, hotkey, cursor, prompt_index, completions):
        return {"job_id": self.job_id, "miner_hotkey": hotkey, "cursor": cursor,
                "prompt_index": prompt_index,
                "checkpoint_sha256": self.manifest["checkpoint_sha256"],
                "rendered_prompt": f"<prompt row-{prompt_index}>",
                "completions": completions, "signature": "ok"}

    def ids(self, body):
        from reliquary.corpus.checks import completion_digest
        from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
        from reliquary.protocol.signatures import corpus_submission_id

        sid = corpus_submission_id(CorpusSubmissionRequest(**body))
        digests = [completion_digest(body["prompt_index"], c["tokens"])
                   for c in body["completions"]]
        return sid, digests

    async def pace(self):
        if not self.interval:
            return
        now = time.monotonic()
        self.next_slot = max(self.next_slot + self.interval, now)
        await asyncio.sleep(self.next_slot - now)

    async def post(self, http, kind, body, url=None):
        """One submission, retried through a dead server until it gets an answer;
        every attempt is logged, and an attempt with no answer is 'unknown'."""
        sid, digests = self.ids(body)
        url = url or self.rng.choice(self.urls)
        while True:
            started = time.perf_counter()
            try:
                response = await http.post(f"{url}/corpus/submit", json=body, timeout=180)
                status = response.status_code
                payload = response.json()
            except Exception as exc:  # the process died mid-request
                status, payload = None, {"error": repr(exc)}
            entry = {"t": time.time(), "kind": kind, "url": url,
                     "hotkey": body["miner_hotkey"], "cursor": body["cursor"],
                     "prompt_index": body["prompt_index"], "submission_id": sid,
                     "digests": digests, "status": status,
                     "accepted": bool(status == 200 and payload.get("accepted")),
                     "reason": payload.get("reason") if status == 200 else payload.get("detail", payload.get("error")),
                     "latency": time.perf_counter() - started}
            if self.args.log_bodies and entry["accepted"]:
                entry["body"] = body
            self.log.write(json.dumps(entry) + "\n")
            self.log.flush()
            if status is None or status == 503:
                if not self.args.retry_unknown:
                    return entry
                await self._wait_up(http, url)
                kind = f"{kind}+retry"
                continue
            return entry

    async def _wait_up(self, http, url):
        for _ in range(600):
            try:
                r = await http.get(f"{url}/corpus/job", timeout=5)
                if r.status_code == 200:
                    return
            except Exception:
                pass
            await asyncio.sleep(0.5)
        raise RuntimeError(f"{url} never came back")

    async def cursor(self, http, hotkey):
        url = self.urls[0]
        await self._wait_up(http, url)
        r = await http.get(f"{url}/corpus/cursor/{hotkey}", timeout=60)
        r.raise_for_status()
        return r.json()["cursor"]

    async def miner(self, http, hotkey):
        cursor = await self.cursor(http, hotkey)
        while self.done < self.args.count:
            self.done += 1
            await self.pace()
            roll = self.rng.random()
            if roll < self.args.copy_frac and self.accepted:
                await self.copy(http)
                continue
            prompt_index = self.walk(hotkey, cursor)
            body = self.body(hotkey, cursor, prompt_index,
                             [self.completion() for _ in range(self.n)])
            if self.args.dual:
                # The same body to every route at once: at most one may accept.
                entries = await asyncio.gather(*(self.post(http, "dual", body, url)
                                                 for url in self.urls))
            else:
                entries = [await self.post(http, "fresh", body)]
            if any(e["accepted"] for e in entries):
                self.accepted.append(body)
            reasons = {e["reason"] for e in entries}
            if any(e["accepted"] for e in entries) or "prompt_full" in reasons or "bad_cursor" in reasons:
                cursor = await self.cursor(http, hotkey) if "bad_cursor" in reasons else cursor + 1
            if any(e["accepted"] for e in entries) and self.rng.random() < self.args.resend_frac / max(self.args.accept_share, 1e-9):
                await self.post(http, "self_resend", body)

    async def copy(self, http):
        source = self.rng.choice(self.accepted)
        copier = self._copier_for(source["prompt_index"])
        if copier is None:
            return
        completions = [dict(c) for c in source["completions"]]
        if self.rng.random() < 0.5:
            # Only one completion copied, the rest fresh.
            keep = self.rng.randrange(self.n)
            completions = [c if i == keep else self.completion() for i, c in enumerate(completions)]
        body = self.body(copier, 0, source["prompt_index"], completions)
        await self.post(http, "cross_copy", body)

    async def run(self):
        import httpx

        limits = httpx.Limits(max_connections=64)
        async with httpx.AsyncClient(limits=limits) as http:
            await asyncio.gather(*(self.miner(http, hk) for hk in self.hotkeys))


def _load(args) -> None:
    load = _Load(args)
    started = time.time()
    asyncio.run(load.run())
    print(f"load done: {load.done} submissions in {time.time() - started:.1f} s")


# --------------------------------------------------------------------------
# check
# --------------------------------------------------------------------------


def _read_logs(paths):
    entries = []
    for path in paths:
        with open(path) as fh:
            entries += [json.loads(line) for line in fh if line.strip()]
    return entries


async def _check(args) -> None:
    from reliquary.infrastructure.corpus_job_store import BucketJobStore
    from reliquary.infrastructure.corpus_record_store import list_submission_ids
    from reliquary.validator.corpus_service import _loaded, verify_ledgers

    store = BucketJobStore()
    job, _ = await store.read_job(args.job)
    report = await verify_ledgers(store, job)
    _, _, state, index = await _loaded(store, job)
    union = set(index) | state.pending
    original = json.load(open(args.original))
    original_seen = set(original["seen"])
    original_filled = sum(original["slots"].values())
    entries = _read_logs(args.log)
    accepted = [e for e in entries if e["accepted"]]
    by_digest: dict[str, list] = {}
    for e in accepted:
        for d in e["digests"]:
            by_digest.setdefault(d, []).append(e["submission_id"])
    twice = {d: s for d, s in by_digest.items() if len(s) > 1}
    accepted_digests = set(by_digest)
    unknown = [e for e in entries if e["status"] is None]
    unknown_landed = [e for e in unknown if set(e["digests"]) <= union and not set(e["digests"]) & accepted_digests]
    partial = [e for e in entries if 0 < len(set(e["digests"]) & union) < len(e["digests"])]
    new_in_union = union - original_seen
    expected_union = original_seen | accepted_digests | {d for e in unknown_landed for d in e["digests"]}
    records = await list_submission_ids(job.job_id)
    accepted_ids = {e["submission_id"] for e in accepted}
    submitted = {d for e in entries for d in e["digests"]}
    out = {
        "verify": report,
        "log_entries": len(entries),
        "accepted_responses": len(accepted),
        "accepted_submissions_distinct": len(accepted_ids),
        "digest_accepted_twice": len(twice),
        "copies_accepted": sum(1 for e in accepted if e["kind"].startswith("cross_copy")),
        "self_resends_accepted": sum(1 for e in accepted if e["kind"].startswith("self_resend")),
        "unknown_attempts": len(unknown),
        "unknown_landed_without_accept_reply": len(unknown_landed),
        "submissions_partially_in_union": len(partial),
        "original_seen": len(original_seen),
        "original_seen_all_present": original_seen <= union,
        "union": len(union),
        "union_minus_original": len(new_in_union),
        "union_has_unsubmitted_digest": bool(new_in_union - submitted),
        "union_equals_original_plus_accepted": union == expected_union,
        "filled": state.slots.filled,
        "filled_minus_original": state.slots.filled - original_filled,
        "accepted_plus_landed": len(accepted_ids) + len(unknown_landed),
        "records": len(records),
        "accepted_without_record": len(accepted_ids - set(records)),
        "records_not_accepted": len(set(records) - accepted_ids - {e["submission_id"] for e in unknown_landed}),
        "reasons": _count(e["kind"] + ":" + str(e["reason"]) for e in entries),
    }
    print(json.dumps(out, indent=2, sort_keys=True))


def _count(items):
    counts: dict[str, int] = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    return dict(sorted(counts.items()))


# --------------------------------------------------------------------------
# timings
# --------------------------------------------------------------------------


def _timings(args) -> None:
    rows: dict[str, list[float]] = {}
    for path in args.log:
        for line in open(path):
            if "corpus submission timing" not in line:
                continue
            for part in line.split(": ", 2)[-1].split():
                if "=" in part:
                    key, value = part.split("=")
                    try:
                        rows.setdefault(key, []).append(float(value))
                    except ValueError:
                        pass
    out = {}
    for key, values in rows.items():
        values.sort()
        out[key] = {"n": len(values), "mean": round(statistics.fmean(values), 4),
                    "p50": round(values[len(values) // 2], 4),
                    "p90": round(values[int(len(values) * 0.9)], 4),
                    "max": round(values[-1], 4)}
    print(json.dumps(out, indent=1))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("seed")
    p.add_argument("--manifest", required=True)
    p.add_argument("--ledger", required=True)
    p = sub.add_parser("serve")
    p.add_argument("--job", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--crash", choices=["after_seal", "after_ledger_write"])
    p.add_argument("--crash-after", type=int, default=1)
    p.add_argument("--no-startup-migration", action="store_true")
    p = sub.add_parser("supervise")
    p.add_argument("--job", required=True)
    p.add_argument("--port", type=int, required=True)
    p.add_argument("phases", nargs="+")
    p = sub.add_parser("load")
    p.add_argument("--url", action="append", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--log", required=True)
    p.add_argument("--hotkeys", type=int, default=20)
    p.add_argument("--count", type=int, default=300)
    p.add_argument("--rate", type=float, default=0.0, help="submissions/min; 0 = as fast as served")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--tag", default="A")
    p.add_argument("--copy-frac", type=float, default=0.05)
    p.add_argument("--resend-frac", type=float, default=0.05)
    p.add_argument("--accept-share", type=float, default=0.5,
                   help="expected accepted share of fresh submissions, to scale resends")
    p.add_argument("--dual", action="store_true")
    p.add_argument("--retry-unknown", action="store_true")
    p.add_argument("--log-bodies", action="store_true", help="keep accepted bodies, to copy later")
    p = sub.add_parser("check")
    p.add_argument("--job", required=True)
    p.add_argument("--original", required=True)
    p.add_argument("--log", action="append", required=True)
    p = sub.add_parser("timings")
    p.add_argument("log", nargs="+")
    args = parser.parse_args()
    if args.command != "timings":
        _require_loopback()
    {"seed": lambda: asyncio.run(_seed(args)), "serve": lambda: _serve(args),
     "supervise": lambda: _supervise(args), "load": lambda: _load(args),
     "check": lambda: asyncio.run(_check(args)), "timings": lambda: _timings(args)}[args.command]()


if __name__ == "__main__":
    main()
