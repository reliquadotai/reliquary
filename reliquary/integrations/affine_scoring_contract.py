"""Bounded I/O checks for the pinned historical V9 gradient computation.

These checks neither authenticate a chain seed nor recompute model gradients.
Private native documents and identity prefixes remain in the local context.
The small chunk/seed replay is pinned by CHUNK_SOURCE_SHA256 and checked against
upstream pure-function goldens; the upstream signer module is never imported.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
import stat
import zipfile
from pathlib import Path

import numpy as np

MAX_JSON = 16 * 1024**2
MAX_DOCS_BYTES = 256 * 1024**2
MAX_ARRAY_BYTES = 512 * 1024**2
MAX_TOKEN = 1 << 20
HEX = re.compile(r"[0-9a-f]{64}")
CHUNK_SOURCE_SHA256 = "014889c11c97022ee498879fc0abee50c4981379cb39dc599917e4c21facd1c6"


def _need(value, reason):
    if not value:
        raise ValueError(reason)


def _pairs(rows):
    result = {}
    for key, value in rows:
        _need(key not in result, "duplicate JSON key")
        result[key] = value
    return result


def _parse(raw):
    def invalid(_):
        raise ValueError("nonfinite JSON number")

    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=invalid)


def _file(path, limit):
    path = Path(path)
    _need(
        not path.is_symlink() and stat.S_ISREG(path.stat().st_mode),
        "ordinary input file required",
    )
    _need(path.stat().st_size <= limit, "input file too large")
    return path


def _json(path, limit=MAX_JSON):
    return _parse(_file(path, limit).read_bytes())


def _hs(seed, *labels):
    text = ":".join(map(str, (seed,) + labels))
    return int(hashlib.sha256(text.encode()).hexdigest()[:8], 16) & 0x7FFFFFFF


def _job(job):
    _need(
        type(job.get("epoch")) is int and 0 <= job["epoch"] <= 10**7, "epoch required"
    )
    # Native multi-shard jobs wait for a shared complete reference-gradient cache.
    _need(
        type(job.get("nshards")) is int
        and job["nshards"] == 1
        and type(job.get("shard")) is int
        and job["shard"] == 0,
        "single shard 0/1 required",
    )
    _need(
        job.get("layers") == 4 and type(job["layers"]) is int,
        "four MLP layers required",
    )
    _need(
        isinstance(job.get("baseCheckpoint"), str)
        and HEX.fullmatch(job["baseCheckpoint"]),
        "full checkpoint required",
    )
    _need(
        job.get("assurance") in ("fixture", "historical-v9-compute"),
        "explicit assurance required",
    )
    return {
        key: job[key]
        for key in (
            "epoch",
            "shard",
            "nshards",
            "layers",
            "baseCheckpoint",
            "assurance",
        )
    }


def _tokens(values, maximum):
    _need(
        type(values) is list
        and len(values) <= maximum
        and all(type(v) is int and 0 <= v < MAX_TOKEN for v in values),
        "bounded integer token array required",
    )


def _arrays(path):
    """Check NPY headers before allocation; never deserialize Python objects."""
    path = _file(path, MAX_ARRAY_BYTES)
    with zipfile.ZipFile(path) as archive:
        entries = archive.infolist()
        _need(
            len(entries) == 2
            and {e.filename for e in entries} == {"tokens.npy", "offsets.npy"},
            "exact NPZ arrays required",
        )
        _need(
            sum(e.file_size for e in entries) <= MAX_ARRAY_BYTES,
            "expanded NPZ too large",
        )
        result = {}
        for entry in entries:
            _need(
                entry.compress_type in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED),
                "NPZ compression unsupported",
            )
            with archive.open(entry) as stream:
                version = np.lib.format.read_magic(stream)
                _need(version in ((1, 0), (2, 0)), "NPY version unsupported")
                reader = (
                    np.lib.format.read_array_header_1_0
                    if version == (1, 0)
                    else np.lib.format.read_array_header_2_0
                )
                shape, fortran, dtype = reader(stream, max_header_size=10000)
                wanted = np.dtype(
                    "int32" if entry.filename == "tokens.npy" else "int64"
                )
                _need(
                    len(shape) == 1
                    and type(shape[0]) is int
                    and shape[0] >= 0
                    and not fortran
                    and dtype == wanted,
                    "one-dimensional native integer NPZ arrays required",
                )
                _need(
                    stream.tell() + shape[0] * dtype.itemsize == entry.file_size,
                    "NPY shape/size mismatch",
                )
            with archive.open(entry) as stream:
                result[entry.filename[:-4]] = np.lib.format.read_array(
                    stream, allow_pickle=False
                )
    return result["tokens"], result["offsets"]


def _seed(record, epoch):
    _need(
        type(record) is dict and record.get("rule") == "sn120-v9-seed-rule-v1",
        "seed rule required",
    )
    _need(
        str(record.get("epoch")) == str(epoch)
        and isinstance(record.get("epoch_id"), str),
        "seed epoch mismatch",
    )
    for field in (
        "anchor_block",
        "seed_block",
        "margin_blocks",
        "anchor_ts_ms",
        "deadline",
    ):
        _need(
            type(record.get(field)) is int and record[field] >= 0,
            "integer seed coordinates required",
        )
    _need(
        record["margin_blocks"] == 10
        and record["seed_block"] == record["anchor_block"] + 10
        and record["anchor_ts_ms"] >= record["deadline"] * 1000,
        "post-deadline seed coordinates required",
    )
    block_hash = record.get("seed_block_hash")
    _need(
        isinstance(block_hash, str) and re.fullmatch(r"0x[0-9a-f]{64}", block_hash),
        "full seed block hash required",
    )
    expected = hashlib.sha256(
        f"sn120-v9-ref-seed-v1:{record['epoch_id']}:{block_hash}".encode()
    ).hexdigest()
    _need(record.get("seed") == expected, "seed derivation mismatch")
    return expected


def _chunks(meta, seed):
    # Native build_chunks, pinned by CHUNK_SOURCE_SHA256; no signer module import.
    by = {}
    for row in meta:
        by.setdefault(row["index"], []).append(row)
    info = {
        i: xs
        for i, xs in by.items()
        if 2 <= sum(x["outcome"] == "positive" for x in xs) <= 6
        and any(x["outcome"] in ("negative", "neutral") for x in xs)
    }
    subjects = {}
    for index in sorted(info):
        subjects.setdefault(info[index][0]["subject"], []).append(index)
    rng = random.Random(_hs(seed, "chunks"))
    chunks = []
    for subject in sorted(subjects):
        tasks = subjects[subject][:]
        rng.shuffle(tasks)
        halves = [tasks[0::2], tasks[1::2]]
        for half in (0, 1):
            for chunk in range(min(6, len(halves[half]) // 4)):
                chunks.append(
                    dict(
                        kind="subject",
                        subject=subject,
                        half=half,
                        c=chunk,
                        tasks=halves[half][chunk * 4 : (chunk + 1) * 4],
                    )
                )
    _need(0 < len(chunks) <= 128, "bounded informative reference chunks required")
    return chunks


def validate_inputs(inputs: Path, job: dict) -> dict:
    binding = _job(job)
    inputs = Path(inputs)
    _need(
        inputs.is_dir() and not inputs.is_symlink(), "ordinary input directory required"
    )
    seed = _seed(_json(inputs / "seed.json"), job["epoch"])
    tasks = _json(inputs / "tasks.json")
    _need(
        type(tasks) is list and 1 <= len(tasks) <= 10000,
        "bounded task snapshot required",
    )
    refs = inputs / "ref"
    _need(
        refs.is_dir() and not refs.is_symlink(), "ordinary reference directory required"
    )
    files = sorted(refs.glob("gen-*.json"))
    _need(1 <= len(files) <= 64, "bounded reference shards required")
    numbers = []
    meta, seen = [], set()
    totals = set()
    for path in files:
        match = re.fullmatch(r"gen-(0|[1-9][0-9]*)\.json", path.name)
        _need(match is not None, "reference shard filename required")
        number = int(match.group(1))
        numbers.append(number)
        record = _json(path)
        _need(
            record.get("epoch") == str(job["epoch"])
            and record.get("base") == job["baseCheckpoint"]
            and record.get("seed") == seed
            and type(record.get("shard")) is int
            and record["shard"] == number,
            "reference epoch/base/seed/shard mismatch",
        )
        _need(
            type(record.get("ref_tasks_total")) is int
            and 1 <= record["ref_tasks_total"] <= 2000,
            "bounded reference task total required",
        )
        totals.add(record["ref_tasks_total"])
        rows = record.get("meta")
        _need(
            type(rows) is list and 0 < len(rows) <= 16000,
            "bounded reference metadata required",
        )
        tokens, offsets = _arrays(path.with_suffix(".npz"))
        _need(
            len(offsets) == len(rows) + 1
            and offsets[0] == 0
            and offsets[-1] == len(tokens)
            and np.all(offsets[1:] >= offsets[:-1])
            and np.all(tokens >= 0)
            and np.all(tokens < MAX_TOKEN),
            "reference token offsets mismatch",
        )
        for n, row in enumerate(rows):
            _need(
                type(row) is dict
                and row.get("kind") == "ref"
                and type(row.get("index")) is int
                and 0 <= row["index"] < len(tasks)
                and type(row.get("j")) is int
                and 0 <= row["j"] < 8,
                "reference task/attempt required",
            )
            key = (row["index"], row["j"])
            _need(key not in seen, "duplicate reference attempt")
            seen.add(key)
            task = tasks[row["index"]]
            _need(
                type(task) is dict and type(task.get("data")) is dict,
                "task asset object required",
            )
            data = task["data"]
            _need(
                isinstance(row.get("subject"), str)
                and row["subject"] == data.get("subject")
                and all(
                    isinstance(data.get(k), str)
                    for k in ("system_prompt", "prompt", "answer")
                ),
                "reference task asset mismatch",
            )
            _need(
                type(row.get("len")) is int
                and row["len"] == int(offsets[n + 1] - offsets[n]) <= 8192,
                "reference length mismatch",
            )
            _need(
                row.get("outcome")
                in ("positive", "negative", "neutral", "indeterminate"),
                "reference outcome required",
            )
            meta.append(dict(row, _shard=number))
    _need(
        set(numbers) == set(range(len(files))) and len(totals) == 1,
        "complete reference shard set required",
    )
    _need(
        {p.stem for p in refs.glob("gen-*.npz")} == {p.stem for p in files},
        "reference NPZ pair mismatch",
    )
    indices = sorted({r["index"] for r in meta})
    _need(
        len(indices) == next(iter(totals)) and len(meta) == len(indices) * 8,
        "eight reference attempts per task required",
    )
    owner = {index: n % len(files) for n, index in enumerate(indices)}
    _need(
        all(r["_shard"] == owner[r["index"]] for r in meta),
        "reference task shard mismatch",
    )
    chunks = _chunks(meta, seed)
    todo = _json(inputs / "todo.json")
    _need(type(todo) is list and 1 <= len(todo) <= 8192, "bounded todo required")
    wanted = set()
    for item in todo:
        _need(
            type(item) is list
            and len(item) == 2
            and isinstance(item[0], str)
            and re.fullmatch(r"[0-9a-f]{16}", item[0])
            and type(item[1]) is int
            and 0 <= item[1] < 9,
            "native todo identity/slot required",
        )
        _need(tuple(item) not in wanted, "duplicate todo document")
        wanted.add(tuple(item))
    docs, prefixes, expected = set(), {}, []
    path = _file(inputs / "docs.jsonl.gz", MAX_DOCS_BYTES)
    size = 0
    with gzip.open(path, "rb") as stream:
        while line := stream.readline(2 * 1024**2 + 1):
            size += len(line)
            _need(
                len(line) <= 2 * 1024**2
                and size <= MAX_DOCS_BYTES
                and len(docs) < 8192,
                "expanded documents too large",
            )
            row = _parse(line)
            _need(
                type(row) is dict
                and isinstance(row.get("miner_full"), str)
                and HEX.fullmatch(row["miner_full"]),
                "full document identity required",
            )
            miner = row.get("miner")
            _need(
                miner == row["miner_full"][:16]
                and prefixes.get(miner, row["miner_full"]) == row["miner_full"],
                "identity prefix collision",
            )
            prefixes[miner] = row["miner_full"]
            slot = row.get("slot")
            _need(
                type(slot) is int and 0 <= slot < 9 and (miner, slot) not in docs,
                "unique document slot required",
            )
            docs.add((miner, slot))
            _need(
                row.get("built_on") == job["baseCheckpoint"][:16],
                "document checkpoint mismatch",
            )
            _need(
                type(row.get("index")) is int and 0 <= row["index"] < len(tasks),
                "document task required",
            )
            _tokens(row.get("prompt"), 12288)
            rollouts = row.get("rollouts")
            _need(
                type(rollouts) is list and len(rollouts) == 8,
                "eight document rollouts required",
            )
            for rollout in rollouts:
                _need(type(rollout) is dict, "rollout object required")
                _tokens(rollout.get("out"), 8192)
                _need(
                    len(row["prompt"]) + len(rollout["out"]) <= 12288,
                    "document context too large",
                )
            if (miner, slot) in wanted and _hs(seed, "g", miner, slot) % job[
                "nshards"
            ] == job["shard"]:
                expected.append(
                    dict(miner=miner, slot=slot, lens=[len(r["out"]) for r in rollouts])
                )
    _need(wanted <= docs, "todo document missing")
    return dict(
        binding=binding,
        expectedDocs=expected,
        chunks=chunks,
        seed=seed,
        outputDimensions=dict(
            documents=len(expected),
            rolloutsPerDocument=8,
            chunks=len(chunks),
            referenceTasks=len(indices),
        ),
        chainSeedVerified=False,
    )


def validate_output(path: Path, job: dict, context: dict) -> dict:
    _need(_job(job) == context["binding"], "validated job changed")
    raw = _file(path, 128 * 1024**2).read_bytes()
    result = _parse(raw)
    _need(
        type(result) is dict
        and result.get("epoch") == str(job["epoch"])
        and result.get("base") == job["baseCheckpoint"]
        and result.get("seed") == context["seed"]
        and type(result.get("shard")) is int
        and result["shard"] == job["shard"]
        and result.get("chunks") == context["chunks"],
        "output epoch/base/seed/shard/chunks mismatch",
    )
    rows = result.get("docs")
    _need(
        type(rows) is list and len(rows) == len(context["expectedDocs"]),
        "output document coverage mismatch",
    )
    for row, expected in zip(rows, context["expectedDocs"]):
        _need(
            type(row) is dict
            and set(row) == {"miner", "slot", "lens", "dots"}
            and all(row.get(k) == v for k, v in expected.items()),
            "output document identity/length mismatch",
        )
        dots = row["dots"]
        _need(
            type(dots) is list
            and len(dots) == 8
            and all(type(v) is list and len(v) == len(context["chunks"]) for v in dots),
            "output dot dimensions mismatch",
        )
        _need(
            all(
                type(x) in (int, float) and math.isfinite(x)
                for values in dots
                for x in values
            ),
            "finite gradient dots required",
        )
    timings = result.get("timings")
    _need(
        type(timings) is dict
        and set(timings) == {"load", "ref_grads", "batch_dots"}
        and all(
            type(x) in (int, float) and math.isfinite(x) and x >= 0
            for x in timings.values()
        ),
        "finite native timings required",
    )
    return dict(
        assurance=job["assurance"],
        outputValidated=True,
        documents=len(rows),
        rollouts=len(rows) * 8,
        referenceChunks=len(context["chunks"]),
        outputSha256=hashlib.sha256(raw).hexdigest(),
        chainSeedVerified=False,
        independentNumericalVerification=False,
        upstreamAccepted=False,
        nativeRewardVerified=False,
    )


def create_fixture(destination: Path) -> dict:
    """Write synthetic native-shaped I/O only; this never runs a model."""
    root = Path(destination)
    _need(not root.is_symlink(), "ordinary fixture destination required")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _need(not any(root.iterdir()), "empty fixture destination required")
    root.chmod(0o700)
    (root / "ref").mkdir(mode=0o700)
    job = dict(
        epoch=42,
        shard=0,
        nshards=1,
        layers=4,
        baseCheckpoint="b" * 64,
        assurance="fixture",
    )
    block_hash = "0x" + "c" * 64
    epoch_id = "synthetic-fixture-e000042"
    seed = hashlib.sha256(
        f"sn120-v9-ref-seed-v1:{epoch_id}:{block_hash}".encode()
    ).hexdigest()
    record = dict(
        rule="sn120-v9-seed-rule-v1",
        epoch=42,
        epoch_id=epoch_id,
        seed=seed,
        seed_block_hash=block_hash,
        seed_block=1010,
        anchor_block=1000,
        margin_blocks=10,
        deadline=1700000000,
        anchor_ts_ms=1700000000001,
    )

    def put(name, value):
        path = root / name
        path.write_text(json.dumps(value, allow_nan=False))
        path.chmod(0o600)

    put("seed.json", record)
    put(
        "tasks.json",
        [
            dict(
                data=dict(
                    subject="Algebra",
                    system_prompt="Synthetic fixture",
                    prompt=str(i),
                    answer="0",
                )
            )
            for i in range(8)
        ],
    )
    put("todo.json", [["a" * 16, 0]])
    doc = dict(
        uid=1,
        miner="a" * 16,
        miner_full="a" * 64,
        slot=0,
        index=0,
        built_on="b" * 16,
        trained=True,
        prompt=[248068, 17],
        labels="native",
        rollouts=[
            dict(
                out=[21 + j, 248046],
                reward=float(j < 4),
                cls="positive" if j < 4 else "negative",
                claimed_matches=True,
            )
            for j in range(8)
        ],
    )
    with gzip.open(root / "docs.jsonl.gz", "wt") as stream:
        stream.write(json.dumps(doc) + "\n")
    (root / "docs.jsonl.gz").chmod(0o600)
    meta, tokens, offsets = [], [], [0]
    for index in range(8):
        for j in range(8):
            tokens.extend([21 + j, 248046])
            offsets.append(len(tokens))
            meta.append(
                dict(
                    kind="ref",
                    uid=None,
                    index=index,
                    subject="Algebra",
                    attempt=0,
                    j=j,
                    outcome="positive" if j < 4 else "negative",
                    correct=float(j < 4),
                    reward=float(j < 4),
                    len=2,
                )
            )
    put(
        "ref/gen-0.json",
        dict(
            epoch="42",
            base=job["baseCheckpoint"],
            seed=seed,
            shard=0,
            ref_tasks_total=8,
            timings=dict(gen=0.0),
            wall=0.0,
            meta=meta,
        ),
    )
    np.savez(
        root / "ref/gen-0.npz",
        tokens=np.asarray(tokens, dtype=np.int32),
        offsets=np.asarray(offsets, dtype=np.int64),
    )
    (root / "ref/gen-0.npz").chmod(0o600)
    context = validate_inputs(root, job)
    output = dict(
        epoch="42",
        base=job["baseCheckpoint"],
        seed=seed,
        shard=0,
        chunks=context["chunks"],
        timings=dict(load=0.0, ref_grads=0.0, batch_dots=0.0),
        docs=[
            dict(
                context["expectedDocs"][0],
                dots=[[0.0] * len(context["chunks"]) for _ in range(8)],
            )
        ],
    )
    return dict(job, expectedNativeOutput=output)
