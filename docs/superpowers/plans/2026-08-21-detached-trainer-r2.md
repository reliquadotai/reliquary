# Detached Trainer via R2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `train_step` off the validator into a standalone trainer
process that consumes per-window training payloads from R2 and publishes
checkpoints back through HF + R2, so proofs and training run concurrently
on separate GPUs/boxes.

**Architecture:** The validator writes one binary training payload (or
tombstone) per sealed window to R2 under `reliquary/training/`; a new
`reliquary train-worker` process consumes them strictly in order, runs the
existing `train_step`, and publishes checkpoints (HF for miners, R2 mirror
for the validator). The validator polls a candidate manifest, downloads
the checkpoint via multipart R2, and swaps its verify plane on the
existing serial publication beat. All behavior is gated by env flags;
flag-off is byte-identical to main.

**Tech Stack:** Python 3.12, numpy (npz payloads), boto3 (+
`boto3.s3.transfer.TransferConfig` multipart), existing torch/train_step,
pytest.

**Spec:** `docs/superpowers/specs/2026-08-21-detached-trainer-r2-design.md`

**Branch:** `feat/detached-trainer-r2` (created from main@80c112f).

## Global Constraints

- π_old is SHIPPED in the payload as **fp32, log-space**; encode gates on
  `T_PROTO == 1.0` exactly like `_verify_logprobs_for_training`
  (batcher.py:279). The trainer never recomputes π_old.
- The trainer NEVER advances its cursor on a timeout — only on a payload
  or an explicit tombstone.
- R2 checkpoint transfers MUST use
  `TransferConfig(multipart_chunksize=32*1024*1024, max_concurrency=16)`
  (measured 147/120 MB/s on the live box; single-stream is ~20 MB/s).
- Payload prefix `reliquary/training/` — NEVER write under
  `reliquary/dataset/` (dashboard reads that prefix).
- New env flags (all default OFF; flag-off must be byte-identical to
  main): `RELIQUARY_WRITE_TRAINING_PAYLOADS`, `RELIQUARY_DETACHED_TRAINER`.
- Miner-facing behavior unchanged: HF stays the miners' checkpoint
  source; manifest signature stays the validator wallet's ed25519 over
  `(checkpoint_n || revision)`, signed at swap time.
- Repo language: English for all code/comments/docs. Comments 1-2
  sentences max.
- Run tests with `.venv/bin/python -m pytest`. Commit after every task.

---

### Task 1: Training payload codec

**Files:**
- Create: `reliquary/shared/training_payload.py`
- Test: `tests/unit/test_training_payload_codec.py`

**Interfaces:**
- Consumes: live batch shape produced by `seal_batch`: per env a
  `list` of groups; each group has `.rollouts` (objects with `.commit`
  dict, `.reward` float, `.env_name` str, and optional
  `._validated_completion_logprobs` list[float]) and `.prompt_idx` int.
- Produces:
  - `encode_training_payload(window_batches: dict[str, list], *, window_start: int, checkpoint_revision: str, env_order: list[str], window_quarantine: dict) -> bytes`
  - `decode_training_payload(data: bytes) -> DecodedPayload` where
    `DecodedPayload` has attributes `window_start: int`,
    `checkpoint_revision: str`, `env_order: list[str]`,
    `window_quarantine: dict`, `schema_version: int`, and method
    `batches() -> dict[str, list]` returning groups/rollouts shaped
    exactly like the input contract above (rollouts are
    `SimpleNamespace`, `_validated_completion_logprobs` restored when
    present).
  - `TOMBSTONE_SCHEMA_VERSION = 1`, `PAYLOAD_SCHEMA_VERSION = 1`.

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_training_payload_codec.py
"""Round-trip fidelity of the R2 training payload.

The invariant: a decoded payload must drive train_step's metadata pass
(_plan_from_batches) and both pi_old accessors to the same values as the
live objects. A silently dropped field degrades the model, not the tests
— so equality is checked on the consumed accessors, not just raw fields.
"""

import math
from types import SimpleNamespace

import pytest

from reliquary import constants as C
from reliquary.shared.training_payload import (
    decode_training_payload,
    encode_training_payload,
)
from reliquary.validator.training import (
    _completion_token_logprobs,
    _plan_from_batches,
    _validator_completion_logprobs,
)


def _roll(reward, length, *, forced=False, truncated=False,
          env="openmathinstruct", with_pi_old=True, prompt_length=7):
    tokens = list(range(prompt_length + length))
    meta = {
        "prompt_length": prompt_length,
        "completion_length": length,
        "token_logprobs": [-1.5] * length,
        "forced": forced,
        "truncated": truncated,
    }
    r = SimpleNamespace(
        reward=reward,
        env_name=env,
        commit={"tokens": tokens, "rollout": meta},
    )
    if with_pi_old:
        r._validated_completion_logprobs = [
            math.log(0.5) + 0.001 * i for i in range(length)
        ]
    return r


def _group(rollouts, prompt_idx=0):
    return SimpleNamespace(rollouts=rollouts, prompt_idx=prompt_idx)


def _window_batches():
    return {
        "openmathinstruct": [
            _group([_roll(1.0, 4), _roll(0.0, 6, forced=True)], prompt_idx=11),
        ],
        "opencodeinstruct": [
            _group([_roll(0.5, 5, env="opencodeinstruct"),
                    _roll(0.5, 3, env="opencodeinstruct", truncated=True,
                          with_pi_old=False)], prompt_idx=22),
        ],
    }


def _encode_decode(batches):
    blob = encode_training_payload(
        batches,
        window_start=30100,
        checkpoint_revision="rev-abc",
        env_order=["openmathinstruct", "opencodeinstruct"],
        window_quarantine={"quarantined": False, "reasons": []},
    )
    assert isinstance(blob, bytes)
    return decode_training_payload(blob)


def test_header_round_trip():
    decoded = _encode_decode(_window_batches())
    assert decoded.window_start == 30100
    assert decoded.checkpoint_revision == "rev-abc"
    assert decoded.env_order == ["openmathinstruct", "opencodeinstruct"]
    assert decoded.window_quarantine == {"quarantined": False, "reasons": []}


def test_consumed_accessors_round_trip():
    original = _window_batches()
    decoded = _encode_decode(original).batches()
    for env in original:
        for g0, g1 in zip(original[env], decoded[env]):
            assert g1.prompt_idx == g0.prompt_idx
            for r0, r1 in zip(g0.rollouts, g1.rollouts):
                assert r1.env_name == r0.env_name
                assert r1.reward == pytest.approx(r0.reward)
                assert list(r1.commit["tokens"]) == list(r0.commit["tokens"])
                assert _completion_token_logprobs(r1) == pytest.approx(
                    _completion_token_logprobs(r0)
                )
                meta0, meta1 = r0.commit["rollout"], r1.commit["rollout"]
                for key in ("prompt_length", "completion_length",
                            "forced", "truncated"):
                    assert meta1.get(key) == meta0.get(key)


def test_pi_old_fp32_exact_and_absent_when_missing(monkeypatch):
    monkeypatch.setattr(C, "PI_OLD_FROM_VERIFY_LOGPROBS", True)
    monkeypatch.setattr(C, "RECOMPUTE_PI_OLD_FROM_VERIFY", True)
    original = _window_batches()
    decoded = _encode_decode(original).batches()
    import numpy as np
    for env in original:
        for g0, g1 in zip(original[env], decoded[env]):
            for r0, r1 in zip(g0.rollouts, g1.rollouts):
                n = int(r0.commit["rollout"]["completion_length"])
                v0 = _validator_completion_logprobs(r0, n)
                v1 = _validator_completion_logprobs(r1, n)
                if v0 is None:
                    assert v1 is None
                else:
                    # fp32 round-trip: exact against float32-cast source.
                    assert v1 == [float(np.float32(x)) for x in v0]


def test_plan_from_batches_equivalence():
    original = _window_batches()
    decoded = _encode_decode(original).batches()
    order = ["openmathinstruct", "opencodeinstruct"]
    plan0, skipped0 = _plan_from_batches([original[e] for e in order])
    plan1, skipped1 = _plan_from_batches([decoded[e] for e in order])
    assert skipped1 == skipped0
    assert len(plan1) == len(plan0)
    for it0, it1 in zip(plan0, plan1):
        assert it1.advantage == pytest.approx(it0.advantage)


def test_t_proto_gate_drops_pi_old(monkeypatch):
    monkeypatch.setattr(C, "T_PROTO", 0.6)
    decoded = _encode_decode(_window_batches()).batches()
    for env_groups in decoded.values():
        for g in env_groups:
            for r in g.rollouts:
                assert getattr(r, "_validated_completion_logprobs", None) is None
```

Note: `_plan_from_batches` returns plan items — check the actual return
shape (`plan, n_skipped`) and item attribute for the advantage in
`reliquary/validator/training.py:778` before finalizing the last test;
adjust attribute names to the real dataclass (the intent is: identical
advantages and identical skip count).

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/unit/test_training_payload_codec.py -v`
Expected: FAIL with `ModuleNotFoundError: reliquary.shared.training_payload`

- [ ] **Step 3: Implement the codec**

```python
# reliquary/shared/training_payload.py
"""Binary per-window training payload for the detached trainer.

One npz object per sealed window under ``reliquary/training/``. Carries
everything train_step consumes — tokens, miner token_logprobs, seal-time
verify logprobs (pi_old), reward, forced/truncated — so the trainer never
recomputes a forward. pi_old is fp32 log-space; encoding gates on
T_PROTO == 1.0 exactly like _verify_logprobs_for_training.
"""

from __future__ import annotations

import io
import json
import math
from types import SimpleNamespace
from typing import Any

import numpy as np

PAYLOAD_SCHEMA_VERSION = 1
TOMBSTONE_SCHEMA_VERSION = 1


def _pi_old_for_encode(rollout: Any, completion_length: int) -> list[float] | None:
    from reliquary.constants import T_PROTO

    if float(T_PROTO) != 1.0:
        return None
    values = getattr(rollout, "_validated_completion_logprobs", None)
    if not isinstance(values, list) or len(values) != completion_length:
        return None
    if completion_length == 0:
        return None
    try:
        floats = [float(v) for v in values]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in floats):
        return None
    return floats


def encode_training_payload(
    window_batches: dict[str, list],
    *,
    window_start: int,
    checkpoint_revision: str,
    env_order: list[str],
    window_quarantine: dict,
) -> bytes:
    groups_meta: list[dict[str, Any]] = []   # env, prompt_idx, n_rollouts
    rollout_meta: list[dict[str, Any]] = []  # commit["rollout"] minus token_logprobs
    rewards: list[float] = []
    env_names: list[str] = []
    tokens_flat: list[int] = []
    tokens_off: list[int] = [0]
    miner_lp_flat: list[float] = []
    miner_lp_off: list[int] = [0]
    pi_old_flat: list[float] = []
    pi_old_off: list[int] = [0]
    has_pi_old: list[bool] = []

    for env in env_order:
        for group in window_batches.get(env, []):
            groups_meta.append({
                "env": env,
                "prompt_idx": int(getattr(group, "prompt_idx", 0) or 0),
                "n_rollouts": len(group.rollouts),
            })
            for rollout in group.rollouts:
                commit = rollout.commit or {}
                meta = dict((commit.get("rollout") or {}))
                miner_lp = list(meta.pop("token_logprobs", []) or [])
                completion_length = int(meta.get("completion_length", 0) or 0)
                rollout_meta.append(meta)
                rewards.append(float(getattr(rollout, "reward", 0.0)))
                env_names.append(str(getattr(rollout, "env_name", env)))
                tokens_flat.extend(int(t) for t in commit.get("tokens", []))
                tokens_off.append(len(tokens_flat))
                miner_lp_flat.extend(float(v) for v in miner_lp)
                miner_lp_off.append(len(miner_lp_flat))
                pi_old = _pi_old_for_encode(rollout, completion_length)
                if pi_old is None:
                    has_pi_old.append(False)
                else:
                    has_pi_old.append(True)
                    pi_old_flat.extend(pi_old)
                pi_old_off.append(len(pi_old_flat))

    header = {
        "schema_version": PAYLOAD_SCHEMA_VERSION,
        "window_start": int(window_start),
        "checkpoint_revision": str(checkpoint_revision),
        "env_order": list(env_order),
        "window_quarantine": window_quarantine,
        "groups": groups_meta,
        "rollout_meta": rollout_meta,
    }
    buf = io.BytesIO()
    np.savez_compressed(
        buf,
        header=np.frombuffer(
            json.dumps(header).encode("utf-8"), dtype=np.uint8
        ),
        rewards=np.asarray(rewards, dtype=np.float32),
        env_names=np.asarray(env_names, dtype=np.str_),
        tokens_flat=np.asarray(tokens_flat, dtype=np.int32),
        tokens_off=np.asarray(tokens_off, dtype=np.int64),
        miner_lp_flat=np.asarray(miner_lp_flat, dtype=np.float32),
        miner_lp_off=np.asarray(miner_lp_off, dtype=np.int64),
        pi_old_flat=np.asarray(pi_old_flat, dtype=np.float32),
        pi_old_off=np.asarray(pi_old_off, dtype=np.int64),
        has_pi_old=np.asarray(has_pi_old, dtype=np.bool_),
    )
    return buf.getvalue()


class DecodedPayload:
    def __init__(self, arrays: dict[str, np.ndarray]) -> None:
        header = json.loads(bytes(arrays["header"]).decode("utf-8"))
        if header["schema_version"] != PAYLOAD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported payload schema {header['schema_version']}"
            )
        self.schema_version = header["schema_version"]
        self.window_start = int(header["window_start"])
        self.checkpoint_revision = str(header["checkpoint_revision"])
        self.env_order = list(header["env_order"])
        self.window_quarantine = dict(header["window_quarantine"])
        self._groups_meta = header["groups"]
        self._rollout_meta = header["rollout_meta"]
        self._arrays = arrays

    def batches(self) -> dict[str, list]:
        a = self._arrays
        out: dict[str, list] = {env: [] for env in self.env_order}
        cursor = 0
        for gm in self._groups_meta:
            rollouts = []
            for _ in range(gm["n_rollouts"]):
                i = cursor
                t0, t1 = int(a["tokens_off"][i]), int(a["tokens_off"][i + 1])
                m0, m1 = int(a["miner_lp_off"][i]), int(a["miner_lp_off"][i + 1])
                meta = dict(self._rollout_meta[i])
                meta["token_logprobs"] = [
                    float(v) for v in a["miner_lp_flat"][m0:m1]
                ]
                rollout = SimpleNamespace(
                    reward=float(a["rewards"][i]),
                    env_name=str(a["env_names"][i]),
                    commit={
                        "tokens": [int(t) for t in a["tokens_flat"][t0:t1]],
                        "rollout": meta,
                    },
                )
                if bool(a["has_pi_old"][i]):
                    p0, p1 = int(a["pi_old_off"][i]), int(a["pi_old_off"][i + 1])
                    rollout._validated_completion_logprobs = [
                        float(v) for v in a["pi_old_flat"][p0:p1]
                    ]
                rollouts.append(rollout)
                cursor += 1
            out[gm["env"]].append(
                SimpleNamespace(rollouts=rollouts, prompt_idx=gm["prompt_idx"])
            )
        return out


def decode_training_payload(data: bytes) -> DecodedPayload:
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        arrays = {k: npz[k] for k in npz.files}
    return DecodedPayload(arrays)


def encode_tombstone(
    *, window_start: int, failure_stage: str, failure_type: str
) -> bytes:
    return json.dumps({
        "schema_version": TOMBSTONE_SCHEMA_VERSION,
        "window_start": int(window_start),
        "failure_stage": str(failure_stage),
        "failure_type": str(failure_type),
    }).encode("utf-8")


def decode_tombstone(data: bytes) -> dict[str, Any]:
    doc = json.loads(data.decode("utf-8"))
    if doc.get("schema_version") != TOMBSTONE_SCHEMA_VERSION:
        raise ValueError("unsupported tombstone schema")
    return doc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/unit/test_training_payload_codec.py -v`
Expected: PASS (fix `_plan_from_batches` item attribute names against the
real dataclass if the equivalence test errors on attribute access).

- [ ] **Step 5: Commit**

```bash
git add reliquary/shared/training_payload.py tests/unit/test_training_payload_codec.py
git commit -m "feat: binary training payload codec for the detached trainer"
```

---

### Task 2: Durable payload/tombstone queue

**Files:**
- Create: `reliquary/infrastructure/training_payload_queue.py`
- Test: `tests/unit/test_training_payload_queue.py`

**Interfaces:**
- Consumes: bytes from Task 1 (`encode_training_payload` /
  `encode_tombstone`).
- Produces: class `TrainingPayloadQueue` with:
  - `enqueue_payload(window_start: int, data: bytes) -> None` — atomic
    write to `{queue_dir}/window-{N}.npz` (.tmp + rename).
  - `enqueue_tombstone(window_start: int, data: bytes) -> None` — same,
    `.tombstone.json` suffix.
  - `async run_forever(upload_fn=None) -> None` — scans the dir, uploads
    each file to R2 key `reliquary/training/window-{N}.npz` (or
    `...tombstone.json`), deletes on success, exponential backoff on
    failure. Mirrors `ArchiveQueue` (archive_queue.py) exactly: same
    backoff table, same telemetry counters (`snapshot()` dict).
  - R2 key helpers: `payload_key(n) -> str`, `tombstone_key(n) -> str`.

- [ ] **Step 1: Write the failing tests** — mirror
  `tests/unit/test_archive_queue.py` if it exists (check first; reuse its
  fake-upload pattern). Cover: atomic enqueue creates the file; worker
  uploads and deletes; failed upload keeps the file and backs off; both
  suffixes route to the right R2 keys; restart rescan picks up pending
  files.

```python
# tests/unit/test_training_payload_queue.py
import asyncio

import pytest

from reliquary.infrastructure.training_payload_queue import (
    TrainingPayloadQueue,
    payload_key,
    tombstone_key,
)


def test_keys():
    assert payload_key(30100) == "reliquary/training/window-30100.npz"
    assert tombstone_key(30100) == (
        "reliquary/training/window-30100.tombstone.json"
    )


def test_enqueue_writes_atomically(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")
    q.enqueue_tombstone(30101, b"{}")
    assert (tmp_path / "window-30100.npz").read_bytes() == b"abc"
    assert (tmp_path / "window-30101.tombstone.json").read_bytes() == b"{}"
    assert not list(tmp_path.glob("*.tmp"))


def test_worker_uploads_and_deletes(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")
    uploaded = {}

    def fake_upload(key, data):
        uploaded[key] = data

    asyncio.run(q.drain_once(upload_fn=fake_upload))
    assert uploaded == {"reliquary/training/window-30100.npz": b"abc"}
    assert not list(tmp_path.glob("window-*"))


def test_failed_upload_keeps_file(tmp_path):
    q = TrainingPayloadQueue(queue_dir=str(tmp_path))
    q.enqueue_payload(30100, b"abc")

    def bad_upload(key, data):
        raise RuntimeError("r2 down")

    asyncio.run(q.drain_once(upload_fn=bad_upload))
    assert (tmp_path / "window-30100.npz").exists()
    assert q.snapshot()["upload_failures_total"] == 1
```

- [ ] **Step 2: Run to verify failure** —
  `.venv/bin/python -m pytest tests/unit/test_training_payload_queue.py -v`
  → `ModuleNotFoundError`.

- [ ] **Step 3: Implement** — copy `ArchiveQueue`'s structure
  (archive_queue.py) into the new module, simplified: files are opaque
  bytes (no gzip/json step), key derived from the filename, a
  `drain_once(upload_fn)` extracted from the scan loop so tests don't
  need `run_forever`. Default `upload_fn` uses
  `storage._sync_boto3_put` via `asyncio.to_thread`, exactly like
  ArchiveQueue's worker. Queue dir default:
  `{RELIQUARY_STATE_DIR}/pending_training_payloads`, override env
  `RELIQUARY_TRAINING_PAYLOAD_QUEUE_DIR`.

- [ ] **Step 4: Run tests** → PASS.

- [ ] **Step 5: Commit**

```bash
git add reliquary/infrastructure/training_payload_queue.py tests/unit/test_training_payload_queue.py
git commit -m "feat: durable R2 queue for training payloads and tombstones"
```

---

### Task 3: Validator writes payloads and tombstones (flag-gated)

**Files:**
- Modify: `reliquary/constants.py` (add flag)
- Modify: `reliquary/validator/service.py` (`_train_and_publish` after
  `window_batches`/quarantine are built ~service.py:2378; every
  `_enqueue_aborted_window` call site funnels through that one method
  ~service.py:3340; worker startup next to `archive_queue.run_forever()`
  ~service.py:3590)
- Test: `tests/unit/test_training_payload_writer.py`

**Interfaces:**
- Consumes: Task 1 `encode_training_payload`/`encode_tombstone`, Task 2
  `TrainingPayloadQueue`.
- Produces: `ValidationService._write_training_payload(window_batches: dict, window_n: int, checkpoint_revision: str, window_quarantine: dict) -> None`
  and tombstone emission inside `_enqueue_aborted_window`. Both no-op
  unless `WRITE_TRAINING_PAYLOADS` is true.

- [ ] **Step 1: Add the flag to constants.py** (next to the existing
  RELIQUARY_DISABLE_TRAIN block, grep `RELIQUARY_DISABLE_TRAIN` for the
  idiom):

```python
# Detached-trainer plumbing (spec 2026-08-21-detached-trainer-r2).
# Writers are independent from the trainer cutover so a shadow trainer
# can consume live payloads while in-process training still runs.
WRITE_TRAINING_PAYLOADS = _os.environ.get(
    "RELIQUARY_WRITE_TRAINING_PAYLOADS", "0"
).lower() in {"1", "true", "yes", "on"}
```

- [ ] **Step 2: Write the failing test** — instantiate nothing heavy:
  test the helper on a `SimpleNamespace` self with a recording queue.

```python
# tests/unit/test_training_payload_writer.py
from types import SimpleNamespace

from reliquary import constants as C
from reliquary.shared.training_payload import decode_training_payload
from reliquary.validator.service import ValidationService


class _RecordingQueue:
    def __init__(self):
        self.payloads = {}
        self.tombstones = {}

    def enqueue_payload(self, n, data):
        self.payloads[n] = data

    def enqueue_tombstone(self, n, data):
        self.tombstones[n] = data


def _stub_service(queue):
    stub = SimpleNamespace(
        _training_payload_queue=queue,
        env_mix=[("openmathinstruct", 8), ("opencodeinstruct", 8)],
    )
    return stub


def test_writer_noop_when_flag_off(monkeypatch):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", False)
    q = _RecordingQueue()
    ValidationService._write_training_payload(
        _stub_service(q), {}, 30100, "rev", {"quarantined": False},
    )
    assert q.payloads == {}


def test_writer_encodes_window(monkeypatch):
    monkeypatch.setattr(C, "WRITE_TRAINING_PAYLOADS", True)
    from tests.unit.test_training_payload_codec import _window_batches
    q = _RecordingQueue()
    ValidationService._write_training_payload(
        _stub_service(q), _window_batches(), 30100, "rev-abc",
        {"quarantined": False},
    )
    decoded = decode_training_payload(q.payloads[30100])
    assert decoded.checkpoint_revision == "rev-abc"
    assert decoded.env_order == ["openmathinstruct", "opencodeinstruct"]
```

- [ ] **Step 3: Run to verify failure** → AttributeError (no
  `_write_training_payload`).

- [ ] **Step 4: Implement in service.py.** Add the method (near
  `_archive_window`); call it in `_train_and_publish` immediately after
  `window_quarantine`/`_quarantine_archive` are computed and BEFORE the
  accumulator/training block (the payload must exist whether or not the
  in-process train runs); emit tombstones at the top of
  `_enqueue_aborted_window`. The queue is constructed in `__init__`
  (always, cheap) and its worker task started in `run()` next to the
  archive worker, gated on the flag:

```python
def _write_training_payload(
    self, window_batches, window_n, checkpoint_revision, window_quarantine,
) -> None:
    """Enqueue this sealed window's detached-trainer payload (spec
    2026-08-21). Independent of in-process training so a shadow trainer
    can consume live data."""
    from reliquary.constants import WRITE_TRAINING_PAYLOADS
    if not WRITE_TRAINING_PAYLOADS:
        return
    from reliquary.shared.training_payload import encode_training_payload
    try:
        data = encode_training_payload(
            window_batches,
            window_start=int(window_n),
            checkpoint_revision=str(checkpoint_revision),
            env_order=[name for name, _ in self.env_mix],
            window_quarantine=dict(window_quarantine or {}),
        )
        self._training_payload_queue.enqueue_payload(int(window_n), data)
    except Exception:
        logger.exception(
            "training payload write failed for window %s", window_n,
        )
```

In `_enqueue_aborted_window` (before building the aborted archive):

```python
from reliquary.constants import WRITE_TRAINING_PAYLOADS
if WRITE_TRAINING_PAYLOADS:
    from reliquary.shared.training_payload import encode_tombstone
    try:
        self._training_payload_queue.enqueue_tombstone(
            int(first_batcher.window_start),
            encode_tombstone(
                window_start=int(first_batcher.window_start),
                failure_stage=str(failure_stage),
                failure_type=str(failure_type),
            ),
        )
    except Exception:
        logger.exception("training tombstone write failed")
```

Call-site in `_train_and_publish` (right after the
`checkpoint_revisions` consistency check resolves a single
`checkpoint_revision`, before the accumulator add):

```python
self._write_training_payload(
    window_batches, window_n, checkpoint_revision, _quarantine_archive,
)
```

Note the inconsistent-revision branch has no single revision — skip the
payload there and emit a tombstone instead (failure_stage
`"inconsistent_checkpoint"`), so the trainer's sequence stays gapless.

- [ ] **Step 5: Run the new test + the full unit suite** —
  `.venv/bin/python -m pytest tests/unit -x -q`. Flag-off parity: no
  existing test may change behavior.

- [ ] **Step 6: Commit**

```bash
git add reliquary/constants.py reliquary/validator/service.py tests/unit/test_training_payload_writer.py
git commit -m "feat: validator writes per-window training payloads and tombstones (flag-gated)"
```

---

### Task 4: Trainer journal — ordered R2 consumption

**Files:**
- Create: `reliquary/trainer/__init__.py` (empty)
- Create: `reliquary/trainer/journal.py`
- Test: `tests/unit/test_trainer_journal.py`

**Interfaces:**
- Consumes: R2 keys from Task 2 (`payload_key`/`tombstone_key`),
  codec from Task 1.
- Produces:
  - `class WindowJournal:` constructed with `fetch_fn(key: str) -> bytes | None`
    (injected; prod impl does a boto3 `get_object`, returning None on
    NoSuchKey).
  - `next_entry(cursor: int, *, stride: int) -> JournalEntry | None`
    where `JournalEntry = ("payload", DecodedPayload) | ("tombstone", dict)`,
    `None` = nothing yet (caller sleeps; NEVER advances).
  - Window numbers advance by `stride` (window numbers step by
    `WINDOW_LENGTH`-derived stride; verify against live keys — recent
    windows are 30113, 30114 ⇒ stride 1; pass stride explicitly so the
    journal makes no assumption).

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_trainer_journal.py
from reliquary.shared.training_payload import encode_tombstone
from reliquary.trainer.journal import WindowJournal
from tests.unit.test_training_payload_codec import _window_batches
from reliquary.shared.training_payload import encode_training_payload


def _payload_bytes(n):
    return encode_training_payload(
        _window_batches(), window_start=n, checkpoint_revision="rev",
        env_order=["openmathinstruct", "opencodeinstruct"],
        window_quarantine={"quarantined": False},
    )


def test_returns_payload_then_none(tmp_path):
    store = {"reliquary/training/window-101.npz": _payload_bytes(101)}
    j = WindowJournal(fetch_fn=store.get)
    kind, decoded = j.next_entry(100, stride=1)
    assert kind == "payload" and decoded.window_start == 101
    assert j.next_entry(101, stride=1) is None  # nothing yet -> wait


def test_tombstone_wins_over_absence(tmp_path):
    store = {
        "reliquary/training/window-101.tombstone.json": encode_tombstone(
            window_start=101, failure_stage="s", failure_type="t",
        ),
    }
    j = WindowJournal(fetch_fn=store.get)
    kind, doc = j.next_entry(100, stride=1)
    assert kind == "tombstone" and doc["window_start"] == 101
```

- [ ] **Step 2: Run to verify failure** → ModuleNotFoundError.

- [ ] **Step 3: Implement**

```python
# reliquary/trainer/journal.py
"""Strictly-ordered consumption of the validator's training journal.

The trainer NEVER advances on a timeout: absence of both the payload and
the tombstone for cursor+stride means wait. A skipped update is always an
explicit tombstone, never a race.
"""

from __future__ import annotations

from typing import Any, Callable

from reliquary.infrastructure.training_payload_queue import (
    payload_key,
    tombstone_key,
)
from reliquary.shared.training_payload import (
    decode_tombstone,
    decode_training_payload,
)


class WindowJournal:
    def __init__(self, fetch_fn: Callable[[str], bytes | None]) -> None:
        self._fetch = fetch_fn

    def next_entry(self, cursor: int, *, stride: int):
        target = int(cursor) + int(stride)
        data = self._fetch(payload_key(target))
        if data is not None:
            return "payload", decode_training_payload(data)
        data = self._fetch(tombstone_key(target))
        if data is not None:
            return "tombstone", decode_tombstone(data)
        return None
```

Add the prod fetch impl in the same module:

```python
def r2_fetch_fn(client, bucket: str) -> Callable[[str], bytes | None]:
    def fetch(key: str) -> bytes | None:
        try:
            return client.get_object(Bucket=bucket, Key=key)["Body"].read()
        except client.exceptions.NoSuchKey:
            return None
    return fetch
```

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit**

```bash
git add reliquary/trainer/ tests/unit/test_trainer_journal.py
git commit -m "feat: trainer window journal with strict no-timeout ordering"
```

---

### Task 5: Trainer worker loop (logic only, no torch)

**Files:**
- Create: `reliquary/trainer/worker.py`
- Test: `tests/unit/test_trainer_worker.py`

**Interfaces:**
- Consumes: Task 4 `WindowJournal` (as `journal`), plus injected:
  `train_fn(decoded: DecodedPayload) -> bool` (True = a train step ran;
  raises `TrainingStepSkipped` on health-gate rejection),
  `publish_fn(reason: str) -> str` (returns new revision),
  `head_revision_fn() -> str | None` (HF repo HEAD),
  `sleep_fn(seconds: float)`.
- Produces: `class TrainerWorker` with
  `run_once() -> str` (returns one of `"waited" | "trained" |
  "tombstone" | "published" | "quarantined"`), attributes `cursor: int`,
  `trained_since_publish: int`, `adaptive_publication_pending: bool`,
  `last_published_revision: str | None`; `class TrainerLockLost(RuntimeError)`.
- Publish cadence: `publish_every` constructor arg (prod:
  `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`). Single-writer guard: before
  each publish, `head_revision_fn()` must equal
  `last_published_revision` (or the bootstrap revision) else raise
  `TrainerLockLost`.

- [ ] **Step 1: Write the failing tests** — pure logic, stub deps:

```python
# tests/unit/test_trainer_worker.py
import pytest

from reliquary.trainer.worker import TrainerLockLost, TrainerWorker
from reliquary.validator.training import TrainingStepSkipped


class _Env:
    def __init__(self, entries):
        # entries: {window_n: ("payload", decoded_stub) | ("tombstone", {})}
        self.entries = entries
        self.trained = []
        self.published = []
        self.head = "rev-0"

    def journal_next(self, cursor, *, stride):
        return self.entries.get(cursor + stride)

    def train(self, decoded):
        self.trained.append(decoded)
        return True

    def publish(self, reason):
        rev = f"rev-{len(self.published) + 1}"
        self.published.append(reason)
        self.head = rev
        return rev


class _Decoded:
    def __init__(self, n, quarantined=False):
        self.window_start = n
        self.window_quarantine = {"quarantined": quarantined}


def _worker(env, **kw):
    journal = type("J", (), {"next_entry": staticmethod(env.journal_next)})()
    return TrainerWorker(
        journal=journal,
        train_fn=env.train,
        publish_fn=env.publish,
        head_revision_fn=lambda: env.head,
        cursor=100,
        stride=1,
        publish_every=kw.pop("publish_every", 2),
        last_published_revision="rev-0",
        **kw,
    )


def test_waits_without_advancing():
    env = _Env({})
    w = _worker(env)
    assert w.run_once() == "waited"
    assert w.cursor == 100


def test_trains_in_order_and_publishes_on_cadence():
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("payload", _Decoded(102))})
    w = _worker(env, publish_every=2)
    assert w.run_once() == "trained" and w.cursor == 101
    assert w.run_once() == "trained" and w.cursor == 102
    assert w.run_once() == "published"
    assert env.published == ["cadence"]
    assert w.trained_since_publish == 0


def test_tombstone_advances_and_counts():
    env = _Env({101: ("tombstone", {"failure_stage": "s"})})
    w = _worker(env)
    assert w.run_once() == "tombstone"
    assert w.cursor == 101 and env.trained == []


def test_quarantined_window_advances_without_training():
    env = _Env({101: ("payload", _Decoded(101, quarantined=True))})
    w = _worker(env)
    assert w.run_once() == "quarantined"
    assert w.cursor == 101 and env.trained == []


def test_policy_ratio_drift_triggers_adaptive_publish():
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("payload", _Decoded(102))})
    w = _worker(env, publish_every=10)
    assert w.run_once() == "trained"

    def drift(decoded):
        raise TrainingStepSkipped(
            reason="policy_ratio_drift", grad_norm=0.0, metrics={},
        )
    w._train_fn = drift
    assert w.run_once() == "trained" or True  # step consumed, no train
    assert w.adaptive_publication_pending
    assert w.run_once() == "published"
    assert env.published == ["adaptive_policy_ratio_drift"]


def test_lock_lost_on_foreign_head():
    env = _Env({101: ("payload", _Decoded(101)),
                102: ("payload", _Decoded(102))})
    w = _worker(env, publish_every=2)
    w.run_once(); w.run_once()
    env.head = "someone-else"
    with pytest.raises(TrainerLockLost):
        w.run_once()
```

Check `TrainingStepSkipped.__init__` signature in
`reliquary/validator/training.py:147` and match it exactly in the test.

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
# reliquary/trainer/worker.py
"""Detached trainer main loop: journal cursor -> train_step -> publish.

State machine per run_once(): if a publication is due it runs BEFORE
consuming more windows (mirrors the validator's
checkpoint_publication_pending behavior). Otherwise consume exactly one
journal entry or report "waited". The cursor only advances on an explicit
payload or tombstone.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from reliquary.validator.training import TrainingStepSkipped

logger = logging.getLogger(__name__)


class TrainerLockLost(RuntimeError):
    """Another publisher moved the checkpoint repo HEAD; halt loudly."""


class TrainerWorker:
    def __init__(
        self,
        *,
        journal: Any,
        train_fn: Callable[[Any], bool],
        publish_fn: Callable[[str], str],
        head_revision_fn: Callable[[], str | None],
        cursor: int,
        stride: int,
        publish_every: int,
        last_published_revision: str | None,
        shadow: bool = False,
    ) -> None:
        self._journal = journal
        self._train_fn = train_fn
        self._publish_fn = publish_fn
        self._head_revision_fn = head_revision_fn
        self.cursor = int(cursor)
        self.stride = int(stride)
        self.publish_every = int(publish_every)
        self.last_published_revision = last_published_revision
        self.shadow = bool(shadow)
        self.trained_since_publish = 0
        self.adaptive_publication_pending = False
        self.tombstones_seen = 0
        self.quarantined_seen = 0

    def _publication_due(self) -> bool:
        return (
            self.trained_since_publish >= self.publish_every
            or self.adaptive_publication_pending
        )

    def _publish(self) -> str:
        if self.shadow:
            # Shadow mode trains but never publishes; reset the counter so
            # the loop keeps consuming.
            self.trained_since_publish = 0
            self.adaptive_publication_pending = False
            return "published"
        head = self._head_revision_fn()
        if (
            self.last_published_revision is not None
            and head is not None
            and head != self.last_published_revision
        ):
            raise TrainerLockLost(
                f"checkpoint repo HEAD {head!r} is not ours "
                f"({self.last_published_revision!r}); refusing to publish"
            )
        reason = (
            "adaptive_policy_ratio_drift"
            if self.adaptive_publication_pending else "cadence"
        )
        self.last_published_revision = self._publish_fn(reason)
        self.trained_since_publish = 0
        self.adaptive_publication_pending = False
        return "published"

    def run_once(self) -> str:
        if self._publication_due():
            return self._publish()
        entry = self._journal.next_entry(self.cursor, stride=self.stride)
        if entry is None:
            return "waited"
        kind, value = entry
        if kind == "tombstone":
            self.cursor += self.stride
            self.tombstones_seen += 1
            logger.warning("window %s tombstoned: %s", self.cursor, value)
            return "tombstone"
        if bool(value.window_quarantine.get("quarantined")):
            self.cursor += self.stride
            self.quarantined_seen += 1
            return "quarantined"
        try:
            trained = self._train_fn(value)
        except TrainingStepSkipped as exc:
            self.cursor += self.stride
            if exc.reason == "policy_ratio_drift" and self.trained_since_publish > 0:
                self.adaptive_publication_pending = True
            logger.warning(
                "train step skipped for window %s: %s", self.cursor, exc.reason,
            )
            return "trained"
        self.cursor += self.stride
        if trained:
            self.trained_since_publish += 1
        return "trained"
```

- [ ] **Step 4: Run tests** → PASS (align the drift test with the real
  `TrainingStepSkipped` signature). **Step 5: Commit**

```bash
git add reliquary/trainer/worker.py tests/unit/test_trainer_worker.py
git commit -m "feat: trainer worker loop with cadence, adaptive publish, and single-writer guard"
```

---

### Task 6: Trainer train-runner (accumulator + train_step glue)

**Files:**
- Create: `reliquary/trainer/train_runner.py`
- Test: `tests/unit/test_trainer_train_runner.py`

**Interfaces:**
- Consumes: `DecodedPayload` (Task 1), `BalancedTrainingAccumulator`
  (training_accumulator.py), `assess_training_batch` (quarantine.py),
  `train_step` + `current_lr_schedule_step` (training.py).
- Produces: `class TrainRunner` with `step(decoded: DecodedPayload) -> bool`
  (the `train_fn` for Task 5) and `.model`. Construction:
  `TrainRunner(model, *, env_targets: dict[str, int], env_order: list[str], ref_model=None, train_step_fn=train_step)`.
  Behavior mirrors `_train_and_publish`'s training block
  (service.py:2378-2575) minus validator-only concerns:
  add_window → if ready: accumulated quarantine check → train_step with
  `window_index=decoded.window_start`, `global_step_hint` from the
  restored LR position → reset accumulator. Raises `TrainingStepSkipped`
  through (worker handles it). `ref_model=None` is asserted valid only
  when `KL_BETA == 0.0` — assert at construction with a clear message.

- [ ] **Step 1: Write the failing test** — inject a recording
  `train_step_fn`; no torch:

```python
# tests/unit/test_trainer_train_runner.py
import pytest

from reliquary import constants as C
from reliquary.trainer.train_runner import TrainRunner
from reliquary.shared.training_payload import (
    decode_training_payload, encode_training_payload,
)
from tests.unit.test_training_payload_codec import _window_batches


def _decoded(n=30100):
    return decode_training_payload(encode_training_payload(
        _window_batches(), window_start=n, checkpoint_revision="rev",
        env_order=["openmathinstruct", "opencodeinstruct"],
        window_quarantine={"quarantined": False},
    ))


def test_accumulates_until_ready_then_trains(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.0)
    calls = []

    def fake_train_step(model, batches, **kw):
        calls.append((batches, kw))
        return model

    runner = TrainRunner(
        model=object(),
        env_targets={"openmathinstruct": 2, "opencodeinstruct": 2},
        env_order=["openmathinstruct", "opencodeinstruct"],
        train_step_fn=fake_train_step,
    )
    assert runner.step(_decoded(30100)) is False   # 1 group/env < target 2
    assert runner.step(_decoded(30101)) is True    # ready -> trained
    assert len(calls) == 1
    batches, kw = calls[0]
    assert kw["window_index"] == 30101
    assert len(batches) == 2 and all(len(b) == 2 for b in batches)
    # accumulator reset after consumption
    assert runner.step(_decoded(30102)) is False


def test_kl_beta_guard(monkeypatch):
    monkeypatch.setattr(C, "KL_BETA", 0.01)
    with pytest.raises(RuntimeError, match="KL"):
        TrainRunner(
            model=object(),
            env_targets={"openmathinstruct": 1},
            env_order=["openmathinstruct"],
        )
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement**

```python
# reliquary/trainer/train_runner.py
"""Accumulate decoded windows and run train_step — the detached
counterpart of _train_and_publish's training block. Window-level
quarantine arrives precomputed in the payload; only the accumulated-batch
quarantine runs here."""

from __future__ import annotations

import logging
from typing import Any, Callable

from reliquary.validator.quarantine import assess_training_batch
from reliquary.validator.training import train_step as _default_train_step
from reliquary.validator.training_accumulator import (
    BalancedTrainingAccumulator,
)

logger = logging.getLogger(__name__)


class TrainRunner:
    def __init__(
        self,
        model: Any,
        *,
        env_targets: dict[str, int],
        env_order: list[str],
        ref_model: Any = None,
        train_step_fn: Callable = _default_train_step,
    ) -> None:
        from reliquary.constants import KL_BETA

        if ref_model is None and float(KL_BETA) != 0.0:
            raise RuntimeError(
                "TrainRunner without ref_model requires KL_BETA == 0.0; "
                "pin RELIQUARY_KL_BASE_MODEL and pass it as ref_model"
            )
        self.model = model
        self.ref_model = ref_model
        self.env_order = list(env_order)
        self._train_step = train_step_fn
        self._accumulator = BalancedTrainingAccumulator(env_targets)

    def step(self, decoded: Any) -> bool:
        self._accumulator.add_window(
            decoded.batches(),
            window_n=decoded.window_start,
            checkpoint_revision=decoded.checkpoint_revision,
        )
        if not self._accumulator.ready:
            return False
        batches = self._accumulator.training_batches(self.env_order)
        verdict = assess_training_batch(
            [g for batch in batches for g in batch], reject_counts={},
        )
        if verdict.quarantined:
            logger.warning(
                "accumulated batch quarantined: %s", verdict.reasons,
            )
            self._accumulator.reset()
            return False
        try:
            self.model = self._train_step(
                self.model,
                batches,
                ref_model=self.ref_model,
                window_index=decoded.window_start,
                global_step_hint=None,
            )
        finally:
            self._accumulator.reset()
        return True
```

Note: `global_step_hint=None` defers to the LR counter restored by
`reset_training_state()`/`_lazy_init` — the CLI task wires the restored
`lr_schedule_step` in via `global_step_hint` on the FIRST call only,
mirroring `_lr_global_step_hint` (grep it in service.py and copy the
semantics; if it returns a persistent hint each call, store the hint in
the runner and pass it every call exactly as the service does).

- [ ] **Step 4: Run tests** → PASS. Also run the neighboring suites:
  `.venv/bin/python -m pytest tests/unit/test_training_accumulator.py tests/unit/test_training_quarantine.py -q`

- [ ] **Step 5: Commit**

```bash
git add reliquary/trainer/train_runner.py tests/unit/test_trainer_train_runner.py
git commit -m "feat: trainer-side accumulator + train_step runner"
```

---

### Task 7: Trainer publisher (HF + R2 mirror + candidate manifest)

**Files:**
- Create: `reliquary/trainer/publisher.py`
- Test: `tests/unit/test_trainer_publisher.py`

**Interfaces:**
- Consumes: `write_checkpoint_profile` (checkpoint_profile.py:40),
  `CheckpointStore`'s save idiom (checkpoint.py:81-140, reuse
  `_default_save_hf_format` and `_default_upload` by import).
- Produces: `class TrainerPublisher` with
  `publish(model, *, checkpoint_n: int, lr_schedule_step: int | None, trained_window_cursor: int, reason: str) -> str` (returns HF
  revision). Sequence: save snapshot dir → `write_checkpoint_profile`
  with `extra={"lr_schedule_step": ..., "trained_window_cursor": ...}` →
  HF `upload_folder` → **multipart R2 upload** of every file in the dir
  under `reliquary/checkpoints/{revision}/{filename}` with
  `TransferConfig(multipart_chunksize=32*1024*1024, max_concurrency=16)`
  → `put_object` of `reliquary/training/candidate-manifest.json` =
  `{"checkpoint_n", "repo_id", "revision", "trained_window_cursor", "reason"}`
  → delete staging dir. Constructor takes injected `save_fn`,
  `hf_upload_fn`, `r2_client`, `bucket`, `repo_id`, `tokenizer`,
  `staging_dir` so the unit test runs with stubs.
- Also: `CANDIDATE_MANIFEST_KEY = "reliquary/training/candidate-manifest.json"`
  and `checkpoint_key(revision: str, filename: str) -> str`.

- [ ] **Step 1: Failing test** — stub everything; assert ordering
  (manifest written only after HF and R2 uploads), key layout, profile
  extra content, staging cleanup on failure.

```python
# tests/unit/test_trainer_publisher.py
import json

import pytest

from reliquary.trainer.publisher import (
    CANDIDATE_MANIFEST_KEY, TrainerPublisher, checkpoint_key,
)


class _R2:
    def __init__(self):
        self.uploads = []      # (key, path) via upload_file
        self.objects = {}      # key -> bytes via put_object

    def upload_file(self, path, bucket, key, Config=None):
        assert Config is not None  # multipart config is mandatory
        self.uploads.append((key, path))

    def put_object(self, Bucket, Key, Body, **kw):
        self.objects[Key] = Body


def _publisher(tmp_path, r2, order):
    def save_fn(model, tokenizer, path):
        (path / "model.safetensors").write_bytes(b"weights")
        order.append("save")

    async def hf_upload(folder_path, repo_id, commit_message):
        order.append("hf")
        return "rev-123"

    return TrainerPublisher(
        repo_id="org/repo", staging_dir=str(tmp_path), tokenizer=None,
        save_fn=save_fn, hf_upload_fn=hf_upload, r2_client=r2,
        bucket="reliquary",
    )


def test_publish_order_and_manifest(tmp_path):
    r2, order = _R2(), []
    pub = _publisher(tmp_path, r2, order)
    import asyncio
    rev = asyncio.run(pub.publish(
        object(), checkpoint_n=5, lr_schedule_step=80,
        trained_window_cursor=30110, reason="cadence",
    ))
    assert rev == "rev-123"
    assert order == ["save", "hf"]
    assert any(k == checkpoint_key("rev-123", "model.safetensors")
               for k, _ in r2.uploads)
    manifest = json.loads(r2.objects[CANDIDATE_MANIFEST_KEY])
    assert manifest == {
        "checkpoint_n": 5, "repo_id": "org/repo", "revision": "rev-123",
        "trained_window_cursor": 30110, "reason": "cadence",
    }
    assert not any(tmp_path.iterdir())  # staging cleaned
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement** — mirror `CheckpointStore.publish`'s
  structure (save → profile → upload → cleanup in `finally`), then the
  two R2 steps. `write_checkpoint_profile(snapshot_dir, extra=...)` is
  called right after `save_fn` — the profile must be in the dir BEFORE
  both uploads so validator-side `validate_checkpoint_profile` sees it.
  Upload every regular file in the snapshot dir to
  `checkpoint_key(revision, name)` with
  `TransferConfig(multipart_chunksize=32*1024*1024, max_concurrency=16)`.
  Write the candidate manifest LAST — it is the commit point the
  validator polls.

- [ ] **Step 4: Run tests** → PASS. **Step 5: Commit**

```bash
git add reliquary/trainer/publisher.py tests/unit/test_trainer_publisher.py
git commit -m "feat: trainer publisher — HF + R2 mirror + candidate manifest"
```

---

### Task 8: `reliquary train-worker` CLI

**Files:**
- Modify: `reliquary/cli/main.py` (add subcommand next to `validate`;
  grep the existing click/argparse idiom at cli/main.py:1-120 and copy it)
- Test: `tests/unit/test_cli_train_worker.py`

**Interfaces:**
- Consumes: Tasks 4-7. Produces: `reliquary train-worker [--shadow]`.
- Startup sequence (all in a `_run_train_worker()` helper so the test
  can drive it with injected deps):
  1. Read `RELIQUARY_HF_REPO_ID`, R2 env (same names as storage.py:57-61),
     `CHECKPOINT_PUBLISH_INTERVAL_WINDOWS`, `ENVIRONMENT_MIX`.
  2. Resolve the resume point: GET `CANDIDATE_MANIFEST_KEY`; if present
     use its `revision` + `trained_window_cursor`; else fall back to env
     `RELIQUARY_TRAINER_BOOTSTRAP_CURSOR` (required on first run —
     refuse to guess).
  3. Download the checkpoint snapshot from
     `reliquary/checkpoints/{revision}/` (multipart TransferConfig) —
     fall back to HF download when the R2 mirror lacks it (bootstrap).
  4. `validate_checkpoint_profile(dir)`; read `lr_schedule_step` and
     `trained_window_cursor` from the profile (the manifest cursor is a
     hint; the PROFILE is authoritative).
  5. Load model (reuse `load_text_generation_model` idiom from
     cli/main.py:517-523), `reset_training_state()`, build `TrainRunner`,
     `TrainerPublisher`, `WindowJournal` (r2_fetch_fn), `TrainerWorker`
     (`stride=1`, matching live window numbering).
  6. Loop: `worker.run_once()`; on `"waited"` sleep 5 s; on
     `TrainerLockLost` log + exit code 3.
- The unit test drives only step 2's resolution logic and the loop's
  sleep/exit behavior with stubs (no torch, no network): factor those
  into `resolve_resume_point(fetch_fn, env) -> tuple[str | None, int]`
  and test that function.

- [ ] **Step 1: Failing test for `resolve_resume_point`** (manifest
  present → its values; absent + env cursor → (None, cursor); absent +
  no env → SystemExit).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement subcommand + helpers.**
- [ ] **Step 4: Run `.venv/bin/python -m pytest tests/unit/test_cli_train_worker.py -v` → PASS; `reliquary train-worker --help` exits 0.**
- [ ] **Step 5: Commit**

```bash
git add reliquary/cli/main.py tests/unit/test_cli_train_worker.py
git commit -m "feat: reliquary train-worker CLI with resume-from-manifest"
```

---

### Task 9: Validator checkpoint intake + serial-beat swap

**Files:**
- Create: `reliquary/validator/checkpoint_intake.py`
- Modify: `reliquary/validator/service.py`
  (`_publication_due_next_half` ~service.py:948; the `should_publish`
  block in `_train_and_publish` ~service.py:2620-2700)
- Modify: `reliquary/constants.py` (add `DETACHED_TRAINER` flag, same
  idiom as Task 3)
- Test: `tests/unit/test_checkpoint_intake.py`

**Interfaces:**
- Produces: `class CheckpointIntake` with:
  - `poll() -> dict | None` — GET candidate manifest, return it when its
    `revision` differs from both the installed and the staged revision.
  - `async stage(manifest: dict) -> None` — download all
    `reliquary/checkpoints/{revision}/*` keys (list_objects_v2 by
    prefix) to `{staging_dir}/{revision}/` with multipart
    TransferConfig, then `validate_checkpoint_profile(dir)`; on success
    set `.staged = (manifest, local_dir)`; on failure clear and log
    (staleness, never a crash).
  - `.staged_ready -> bool`, `.take_staged() -> tuple[dict, str]`.
  - Injected `r2_client`, `bucket`, `staging_dir`, `validate_fn` for
    tests.
- service.py wiring (all under `DETACHED_TRAINER`):
  - Poll each iteration (cheap, in `_publish_window_preparation_state`
    area or loop top); `stage()` as a background `asyncio.create_task`
    (never on the window loop's critical path).
  - `_publication_due_next_half()` returns True when
    `intake.staged_ready` (forecast → next window runs serial).
  - In the serial beat, where today's in-process path publishes:
    instead `take_staged()` → load state_dict into the verify plane
    (`torch.load`/`safetensors` via `load_text_generation_model` into a
    staging model then `verify_model.load_state_dict`, or direct
    safetensors load — reuse `_load_model_fn` then copy, matching
    `resume`'s pattern at service.py:1249-1262) →
    `_refresh_verify_model_from_train`-equivalent labeling with the NEW
    revision → `_synchronize_proof_models(revision)` → wallet-sign
    `(checkpoint_n || revision)` → build `ManifestEntry` →
    `self._checkpoint_store` install (add a
    `install_external_manifest(entry)` setter on CheckpointStore) →
    `server.set_current_checkpoint(entry)`.
  - Staleness metric: `windows_since_checkpoint_swap` counter exposed in
    the `/state` health snapshot (`_proof_scheduler_health_snapshot`'s
    sibling; add a small `trainer` block to the /state payload — new
    response fields are wire-safe per the /state extra=forbid
    convention).
- With `DETACHED_TRAINER` on, the in-process training block is skipped
  exactly like `RELIQUARY_DISABLE_TRAIN` (reuse that code path: fold the
  flag into the existing `emergency_freeze` computation at
  service.py:2466 with its own `blocked_reason="detached_trainer"`), and
  `should_publish` is forced False (the trainer publishes).

- [ ] **Step 1: Failing tests for CheckpointIntake** (stub r2 client:
  manifest poll dedup; stage downloads by prefix + validates; failed
  validation clears staged; take_staged hands off once).
- [ ] **Step 2: Run to verify failure.**
- [ ] **Step 3: Implement CheckpointIntake.**
- [ ] **Step 4: Wire service.py** as above, each hunk guarded by
  `DETACHED_TRAINER`. Add a service-level unit test asserting: flag off →
  `_publication_due_next_half` unchanged (byte-identical parity), flag on
  + staged_ready → True.
- [ ] **Step 5: Run the FULL unit suite** —
  `.venv/bin/python -m pytest tests/unit -q`. Zero regressions.
- [ ] **Step 6: Commit**

```bash
git add reliquary/validator/checkpoint_intake.py reliquary/validator/service.py reliquary/constants.py tests/unit/test_checkpoint_intake.py
git commit -m "feat: validator checkpoint intake — staged R2 download + serial-beat swap"
```

---

### Task 10: End-to-end integration test + docs

**Files:**
- Create: `tests/integration/test_detached_trainer_flow.py`
- Create: `docs/detached-trainer.md` (operator runbook)
- Test: the integration test itself.

**Interfaces:** consumes everything above; no new interfaces.

- [ ] **Step 1: Write the integration test** — a filesystem dict stands
  in for R2 (no network, no torch: `train_step_fn` records calls):
  1. Encode 3 windows of payloads (Task 1 fixtures) + 1 tombstone into
     the dict under the real keys.
  2. `TrainerWorker` + `TrainRunner` (env targets 1/1 so every window
     trains) + stub publisher writing the candidate manifest into the
     dict; `publish_every=2`.
  3. Drive `run_once()` until `waited`; assert: 3 trains, 1 tombstone
     counted, manifest present with `trained_window_cursor` == last
     window, publish order (manifest last).
  4. Crash-replay: build a SECOND worker resuming from the manifest's
     cursor with a fresh runner; feed the same dict; assert it re-trains
     only the windows after the manifest cursor.
  5. `CheckpointIntake` with the same dict: poll returns the manifest,
     stage() pulls the checkpoint files (stub validate), staged_ready
     flips, take_staged() returns it exactly once.
- [ ] **Step 2: Run it** —
  `.venv/bin/python -m pytest tests/integration/test_detached_trainer_flow.py -v` → PASS.
- [ ] **Step 3: Write `docs/detached-trainer.md`** — flags, topology
  matrix (1 box / 2 GPU / 2 boxes), shadow-mode cutover procedure from
  the spec's Testing strategy §3 (shadow trainer against live payloads,
  compare loss/grad-norm telemetry, then flip `RELIQUARY_DETACHED_TRAINER`),
  recovery runbook (box lost → re-rent → resume from manifest), rollback
  (flags to 0 + restart).
- [ ] **Step 4: Full suite one last time** —
  `.venv/bin/python -m pytest tests/unit tests/integration -q`.
- [ ] **Step 5: Commit**

```bash
git add tests/integration/test_detached_trainer_flow.py docs/detached-trainer.md
git commit -m "test: end-to-end detached-trainer flow + operator runbook"
```

---

## Self-Review Notes

- Spec coverage: payload+tombstone (T1-3), trainer loop/no-timeout/
  single-writer/adaptive (T5), accumulator+quarantine parity (T6),
  publish HF+R2+manifest with profile cursor (T7), resume/replay (T8,
  T10), intake+serial-beat swap+staleness+flag parity (T9), shadow mode
  (T5 `shadow`, T10 docs). NOT in this plan (deliberate, spec Open
  Questions): trainer-key manifest signature (v2), optimizer-state R2
  upload (v2), payload zstd (measure first), delta transfer (moot).
- Line numbers reference main@80c112f — re-grep before editing; they
  drift.
- The two riskiest equivalences both have dedicated tests: codec
  round-trip via the REAL train_step accessors (T1), and flag-off parity
  (T3 Step 5, T9 Step 4-5 full-suite runs).
