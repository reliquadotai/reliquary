"""Opt-in service policy state, using the existing window/proof/weight clocks."""
from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
import threading
import time

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.protocol.service_contract import ServiceContract, _identifier, _integer
from reliquary.services.admission_policy import service_signal_admits, validate_submission_policy  # noqa: F401
from reliquary.services.observations import observation_id, observation_signal, validate_observation


SUPPORTED_SERVICE_CAPABILITIES = frozenset({
    "environment-reward/v1", "legacy/v1", "public-group-pool/v1", "all/v1",
    "dataset-epoch/v1", "static/v1", "adaptive-rotation/v1", "trainer-driven/v1",
    "task-scoped/v1", "exploration-discount/v1",
})
SERVICE_PAYMENT_POLICY = "service-budgeted-exploration/v1"


class ServicePolicyLimit(ValueError):
    """A proven group is outside the ordered policy, so it earns no entitlement."""


class ServiceRuntime:
    """One qualified controller per task/run. SQLite preserves caps across contexts."""

    def __init__(self, path: str | Path, contract: ServiceContract, qualification: dict, *, now: float | None = None):
        value = contract.to_dict()
        if value["service_kind"] != "adaptive_training":
            raise ValueError("validator requires an adaptive training service")
        contract.require_capabilities(set(SUPPORTED_SERVICE_CAPABILITIES))
        if value["scoring"]["weights_bps"] != {"reward": 10000}:
            raise ValueError("runtime has no qualified multi-metric reward adapter")
        if not isinstance(qualification, dict) or qualification.get("schema") != "service-runtime-qualification/v1" or qualification.get("qualified") is not True:
            raise ValueError("runtime qualification is required")
        if qualification.get("contract_sha256") != contract.sha256:
            raise ValueError("runtime qualification contract mismatch")
        for field in ("dataset", "checkpoint", "environment", "generation_contract_sha256"):
            if qualification.get(field) != value[field]:
                raise ValueError(f"runtime qualification {field} mismatch")
        _identifier(qualification.get("qualification_id"), "qualification_id")
        capabilities = qualification.get("supported_capabilities")
        if not isinstance(capabilities, list) or any(not isinstance(c, str) for c in capabilities):
            raise ValueError("invalid qualification capabilities")
        contract.require_capabilities(set(capabilities))
        rows = qualification.get("row_ids")
        if not isinstance(rows, list) or not rows or len(rows) > 1_000_000:
            raise ValueError("qualification needs bounded source row IDs in prompt index order")
        for row in rows:
            _identifier(row, "row_id")
        if len(set(rows)) != len(rows):
            raise ValueError("qualification needs unique source row IDs in prompt index order")
        self.row_ids = tuple(rows)
        initial_cursor = qualification.get("initial_journal_cursor", -1)
        if type(initial_cursor) is not int or initial_cursor < -1:
            raise ValueError("invalid qualified initial journal cursor")
        self.order_contract = contract
        self.contract = contract
        self.qualification = json.loads(canonical_json_bytes(qualification))
        self.lock = threading.RLock()
        self.db = sqlite3.connect(path, timeout=30, check_same_thread=False)
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.executescript("""
            CREATE TABLE IF NOT EXISTS service_orders(id TEXT PRIMARY KEY, started REAL NOT NULL, clock REAL NOT NULL, groups INTEGER NOT NULL DEFAULT 0, tokens INTEGER NOT NULL DEFAULT 0);
            CREATE TABLE IF NOT EXISTS service_contexts(id TEXT PRIMARY KEY, order_id TEXT NOT NULL, contract TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS service_pools(context TEXT NOT NULL, epoch INTEGER NOT NULL, randomness TEXT NOT NULL, PRIMARY KEY(context,epoch));
            CREATE TABLE IF NOT EXISTS service_observations(seq INTEGER PRIMARY KEY AUTOINCREMENT, id TEXT UNIQUE NOT NULL, order_id TEXT NOT NULL, context TEXT NOT NULL, payload TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS service_payments(order_id TEXT NOT NULL, context TEXT NOT NULL, row_id TEXT NOT NULL, refresh INTEGER NOT NULL, window INTEGER NOT NULL, hotkey TEXT NOT NULL, amount REAL NOT NULL, PRIMARY KEY(order_id,context,row_id,refresh));
            CREATE TABLE IF NOT EXISTS service_settled(order_id TEXT NOT NULL, window INTEGER NOT NULL, aborted INTEGER NOT NULL, PRIMARY KEY(order_id,window));
            CREATE TABLE IF NOT EXISTS service_settlement_maps(order_id TEXT NOT NULL, window INTEGER NOT NULL, payload TEXT NOT NULL, PRIMARY KEY(order_id,window));
            CREATE TABLE IF NOT EXISTS service_training_journal(order_id TEXT NOT NULL, key INTEGER NOT NULL, window INTEGER NOT NULL, digest TEXT NOT NULL, groups TEXT NOT NULL, PRIMARY KEY(order_id,key));
            CREATE TABLE IF NOT EXISTS service_training_stride(order_id TEXT PRIMARY KEY, stride INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS service_consumption(order_id TEXT PRIMARY KEY, cursor INTEGER NOT NULL, q INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS service_windows(order_id TEXT NOT NULL, window INTEGER NOT NULL, context TEXT NOT NULL, envelope TEXT NOT NULL, PRIMARY KEY(order_id,window));
        """)
        instant = time.time() if now is None else now
        if not math.isfinite(instant):
            raise ValueError("invalid order clock")
        try:
            with self.db:
                self.db.execute("BEGIN IMMEDIATE")
                if self.db.execute("SELECT 1 FROM service_orders WHERE id<>? LIMIT 1", (contract.sha256,)).fetchone():
                    raise ValueError("existing service task/run journal belongs to another order; use a new run")
                self.db.execute("INSERT OR IGNORE INTO service_orders(id,started,clock) VALUES(?,?,?)", (contract.sha256, instant, instant))
                self.db.execute("UPDATE service_orders SET clock=MAX(clock,?) WHERE id=?", (instant, contract.sha256))
        except BaseException:
            self.db.close()
            raise
        self._remember_context(contract)
        self.view = None

    def _retire_view(self):
        """Release live projections; historical contexts remain in SQLite."""
        view, self.view = self.view, None
        if view is not None:
            with view._lock:
                if view.eligibility is not None:
                    view.eligibility.close()

    def close(self):
        with self.lock:
            try:
                self._retire_view()
            finally:
                self.db.close()

    def record_training_journal(self, key: int, data: bytes, *, is_tombstone: bool, batches: dict, stride: int):
        """A telemetry projection after the existing payload queue's durable commit."""
        import hashlib
        _integer(key, "journal key", 0)
        _integer(stride, "journal stride", 1)
        identities = [] if is_tombstone else sorted({f"{env}:{group.prompt_idx}" for env, groups in batches.items() for group in groups})
        row = (self.order_contract.sha256, key, key // stride, hashlib.sha256(data).hexdigest(), canonical_json_bytes(identities).decode())
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO service_training_stride VALUES(?,?)", (self.order_contract.sha256, stride))
            if self.db.execute("SELECT stride FROM service_training_stride WHERE order_id=?", (self.order_contract.sha256,)).fetchone()[0] != stride:
                raise ValueError("service trainer journal stride changed")
            self.db.execute("INSERT OR IGNORE INTO service_training_journal VALUES(?,?,?,?,?)", row)
            previous = self.db.execute("SELECT window,digest,groups FROM service_training_journal WHERE order_id=? AND key=?", row[:2]).fetchone()
            if previous != row[2:]:
                raise ValueError("service consumption projection conflicts with its journal key")

    def record_consumption(self, cursor: int) -> int:
        """Only an adopted trainer cursor turns enqueued groups into measured Q."""
        _integer(cursor, "consumed trainer cursor", 0)
        initial = self.qualification.get("initial_journal_cursor", -1)
        if type(initial) is not int or initial < -1:
            raise ValueError("invalid qualified initial journal cursor")
        with self.lock, self.db:
            state = self.db.execute("SELECT cursor,q FROM service_consumption WHERE order_id=?", (self.order_contract.sha256,)).fetchone()
            previous, q = state if state is not None else (initial, 0)
            if cursor < previous:
                raise ValueError("trainer consumption cursor moved backwards")
            if cursor == previous:
                return q
            config = self.db.execute("SELECT stride FROM service_training_stride WHERE order_id=?", (self.order_contract.sha256,)).fetchone()
            stride = config[0] if config is not None else None
            if stride is None:
                q = 0
            else:
                first_window, last_window = (previous + 1) // stride, (cursor + 1) // stride - 1
                if last_window >= first_window:
                    rows = self.db.execute("SELECT key,window,groups FROM service_training_journal WHERE order_id=? AND key>=? AND key<? ORDER BY key", (self.order_contract.sha256, first_window * stride, (last_window + 1) * stride)).fetchall()
                    # Only complete consumed windows qualify Q. Missing facts cannot invent throughput.
                    if len(rows) != (last_window - first_window + 1) * stride:
                        q = 0
                    else:
                        windows = {}
                        for _, window, payload in rows:
                            windows.setdefault(window, set()).update(json.loads(payload))
                        q = sum(len(groups) for groups in windows.values()) // len(windows)
            self.db.execute("INSERT INTO service_consumption VALUES(?,?,?) ON CONFLICT(order_id) DO UPDATE SET cursor=excluded.cursor,q=excluded.q", (self.order_contract.sha256, cursor, q))
            return q

    def measured_consumption(self) -> int:
        with self.lock:
            row = self.db.execute("SELECT q FROM service_consumption WHERE order_id=?", (self.order_contract.sha256,)).fetchone()
            return row[0] if row is not None else 0

    def prepare_view(self, *, window: int, distinct_groups_per_window: int = 0,
                     now: float | None = None):
        """Project verified facts before exposing the exact adopted context.

        The operator-approved runtime qualification supplies the source index
        map and RL provenance. Checkpoint staging/adoption and shared order
        limits remain owned by this controller's caller.
        """
        import os
        from reliquary.constants import M_ROLLOUTS
        from reliquary.services.eligibility import EligibilityStore, _time
        from reliquary.services.runtime_view import ServicePolicyView

        _integer(window, "window", 0)
        _integer(distinct_groups_per_window, "distinct_groups_per_window", 0)
        instant = _time(time.time() if now is None else now)
        with self.lock:
            clock = self.db.execute("SELECT clock FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()[0]
            instant = max(instant, clock)
            with self.db:
                self.db.execute("UPDATE service_orders SET clock=? WHERE id=?", (instant, self.order_contract.sha256))
            value = self.contract.to_dict()
            group_size = _integer(self.qualification.get("group_size"), "qualified group_size", 2, 65536)
            if group_size != M_ROLLOUTS or value["policies"]["sampling"].get("group_size", group_size) != group_size:
                if self.view is not None:
                    self.view.close_admission()
                raise ValueError("qualified group size differs from the active runtime/ordered sampling")
            view = self.view
            if view is not None and view.contract.context_sha256 != self.contract.context_sha256:
                self._retire_view()
                view = None
            if view is None:
                store = None
                if value["policies"]["eligibility"]["kind"] == "dataset-epoch/v1":
                    filename = self.db.execute("PRAGMA database_list").fetchone()[2]
                    path = Path(filename).parent / "eligibility.sqlite3" if filename else ":memory:"
                    store = EligibilityStore(path, self.contract, self.row_ids)
                    try:
                        store.qualify_feed({"context_sha256": self.contract.context_sha256,
                                            "generation_verified": True, "sampling_verified": True,
                                            "rl_group_comparable": True, "group_size": group_size,
                                            "source": "verified-service-groups",
                                            "qualification_id": self.qualification["qualification_id"],
                                            "qualified_order_sha256": self.order_contract.sha256})
                        view = ServicePolicyView(self.contract, self.row_ids, store)
                    except BaseException:
                        store.close()
                        raise
                else:
                    view = ServicePolicyView(self.contract, self.row_ids)
            self.view = view
            view._lock.acquire()
            try:
                if view.eligibility is not None:
                    store = view.eligibility
                    cursor = self.db.execute(
                        "SELECT seq,payload FROM service_observations WHERE order_id=? AND context=? AND seq>? ORDER BY seq",
                        (self.order_contract.sha256, self.contract.context_sha256, getattr(view, "_projection_seq", 0)))
                    pending = cursor.fetchone()
                    if store._current() is None:
                        start = self.db.execute("SELECT started FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()[0]
                        store.begin(window=json.loads(pending[1])["window"] if pending is not None else window, now=start)
                    # Global observation IDs make already projected work a
                    # no-op across epochs. A missing older fact beyond the
                    # durable frontier fails closed; it cannot be reassigned
                    # to a newer epoch merely to repair coverage.
                    clock = store.db.execute("SELECT last_now FROM eligibility_contexts WHERE context=?", (self.contract.context_sha256,)).fetchone()[0]
                    while pending is not None:
                        sequence, payload = pending
                        store.record(json.loads(payload), now=clock)
                        # This cache only avoids repeated parsing within a
                        # live context. Restart always replays the durable
                        # journal and relies on the store's global dedupe.
                        view._projection_seq = sequence
                        pending = cursor.fetchone()
                population = panel = None
                panel_path = os.environ.get("RELIQUARY_SERVICE_PANEL")
                if panel_path and value["policies"]["cooldown"]["kind"] == "adaptive-rotation/v1":
                    path = Path(panel_path)
                    if not path.is_absolute() or not path.is_file():
                        raise ValueError("service panel must be an absolute regular file")
                    with path.open("rb") as handle:
                        raw = handle.read(4 * 1024 * 1024 + 1)
                    if len(raw) > 4 * 1024 * 1024:
                        raise ValueError("service panel exceeds its file size bound")
                    document = json.loads(raw)
                    if not isinstance(document, dict) or set(document) != {"population", "panel"}:
                        raise ValueError("service panel requires population and panel")
                    population, panel = document["population"], document["panel"]
                    if not isinstance(panel, dict) or panel.get("group_size") != group_size:
                        raise ValueError("service panel group differs from the qualified runtime")
                    source_rows = set(self.row_ids)
                    if not isinstance(panel.get("observations"), list) or any(
                        not isinstance(row, dict) or row.get("row_id") not in source_rows
                        for row in panel["observations"]
                    ):
                        raise ValueError("service panel row is absent from the qualified source map")
                view.refresh(window=window, now=instant, population=population, panel=panel,
                             distinct_groups_per_window=distinct_groups_per_window)
                # Context-local epochs cannot reopen the original order's
                # consumed budget or deadline after checkpoint adoption.
                start, clock, groups, tokens = self.db.execute(
                    "SELECT started,clock,groups,tokens FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()
                limits = self.order_contract.to_dict()["limits"]
                if (max(instant, clock) >= start + limits["deadline_seconds"]
                        or groups >= limits["max_groups"] or tokens >= limits["max_tokens"]):
                    view.close_admission()
                return view
            except BaseException:
                view.close_admission()
                raise
            finally:
                view._lock.release()

    def _remember_context(self, contract):
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO service_contexts VALUES(?,?,?)",
                            (contract.context_sha256, self.order_contract.sha256, contract.canonical.decode()))

    def adopt(self, *, repo: str, revision: str, sha256: str):
        with self.lock:
            value = self.order_contract.to_dict()
            if repo != value["checkpoint"]["repo"]:
                raise ValueError("adoption cannot change checkpoint repository")
            value["checkpoint"] = {"repo": repo, "revision": revision, "sha256": sha256}
            contract = ServiceContract.from_dict(value)
            self._remember_context(contract)
            if contract.context_sha256 != self.contract.context_sha256:
                self._retire_view()
            self.contract = contract
            return contract

    def announcement(self, *, window: int, randomness: str) -> dict:
        sampling = self.contract.to_dict()["policies"]["sampling"]
        stride = sampling.get("renewal_windows", 1)
        _integer(window, "window", 0)
        epoch = window // stride
        from reliquary.protocol.service_contract import _sha
        _sha(randomness, "pool randomness")
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO service_pools VALUES(?,?,?)", (self.contract.context_sha256, epoch, randomness))
            beacon = self.db.execute("SELECT randomness FROM service_pools WHERE context=? AND epoch=?", (self.contract.context_sha256, epoch)).fetchone()[0]
        return {"contract": self.contract.to_dict(), "supported_capabilities": sorted(SUPPORTED_SERVICE_CAPABILITIES),
                "pool_epoch": epoch, "pool_randomness": beacon}

    def active(self, *, now: float | None = None) -> bool:
        instant = time.time() if now is None else now
        if not math.isfinite(instant):
            raise ValueError("invalid order clock")
        with self.lock, self.db:
            start, previous = self.db.execute("SELECT started,clock FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()
            instant = max(instant, previous)
            self.db.execute("UPDATE service_orders SET clock=? WHERE id=?", (instant, self.order_contract.sha256))
            limits = self.order_contract.to_dict()["limits"]
            groups, tokens = self.db.execute("SELECT groups,tokens FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()
            return instant < start + limits["deadline_seconds"] and groups < limits["max_groups"] and tokens < limits["max_tokens"]

    def record_verified(self, row: dict, *, hotkey: str, purpose: str, window_pool: float, slots: int, now: float | None = None) -> dict:
        """Called only after the proof scheduler's authoritative PASSED decision."""
        value = validate_observation(row, self.contract)
        group_size = _integer(self.qualification.get("group_size"), "qualified group_size", 2, 65536)
        if value["expected_samples"] != group_size:
            raise ValueError("service observation differs from the qualified group size")
        if value["verification"] != {"generation": "verified", "sampling": "verified", "grading": "graded"}:
            raise ValueError("authoritative service observations require complete verification")
        if value["row_id"] not in self.row_ids or purpose not in {"training", "exploration"}:
            raise ValueError("unknown service row or purpose")
        signal = observation_signal(value, self.contract)
        if (purpose == "training" and not signal.in_zone) or (purpose == "exploration" and signal.category not in {"uniform-low", "uniform-high", "uniform-intermediate"}):
            raise ValueError("signal does not match signed service purpose")
        _integer(slots, "exploration slots", 1)
        _identifier(hotkey, "service hotkey")
        if type(window_pool) not in (int, float) or not math.isfinite(window_pool) or not 0 <= window_pool <= 1:
            raise ValueError("invalid absolute window pool")
        identity, payload = observation_id(value), canonical_json_bytes(value).decode()
        with self.lock:
            active = self.active(now=now)
            self.db.execute("BEGIN IMMEDIATE")
            try:
                frozen = self.db.execute("SELECT context,envelope FROM service_windows WHERE order_id=? AND window=?",
                                         (self.order_contract.sha256, value["window"])).fetchone()
                if frozen is None:
                    raise ServicePolicyLimit("service observation has no frozen window envelope")
                envelope = json.loads(frozen[1])
                if (frozen[0] != self.contract.context_sha256 or envelope["contract"] != self.contract.to_dict()
                        or envelope["window_pool"] != window_pool or envelope["slots"] != slots):
                    raise ServicePolicyLimit("service observation differs from its frozen window envelope")
                existing = self.db.execute("SELECT payload FROM service_observations WHERE id=?", (identity,)).fetchone()
                if existing is not None:
                    if existing[0] != payload:
                        raise ServicePolicyLimit("observation identity conflict")
                    self.db.commit()
                    if self.view is not None:
                        self.view.observe(value, window=value["window"], now=time.time() if now is None else now)
                    return {"inserted": False, "amount": 0.0, "observation_id": identity}
                if not active:
                    started, clock = self.db.execute("SELECT started,clock FROM service_orders WHERE id=?",
                                                    (self.order_contract.sha256,)).fetchone()
                    if clock >= started + self.order_contract.to_dict()["limits"]["deadline_seconds"]:
                        raise ServicePolicyLimit("service order deadline reached")
                    raise ServicePolicyLimit("service order budget exhausted")
                used_groups, used_tokens = self.db.execute("SELECT groups,tokens FROM service_orders WHERE id=?", (self.order_contract.sha256,)).fetchone()
                limits = self.order_contract.to_dict()["limits"]
                if used_groups >= limits["max_groups"] or used_tokens + sum(value["tokens"]) > limits["max_tokens"]:
                    raise ServicePolicyLimit("service order budget exhausted")
                if self.db.execute("SELECT 1 FROM service_settled WHERE order_id=? AND window=?",
                                   (self.order_contract.sha256, value["window"])).fetchone():
                    raise ServicePolicyLimit("window has already settled")
                amount = 0.0
                policy = self.contract.to_dict()["policies"]["reward"]
                if purpose == "exploration":
                    if policy["kind"] != "exploration-discount/v1" or sum(value["tokens"]) > policy["max_tokens_per_group"]:
                        raise ServicePolicyLimit("exploration group is outside its budget")
                    window = value["window"]
                    refresh = window // policy["refresh_windows"]
                    count = self.db.execute("SELECT COUNT(*) FROM service_payments WHERE order_id=? AND window=?", (self.order_contract.sha256, window)).fetchone()[0]
                    if count < slots:
                        amount = window_pool * policy["budget_bps"] / 10000 / slots / policy["divisor"]
                        inserted = self.db.execute("INSERT OR IGNORE INTO service_payments VALUES(?,?,?,?,?,?,?)", (self.order_contract.sha256, self.contract.context_sha256, value["row_id"], refresh, window, hotkey, amount)).rowcount
                        if not inserted:
                            amount = 0.0
                self.db.execute("INSERT INTO service_observations(id,order_id,context,payload) VALUES(?,?,?,?)", (identity, self.order_contract.sha256, self.contract.context_sha256, payload))
                self.db.execute("UPDATE service_orders SET groups=groups+1,tokens=tokens+? WHERE id=?", (sum(value["tokens"]), self.order_contract.sha256))
                self.db.commit()
            except BaseException:
                self.db.rollback()
                raise
            if self.view is not None:
                self.view.observe(value, window=value["window"], now=time.time() if now is None else now)
        return {"inserted": True, "amount": amount, "observation_id": identity}

    def open_window(self, window: int, *, window_pool: float, slots: int):
        _integer(window, "window", 0)
        _integer(slots, "slots", 1)
        if type(window_pool) not in (int, float) or not math.isfinite(window_pool) or not 0 <= window_pool <= 1:
            raise ValueError("invalid service window envelope")
        payload = canonical_json_bytes({"contract": self.contract.to_dict(), "window_pool": window_pool, "slots": slots}).decode()
        with self.lock, self.db:
            self.db.execute("INSERT OR IGNORE INTO service_windows VALUES(?,?,?,?)", (self.order_contract.sha256, window, self.contract.context_sha256, payload))
            saved = self.db.execute("SELECT envelope FROM service_windows WHERE order_id=? AND window=?", (self.order_contract.sha256, window)).fetchone()[0]
            if saved != payload:
                raise ValueError("service window envelope is already frozen")

    def reconcile_archive(self, archive: dict, *, aborted: bool = False) -> dict:
        """Compose with the selected-group journal before its existing durable enqueue."""
        if type(aborted) is not bool:
            raise ValueError("invalid service settlement disposition")
        window = _integer(archive.get("window_start"), "window", 0)
        with self.lock, self.db:
            self.db.execute("BEGIN IMMEDIATE")
            frozen = self.db.execute("SELECT envelope FROM service_windows WHERE order_id=? AND window=?", (self.order_contract.sha256, window)).fetchone()
            if frozen is None:
                raise ValueError("service recovery has no frozen window envelope")
            envelope = json.loads(frozen[0])
            settled = self.db.execute("SELECT aborted FROM service_settled WHERE order_id=? AND window=?", (self.order_contract.sha256, window)).fetchone()
            if settled is not None and settled[0] != int(aborted):
                raise ValueError("service window settlement disposition is frozen")
            exploration = {}
            if not aborted:
                for hotkey, amount in self.db.execute("SELECT hotkey,amount FROM service_payments WHERE order_id=? AND window=?", (self.order_contract.sha256, window)):
                    exploration[hotkey] = exploration.get(hotkey, 0.0) + amount
            result = dict(archive)
            service_fields = {"service_payment_policy": SERVICE_PAYMENT_POLICY,
                              "service_order_contract": self.order_contract.to_dict(),
                              "service_context_contract": envelope["contract"],
                              "service_window_pool": envelope["window_pool"],
                              "service_exploration_slots": envelope["slots"],
                              "exploration_rewards_by_hotkey": exploration}
            enriched = any(key in archive for key in service_fields)
            if enriched and any(archive.get(key) != value for key, value in service_fields.items()):
                raise ValueError("service archive differs from its frozen settlement")
            rewards = archive.get("rewards_by_hotkey", {})
            if not isinstance(rewards, dict) or any(
                not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v) or v < 0
                for k, v in rewards.items()
            ):
                raise ValueError("invalid service reward fraction")
            rewards = dict(rewards)
            if not enriched:
                for hotkey, amount in exploration.items():
                    rewards[hotkey] = rewards.get(hotkey, 0.0) + amount
            result.update(**service_fields, rewards_by_hotkey=rewards)
            validate_service_archive(result, self.order_contract, cap=envelope["window_pool"])
            monetary = canonical_json_bytes({**service_fields, "rewards_by_hotkey": rewards}).decode()
            previous = self.db.execute("SELECT payload FROM service_settlement_maps WHERE order_id=? AND window=?",
                                       (self.order_contract.sha256, window)).fetchone()
            if previous is not None and previous[0] != monetary:
                raise ValueError("service window reward map is already frozen")
            # Validate the complete map before freezing either settlement row.
            self.db.execute("INSERT OR IGNORE INTO service_settled VALUES(?,?,?)", (self.order_contract.sha256, window, int(aborted)))
            self.db.execute("INSERT OR IGNORE INTO service_settlement_maps VALUES(?,?,?)", (self.order_contract.sha256, window, monetary))
            return result

    def snapshot(self, *, after: int = 0, limit: int = 1000) -> dict:
        _integer(after, "after", 0)
        _integer(limit, "limit", 1, 10000)
        with self.lock:
            rows = self.db.execute("SELECT seq,payload FROM service_observations WHERE order_id=? AND context=? AND seq>? ORDER BY seq LIMIT ?", (self.order_contract.sha256, self.contract.context_sha256, after, limit)).fetchall()
        return {"schema": "service-observation-delta/v1", "contract_sha256": self.contract.sha256,
                "context_sha256": self.contract.context_sha256, "watermark": rows[-1][0] if rows else after,
                "observations": [json.loads(row[1]) for row in rows]}

    def training_pool(self, original):
        policy = self.contract.to_dict()["policies"]["reward"]
        fraction = 1 - policy["budget_bps"] / 10000 if policy["kind"] == "exploration-discount/v1" else 1.0
        return {k: v * fraction for k, v in original.items()} if isinstance(original, dict) else original * fraction


def validate_service_archive(record: dict, ordered: ServiceContract, *, cap: float) -> None:
    """Weight readers reject incompatible service journals before assigning money."""
    if type(cap) not in (int, float) or not math.isfinite(cap) or not 0 <= cap <= 1 + 1e-12:
        raise ValueError("invalid service archive cap")
    if record.get("service_payment_policy") != SERVICE_PAYMENT_POLICY:
        raise ValueError("service archive payment policy is missing or unsupported")
    root = ServiceContract.from_dict(record.get("service_order_contract"))
    context = ServiceContract.from_dict(record.get("service_context_contract"))
    if root != ordered:
        raise ValueError("service archive belongs to another order revision")
    expected, actual = root.to_dict(), context.to_dict()
    if expected["checkpoint"]["repo"] != actual["checkpoint"]["repo"]:
        raise ValueError("service checkpoint repository changed")
    expected.pop("checkpoint")
    actual.pop("checkpoint")
    if expected != actual:
        raise ValueError("service context changes more than its adopted checkpoint")
    pool = record.get("service_window_pool")
    if type(pool) not in (int, float) or not math.isfinite(pool) or not 0 <= pool <= cap + 1e-12:
        raise ValueError("service archive exceeds its declared absolute envelope")
    _integer(record.get("service_exploration_slots"), "service exploration slots", 1)
    rewards, exploration = record.get("rewards_by_hotkey"), record.get("exploration_rewards_by_hotkey")
    if not isinstance(rewards, dict) or not isinstance(exploration, dict):
        raise ValueError("service archive reward maps are required")
    for values in (rewards, exploration):
        if any(not isinstance(k, str) or type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= pool + 1e-12
               for k, v in values.items()):
            raise ValueError("invalid service reward fraction")
    if any(v > rewards.get(k, 0) + 1e-12 for k, v in exploration.items()):
        raise ValueError("exploration is absent from the authoritative reward map")
    policy = context.to_dict()["policies"]["reward"]
    reserve = pool * policy["budget_bps"] / 10000 if policy["kind"] == "exploration-discount/v1" else 0
    discounted = reserve / policy["divisor"] if reserve else 0
    training = math.fsum(rewards.values()) - math.fsum(exploration.values())
    if math.fsum(exploration.values()) > discounted + 1e-12 or training > pool - reserve + 1e-12:
        raise ValueError("service reward lane exceeds its reserved fraction")
    if record.get("window_status") == "aborted" and exploration:
        raise ValueError("aborted service window cannot award exploration")
