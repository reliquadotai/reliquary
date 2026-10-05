"""A context-bound memory view for admission and window-boundary policy refresh."""

from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import json
import threading

from reliquary.protocol.release_contract import canonical_json_bytes, canonical_sha256
from reliquary.protocol.service_contract import ServiceContract, _identifier, _integer
from reliquary.services.cooldown_policy import propose_cooldown
from reliquary.services.eligibility import EligibilityStore, _time
from reliquary.services.observations import validate_observation


class ServicePolicyView:
    """The owner supplies the exact adopted context and frozen source index map.

    An epoch policy without a qualified feed stays closed. The owner retains
    the original order's shared budget across adopted contexts; no view rewrites
    that order, a checkpoint, or optimizer state.
    """

    def __init__(self, contract: ServiceContract, row_ids: tuple[str, ...],
                 eligibility: EligibilityStore | None = None, cooldown: dict | None = None):
        if not isinstance(row_ids, tuple) or not row_ids or len(set(row_ids)) != len(row_ids):
            raise ValueError("policy view needs the frozen unique source index map")
        for identifier in row_ids:
            _identifier(identifier, "source row id")
        policy = contract.to_dict()["policies"]
        epochs = policy["eligibility"]["kind"] == "dataset-epoch/v1"
        if epochs and eligibility is None:
            raise ValueError("epoch activation requires a qualified eligibility store")
        if eligibility is not None and (not epochs or eligibility.contract != contract
                                        or set(eligibility.source_rows()) != set(row_ids)):
            raise ValueError("eligibility store differs from the adopted context/source index map")
        self.contract, self.row_ids, self.eligibility = contract, row_ids, eligibility
        self._indices = {identifier: index for index, identifier in enumerate(row_ids)}
        self._lock = threading.RLock()
        self._revision = 0
        self._eligible = frozenset()
        self._epoch_snapshot = None
        self._active_rows = frozenset()
        self._active_population = None
        self._window = None
        self._last_now = None
        self._cooldown_policy = policy["cooldown"]
        self._previous = deepcopy(cooldown)
        self._cooldown_windows = self._fallback_windows()
        self._fallback = self._cooldown_policy["kind"] == "adaptive-rotation/v1"
        self._reasons = ["unprepared"]
        if eligibility is not None:
            eligibility.db.execute("CREATE TABLE IF NOT EXISTS service_policy_views (context TEXT PRIMARY KEY, contract TEXT NOT NULL, payload TEXT NOT NULL)")
            eligibility.db.commit()
            saved = eligibility.db.execute("SELECT contract,payload FROM service_policy_views WHERE context=?", (contract.context_sha256,)).fetchone()
            if saved is not None:
                if saved[0] != contract.sha256:
                    raise ValueError("saved policy view belongs to another ordered context")
                state = json.loads(saved[1])
                self._window = _integer(state["window"], "saved window", 0)
                self._last_now = _time(state["last_now"])
                self._cooldown_windows = state["cooldown_windows"]
                self._previous = state["proposal"] if cooldown is None else deepcopy(cooldown)
                self._fallback, self._reasons = state["fallback"], state["reasons"]
        self._validate_restored_cooldown()

    def _fallback_windows(self) -> int:
        key = "windows" if self._cooldown_policy["kind"] == "static/v1" else "fallback_windows"
        return self._cooldown_policy[key]

    def _validate_restored_cooldown(self) -> None:
        policy = self._cooldown_policy
        if policy["kind"] == "static/v1":
            if self._previous is not None or self._cooldown_windows != policy["windows"]:
                raise ValueError("static cooldown differs from its frozen policy")
            return
        _integer(self._cooldown_windows, "saved cooldown", policy["min_windows"], policy["max_windows"])
        if self._previous is not None:
            if (self._previous.get("schema") != "cooldown-suggestion/v1"
                    or self._previous.get("contract_sha256") != self.contract.sha256
                    or self._previous.get("context_sha256") != self.contract.context_sha256):
                raise ValueError("saved cooldown differs from the adopted context")
            _integer(self._previous.get("windows"), "previous cooldown", policy["min_windows"], policy["max_windows"])
            if self._window is None:
                self._cooldown_windows = self._previous["windows"]

    @property
    def cooldown_windows(self) -> int:
        with self._lock:
            return self._cooldown_windows

    @property
    def revision(self) -> int:
        """Invalidate a cached membership projection without scanning row IDs."""
        with self._lock:
            return self._revision

    def eligible(self, prompt_idx: int) -> bool:
        with self._lock:
            return type(prompt_idx) is int and prompt_idx in self._eligible

    def blocked_indices(self) -> list[int]:
        with self._lock:
            return [index for index in range(len(self.row_ids)) if index not in self._eligible]

    def close_admission(self) -> None:
        """Close a cached projection after a failed refresh or shared-order limit."""
        with self._lock:
            if self._eligible:
                self._eligible = frozenset()
                self._revision += 1

    def _update_membership(self, window: int, now: float) -> None:
        if self.eligibility is None:
            eligible = frozenset(range(len(self.row_ids)))
        else:
            prior_epoch = None if self._epoch_snapshot is None else self._epoch_snapshot["dataset_epoch"]
            self._epoch_snapshot = self.eligibility.snapshot(window=window, now=now)
            epoch = self._epoch_snapshot["dataset_epoch"]
            if prior_epoch != epoch:
                rows = self.eligibility.active_rows()
                self._active_rows = frozenset(rows)
                self._active_population = {"id": f"epoch-{epoch}", "kind": "active",
                                           "sha256": canonical_sha256(rows), "size": len(rows)}
            eligible = frozenset(self._indices[row] for row in self.eligibility.eligible_rows(window=window, now=now))
        if eligible != self._eligible:
            self._eligible = eligible
            self._revision += 1

    def _persist(self) -> None:
        if self.eligibility is not None:
            payload = canonical_json_bytes({"window": self._window, "last_now": self._last_now, "cooldown_windows": self._cooldown_windows,
                                            "proposal": self._previous, "fallback": self._fallback, "reasons": self._reasons}).decode()
            with self.eligibility.db:
                self.eligibility.db.execute("INSERT INTO service_policy_views(context,contract,payload) VALUES(?,?,?) ON CONFLICT(context) DO UPDATE SET payload=excluded.payload",
                                            (self.contract.context_sha256, self.contract.sha256, payload))

    @contextmanager
    def _refresh_guard(self):
        with self._lock:
            try:
                yield
            except BaseException:
                self._eligible = frozenset()
                self._revision += 1
                raise

    def refresh(self, *, window: int, now: float, population: dict | None = None,
                panel: dict | None = None, distinct_groups_per_window: int = 0) -> dict:
        """Advance a ready epoch and cooldown only at a new window boundary."""
        _integer(window, "window", 0)
        now = _time(now)
        with self._refresh_guard():
            if self._window is not None and window < self._window:
                raise ValueError("policy view cannot move backwards to another window")
            boundary = self._window != window
            if self.eligibility is not None:
                snapshot = self.eligibility.begin(window=window, now=now)
                if boundary and snapshot["status"] == "ready":
                    self.eligibility.advance(expected_epoch=snapshot["dataset_epoch"], window=window, now=now)
            self._update_membership(window, now)
            self._last_now = now
            if not boundary:
                # An identical frontier can be retried after a lost response
                # or write failure, but cannot apply different panel evidence.
                if panel is not None:
                    if self._previous is None or self._previous["window"] != window or population is None:
                        raise ValueError("cooldown evidence can change only at a new window boundary")
                    propose_cooldown(self.contract, population, panel, window=window,
                                     distinct_groups_per_window=distinct_groups_per_window,
                                     previous=self._previous)
                self._persist()
                return self.snapshot()
            if self._cooldown_policy["kind"] == "adaptive-rotation/v1":
                if panel is None:
                    self._cooldown_windows = self._fallback_windows()
                    self._fallback, self._reasons = True, ["missing-panel"]
                    if self._previous is not None:
                        self._previous = {**self._previous, "window": window,
                                          "windows": self._cooldown_windows, "fallback": True,
                                          "reasons": self._reasons,
                                          "state": {**self._previous["state"], "ema_windows_bps": None}}
                else:
                    if population is None:
                        raise ValueError("cooldown panel requires its qualified population")
                    expected_size = len(self.row_ids) if population.get("kind") == "source" else self._epoch_snapshot["population_rows"]
                    if population.get("size") != expected_size:
                        raise ValueError("cooldown population count differs from the source/active epoch")
                    if population.get("kind") == "active":
                        if population != self._active_population:
                            raise ValueError("cooldown population differs from the exact active epoch")
                        if any(row.get("row_id") not in self._active_rows for row in panel.get("observations", [])):
                            raise ValueError("cooldown panel response is outside its active epoch population")
                    previous = self._previous
                    if previous is not None and previous["population"] != population:
                        previous = None
                    proposal = propose_cooldown(self.contract, population, panel, window=window,
                                                distinct_groups_per_window=distinct_groups_per_window,
                                                previous=previous)
                    # A fallback with no earlier panel still supplies a real
                    # prior horizon; recovery cannot jump around max_change.
                    if self._window is not None and previous is None and not proposal["fallback"]:
                        change = proposal["windows"] - self._cooldown_windows
                        bound = self._cooldown_policy["max_change_windows"]
                        if abs(change) > bound:
                            proposal["windows"] = self._cooldown_windows + (bound if change > 0 else -bound)
                            proposal["reasons"].append("rate_limited")
                    self._previous = proposal
                    self._cooldown_windows = proposal["windows"]
                    self._fallback, self._reasons = proposal["fallback"], proposal["reasons"]
            else:
                self._fallback, self._reasons = False, ["static-policy"]
            self._window = window
            self._last_now = now
            self._persist()
            self._revision += 1
            return self.snapshot()

    def observe(self, row: dict, *, window: int, now: float) -> bool:
        """Persist qualified facts under the same lock used by admission reads."""
        _integer(window, "window", 0)
        now = _time(now)
        with self._refresh_guard():
            if self._window != window or row.get("window") != window:
                raise ValueError("observation belongs to a different policy window")
            value = validate_observation(row, self.contract)
            # Worker callbacks can acquire this lock out of wall-clock order.
            # Keeping the maximum cannot refund a deadline or reopen an epoch.
            if self._last_now is not None:
                now = max(now, self._last_now)
            if value["row_id"] not in self._indices:
                raise ValueError("observation row is absent from the frozen source index map")
            if self.eligibility is None:
                return False
            inserted = self.eligibility.record(value, now=now)
            self._last_now = now
            self._update_membership(window, now)
            self._persist()
            if inserted:
                self._revision += 1
            return inserted

    def advance(self, *, expected_epoch: int, window: int, now: float) -> dict:
        with self._refresh_guard():
            if self.eligibility is None or (self._window is not None and window < self._window):
                raise ValueError("epoch transition requires a new window frontier")
            if window == self._window:
                current = self.eligibility.snapshot(window=window, now=now)
                if current["dataset_epoch"] == expected_epoch + 1:
                    return self.snapshot()
                raise ValueError("epoch transition requires a new window frontier")
            self.eligibility.advance(expected_epoch=expected_epoch, window=window, now=now)
            return self.refresh(window=window, now=now)

    def snapshot(self) -> dict:
        with self._lock:
            return deepcopy({"schema": "service-policy-view/v1", "context_sha256": self.contract.context_sha256,
                             "contract_sha256": self.contract.sha256, "window": self._window,
                             "revision": self._revision,
                             "source_rows": len(self.row_ids), "eligible_rows": len(self._eligible),
                             "eligibility": self._epoch_snapshot, "cooldown_windows": self._cooldown_windows,
                             "active_population": self._active_population,
                             "cooldown_fallback": self._fallback, "cooldown_reasons": self._reasons,
                             "cooldown_proposal": self._previous})
