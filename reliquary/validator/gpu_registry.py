"""Durable GPU admission and per-lease attribution, separate from reward/quota math.

An admission is a recent NVIDIA hardware attestation presented by a hotkey. It is
not proof that the model ran in that GPU, nor proof of local attachment. Active
one-device/one-hotkey uniqueness is local to this validator/run, not subnet-wide.
"""
from __future__ import annotations

import hashlib
import json
import secrets
import sqlite3
import os
import stat
import threading
import time
from dataclasses import asdict

from reliquary.protocol.gpu_attestation import canonical, challenge_nonce
from reliquary.validator.gpu_attestation import AttestationError, NvidiaVerifier

MODES = ("off", "observe", "required")


class AdmissionError(ValueError):
    def __init__(self, reason: str, status: int = 409):
        super().__init__(reason)
        self.reason, self.status = reason, status


class GpuRegistry:
    def __init__(self, path, *, validator: str, run: str, mode: str, verifier=None,
                 clock=time.time, admission_ttl_s: int = 300):
        if mode not in MODES or mode == "off":
            raise ValueError("GPU registry mode must be observe or required")
        if not 30 <= admission_ttl_s <= 300:
            raise ValueError("GPU admission lifetime must be 30..300 seconds")
        if not validator or not run:
            raise ValueError("GPU admission requires a validator and run")
        self.validator, self.run, self.mode = validator, run, mode
        self.verifier = verifier if verifier is not None else NvidiaVerifier()
        self.clock, self.ttl = clock, admission_ttl_s
        self.scope = hashlib.sha256(canonical([validator, run])).hexdigest()
        self._lock = threading.RLock()
        # Dedicated state under a trusted private validator directory, never a wallet path.
        fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        try:
            if not stat.S_ISREG(os.fstat(fd).st_mode):
                raise ValueError("GPU state must be a regular file")
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        self._db = sqlite3.connect(path, timeout=5, check_same_thread=False, isolation_level=None)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS gpu_challenges (
                id TEXT PRIMARY KEY, scope TEXT NOT NULL, hotkey TEXT NOT NULL,
                nonce TEXT NOT NULL, issued REAL NOT NULL, expires REAL NOT NULL, used INTEGER NOT NULL DEFAULT 0);
            CREATE INDEX IF NOT EXISTS gpu_challenge_owner ON gpu_challenges(scope, hotkey, issued);
            CREATE TABLE IF NOT EXISTS gpu_admissions (
                scope TEXT NOT NULL, hotkey TEXT NOT NULL, device TEXT NOT NULL,
                verified REAL NOT NULL, expires REAL NOT NULL, report TEXT NOT NULL, declarations TEXT NOT NULL,
                PRIMARY KEY(scope, hotkey));
            CREATE INDEX IF NOT EXISTS gpu_admission_device ON gpu_admissions(scope, device, expires);
            CREATE TABLE IF NOT EXISTS gpu_lease_hardware (
                scope TEXT NOT NULL, lease_id TEXT NOT NULL, hotkey TEXT NOT NULL,
                admission TEXT NOT NULL, PRIMARY KEY(scope, lease_id));
        """)

    def close(self):
        with self._lock:
            self._db.close()

    def challenge(self, hotkey: str) -> dict:
        now = float(self.clock())
        with self._lock, self._transaction():
            self._db.execute("DELETE FROM gpu_challenges WHERE expires < ?", (now - 60,))
            count = self._db.execute("SELECT count(*) FROM gpu_challenges WHERE scope=? AND hotkey=? AND issued>?",
                                     (self.scope, hotkey, now - 60)).fetchone()[0]
            if count >= 6:
                raise AdmissionError("gpu_challenge_rate", 429)
            challenge_id = secrets.token_hex(16)
            nonce = challenge_nonce(challenge_id=challenge_id, secret=secrets.token_hex(32), hotkey=hotkey,
                                    validator=self.validator, run=self.run)
            self._db.execute("INSERT INTO gpu_challenges(id,scope,hotkey,nonce,issued,expires) VALUES(?,?,?,?,?,?)",
                             (challenge_id, self.scope, hotkey, nonce, now, now + 120))
        return {"challenge_id": challenge_id, "nonce": nonce, "issued_at": now, "expires_at": now + 120,
                "validator_hotkey": self.validator, "run": self.run,
                "profile": "nvidia-nras-single-gpu-v1"}

    def _transaction(self):
        from contextlib import contextmanager

        @contextmanager
        def transaction():
            self._db.execute("BEGIN IMMEDIATE")
            try:
                yield
                self._db.execute("COMMIT")
            except BaseException:
                self._db.execute("ROLLBACK")
                raise
        return transaction()

    def attest(self, hotkey: str, challenge_id: str, bundle, declarations: dict) -> dict:
        now = float(self.clock())
        # Consume before expensive verification, even for an invalid signed proof.
        # Atomic SQLite state prevents concurrent/restarted processes accepting it twice.
        with self._lock, self._transaction():
            challenge = self._db.execute("SELECT * FROM gpu_challenges WHERE id=? AND scope=? AND hotkey=?",
                                         (challenge_id, self.scope, hotkey)).fetchone()
            if challenge is None:
                raise AdmissionError("gpu_challenge_unknown")
            if challenge["used"] or challenge["expires"] <= now:
                raise AdmissionError("gpu_challenge_used_or_expired")
            self._db.execute("UPDATE gpu_challenges SET used=1 WHERE id=?", (challenge_id,))
        try:
            report = self.verifier.verify(bundle, nonce=challenge["nonce"], now=now, issued_after=challenge["issued"])
        except AttestationError as exc:
            raise AdmissionError(exc.reason, 403) from exc
        device = hashlib.sha256(canonical([self.scope, report.ueid])).hexdigest()
        accepted_at = float(self.clock())
        expires = min(report.expires_at, report.issued_at + self.ttl)
        if accepted_at >= min(expires, challenge["expires"]):
            raise AdmissionError("gpu_verification_expired", 403)
        # Keep only a pseudonym of the hardware's UEID in durable/public records.
        info = asdict(report)
        del info["ueid"]
        with self._lock, self._transaction():
            conflict = self._db.execute("SELECT hotkey FROM gpu_admissions WHERE scope=? AND device=? AND expires>? "
                                        "AND hotkey<>?", (self.scope, device, accepted_at, hotkey)).fetchone()
            if conflict:
                raise AdmissionError("gpu_already_admitted")
            current = self._db.execute("SELECT device FROM gpu_admissions WHERE scope=? AND hotkey=? AND expires>?",
                                      (self.scope, hotkey, accepted_at)).fetchone()
            if current and current["device"] != device:
                raise AdmissionError("hotkey_already_has_gpu")
            self._db.execute("INSERT INTO gpu_admissions VALUES(?,?,?,?,?,?,?) ON CONFLICT(scope,hotkey) DO UPDATE SET "
                             "device=excluded.device,verified=excluded.verified,expires=excluded.expires,"
                             "report=excluded.report,declarations=excluded.declarations",
                             (self.scope, hotkey, device, accepted_at, expires, canonical(info).decode(),
                              canonical(declarations).decode()))
        return self.status(hotkey)

    def status(self, hotkey: str) -> dict:
        with self._lock:
            row = self._db.execute("SELECT * FROM gpu_admissions WHERE scope=? AND hotkey=?",
                                   (self.scope, hotkey)).fetchone()
        active = row is not None and row["expires"] > float(self.clock())
        answer = {"mode": self.mode, "attested": active, "admission_allowed": active or self.mode != "required",
                  "execution_attested": False, "scope": "validator_run", "evidence": "nvidia_nras" if row else None}
        if row:
            answer.update(device_id=row["device"], verified_at=row["verified"], expires_at=row["expires"],
                          hardware=json.loads(row["report"]), declared=json.loads(row["declarations"]),
                          declarations_attested=False)
        return answer

    def require_admission(self, hotkey: str) -> dict:
        status = self.status(hotkey)
        if not status["admission_allowed"]:
            raise AdmissionError("gpu_attestation_required", 403)
        return status

    def bind_leases(self, hotkey: str, leases, admission: dict):
        """Snapshot admission checked before allocation; refresh never rewrites provenance.

        Does not change a lease or a payout. Existing submissions remain receivable
        after admission expiry, so renewal failures cannot strand generated work.
        """
        with self._lock, self._transaction():
            for lease in leases:
                self._db.execute("INSERT OR IGNORE INTO gpu_lease_hardware VALUES(?,?,?,?)",
                                 (self.scope, lease.lease_id, hotkey, canonical(admission).decode()))

    def lease_hardware(self, lease_id: str) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT admission FROM gpu_lease_hardware WHERE scope=? AND lease_id=?",
                                   (self.scope, lease_id)).fetchone()
        return None if row is None else json.loads(row[0])

    def fleet(self) -> dict:
        now = float(self.clock())
        with self._lock:
            rows = self._db.execute("SELECT report FROM gpu_admissions WHERE scope=? AND expires>?",
                                    (self.scope, now)).fetchall()
        models = {}
        for row in rows:
            model = json.loads(row[0])["hardware_model"]
            models[model] = models.get(model, 0) + 1
        return {"mode": self.mode, "as_of": now, "active_attested_devices": len(rows),
                "hardware_models": models, "scope": "validator_run", "execution_attested": False}


def registry_from_env(path, *, validator: str, run: str, env=None, clock=time.time):
    import os

    env = os.environ if env is None else env
    mode = env.get("RELIQUARY_RL_GPU_ADMISSION", "off").strip().lower()
    if mode not in MODES:
        raise ValueError("RELIQUARY_RL_GPU_ADMISSION must be off, observe or required")
    if mode == "off":
        return None
    try:
        return GpuRegistry(path, validator=validator, run=run, mode=mode, clock=clock)
    except ImportError as exc:
        raise RuntimeError("GPU admission needs the gpu-attestation dependency extra") from exc
