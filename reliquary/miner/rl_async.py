"""The reference miner of the asynchronous RL protocol (docs/rl-async-design.md §3, P6/P7 of the implementation map).

It runs next to the window miner (``mine``), which stays for the live run. One process, five loops:

* **prompt state** (``GET /rl/prompts/state?since=``): a local table of every prompt the validator has scanned
  (``PromptPlanner``), read incrementally from a cursor. Blacklisted prompts are never asked for; a cooldown is
  dropped once the validator's latest checkpoint reaches its ``until_k``. The default order prefers prompts never
  scanned, then prompts whose history was in zone; ``priority`` overrides it.
* **quota** (``GET /rl/quota``): the measured concurrency the validator grants this hotkey. The miner never holds
  more open leases than ``min(quota, max_open)``, and asks for no more than it can start soon (``queue_groups``):
  every lease ends in a submission or an expiry, there is no release, and an expiry halves the quota.
* **leases** (``POST /rl/lease``, signed): batches of prompts at the checkpoint the miner holds. A refusal says only
  ``taken`` / ``cooling`` / ``blacklisted`` / ``quota`` / ``stale`` / ``full`` per prompt; each moves the planner or the
  scheduler (``_note_refusal``).
* **generation**: ``ROLLOUTS`` rollouts per lease, earliest expiry first, through the narrow ``GroupGenerator``
  interface (the draw itself, Gumbel and runner-keyed, is the profile's: ``VllmGroupGenerator`` sends a batch's
  leases x ``ROLLOUTS`` requests in ONE ``VLLMRolloutGenerator.generate_batch`` call, each request keyed by its own
  lease). Episode envs (multi-turn, signed sandboxes) are played by ``rl_async_episodes.EpisodeLeaseGenerator``
  (its ``agenerate``), through the window path's episode machinery bound to the lease's draw. New weights are swapped in between two generation batches; a continuous generator that
  ``swaps_running`` (single-turn vLLM, in-place push) takes them at once, its running rollouts resumed on them. A
  lease taken at k may be generated on a newer checkpoint, in part or whole, and the submission says which
  (``k_generated``): the draw is keyed by the lease, and the trainer judges every token on its own weights.
* **checkpoints** (``GET /rl/checkpoint`` + ``CheckpointSync``): deltas are downloaded and applied to a CPU copy of
  the weights off the generation path; the follower waits for the swap before patching that copy again.

Every generated group is submitted (``POST /rl/submit``, the wire of ``protocol/rl_submit.py`` signed with
``signatures.build_rl_submit_binding``), whatever it looks like, with retries until the lease's expiry; the validator
answers a resend of an already graded lease with its original verdict (the lease id is the idempotency key). Open
leases are journaled (``state_dir``) and, at start, read back from the validator (``GET /rl/lease/mine``), so a
restarted miner still generates and submits them before they expire, even without its journal.
"""
from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import random
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

ROLLOUTS = 16                       # design §4.4: all 16 rollout indices of a lease, each bound to its nonce
LEASE_BATCH = 8
QUEUE_GROUPS = 8                    # leases waiting for generation the miner may hold (beyond the running batch)
BATCH_GROUPS = 4                    # leases generated per batch; weights are swapped between batches
STATE_POLL_S = 15.0
QUOTA_POLL_S = 15.0
CHECKPOINT_POLL_S = 10.0
LEASE_POLL_S = 2.0
SUBMIT_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0)
OUTCOME_WAIT_S = 1800.0             # fallback wait for a held prompt's scan while the env's lease expiry is unknown
GENERATION_ATTEMPTS = 2
STREAM_STEP_S = 0.5                 # one continuous-batching call steps the engine this long (or to a finished group)
UNKNOWN_SAMPLE = 4                  # never-listed prompts drawn per wanted prompt

# The validator's words (``validator/rl_leases.py``), repeated here: the miner does not import the validator.
TAKEN, COOLING, BLACKLISTED, QUOTA, STALE = "taken", "cooling", "blacklisted", "quota", "stale"
# An episode task the validator's sandbox fleet cannot host now (retry later) or at all (another task meanwhile).
FLEET_UNAVAILABLE, TASK_UNSERVABLE = "fleet_unavailable", "task_unservable"
FLEET_RETRY_S, UNSERVABLE_HOLD_S = 60.0, 3600.0
FULL = "full"          # the validator's global cap (its trainer is behind): like quota, wait for the next poll
COOLDOWN, IN_ZONE = "cooldown", "in_zone"


# --------------------------------------------------------------------------------------------------- planner
@dataclass
class PromptRecord:
    """What the validator published about one prompt: its state (None: leasable, ``cooldown`` until
    ``until_k``, ``blacklisted``) and its scans as (checkpoint, rewards, verdict)."""
    state: str | None = None
    until_k: int | None = None
    scans: list[tuple[int, tuple[float, ...], str]] = field(default_factory=list)


def default_priority(record: PromptRecord | None) -> float | None:
    """Lower is asked first; None never. Never scanned: 0. Scanned: 2 minus its share of in-zone scans (an in-zone
    prompt that has cooled down is likely in zone again; one that was only too hard comes last)."""
    if record is None or not record.scans:
        return 0.0
    return 2.0 - sum(verdict == IN_ZONE for _, _, verdict in record.scans) / len(record.scans)


class PromptPlanner:
    """One environment's prompt table, fed by ``GET /rl/prompts/state`` pages, and the miner's prompt choice."""

    def __init__(self, environment: str, prompt_range: tuple[int, int], *,
                 priority: Callable[[PromptRecord | None], float | None] = default_priority,
                 rng: random.Random | None = None, clock: Callable[[], float] = time.time,
                 excluded: frozenset[int] = frozenset()) -> None:
        start, end = int(prompt_range[0]), int(prompt_range[1])
        if not 0 <= start < end:
            raise ValueError("prompt range must be a non-empty [start, end)")
        self.environment, self.range = environment, (start, end)
        self.excluded = frozenset(int(p) for p in excluded)    # rows of the range the validator refuses to lease
        self.priority, self._rng, self._clock = priority, rng or random.Random(), clock
        self.cursor = 0
        self.latest: int | None = None
        self.records: dict[int, PromptRecord] = {}
        self._waiting: dict[int, float] = {}        # prompt -> until: held, submitted or taken, no scan seen yet
        # The env's lease lifetime (the validator's ``lease_expiry_s``, from a lease answer or the quota): a prompt
        # a rival holds is free again, or scanned, within it. ``OUTCOME_WAIT_S`` until it is known.
        self.lease_expiry_s: float | None = None

    def apply(self, page: Mapping[str, Any]) -> bool:
        """Fold one page in; True when the validator has a next page ready (``more``)."""
        if page.get("latest") is not None:
            self.latest = max(int(page["latest"]), self.latest if self.latest is not None else -1)
        for scan in page.get("scans", ()):
            if scan.get("environment", self.environment) != self.environment or int(scan["seq"]) <= self.cursor:
                continue
            prompt = int(scan["prompt_idx"])
            self.records.setdefault(prompt, PromptRecord()).scans.append(
                (int(scan["checkpoint"]), tuple(float(r) for r in scan["rewards"]), str(scan["verdict"])))
            self._waiting.pop(prompt, None)
        for entry in page.get("prompts", ()):
            if entry.get("environment", self.environment) != self.environment:
                continue
            record = self.records.setdefault(int(entry["prompt_idx"]), PromptRecord())
            record.state = entry.get("state")
            record.until_k = None if entry.get("until_k") is None else int(entry["until_k"])
        self.cursor = max(self.cursor, int(page.get("cursor", self.cursor)))
        return bool(page.get("more"))

    def note_lease_expiry(self, seconds) -> None:
        if seconds is not None and float(seconds) > 0:
            self.lease_expiry_s = float(seconds)

    def wait_for_outcome(self, prompt: int, until: float | None = None) -> None:
        """Keep ``prompt`` out of the picks until a scan of it arrives, or until ``until`` (default: one lease
        lifetime from now, the longest a rival's lease of it can stay open): we hold it, we submitted it, or a
        rival holds it (``taken``) and may let it expire."""
        if until is None:
            until = self._clock() + (self.lease_expiry_s if self.lease_expiry_s is not None else OUTCOME_WAIT_S)
        prompt = int(prompt)
        self._waiting[prompt] = max(float(until), self._waiting.get(prompt, float("-inf")))

    def mark(self, prompt: int, state: str) -> None:
        """A lease refusal the table did not know yet (``cooling`` / ``blacklisted``)."""
        record = self.records.setdefault(int(prompt), PromptRecord())
        if state == BLACKLISTED:
            record.state, record.until_k = BLACKLISTED, None
        else:
            self.wait_for_outcome(prompt)

    def leasable(self, prompt: int) -> bool:
        if not self.range[0] <= prompt < self.range[1] or prompt in self.excluded:
            return False
        until = self._waiting.get(prompt)
        if until is not None:
            if self._clock() < until:
                return False
            del self._waiting[prompt]
        record = self.records.get(prompt)
        if record is None:
            return True
        if record.state == BLACKLISTED:
            return False
        if record.state == COOLDOWN:
            if record.until_k is not None and self.latest is not None and self.latest >= record.until_k:
                record.state, record.until_k = None, None       # the cooldown is over (the validator's rule)
            else:
                return False
        return True

    def pick(self, n: int) -> list[int]:
        """Up to ``n`` leasable prompts in priority order (ties at random): the known ones, plus a random sample of
        never-listed prompts of the range."""
        if n <= 0:
            return []
        candidates = {p for p in self.records if self.leasable(p)}
        start, end = self.range
        for _ in range(n * UNKNOWN_SAMPLE):
            prompt = self._rng.randrange(start, end)
            if prompt not in self.records and self.leasable(prompt):
                candidates.add(prompt)
        ranked = []
        for prompt in candidates:
            score = self.priority(self.records.get(prompt))
            if score is not None:
                ranked.append((float(score), self._rng.random(), prompt))
        return [prompt for _, _, prompt in sorted(ranked)[:n]]


# ------------------------------------------------------------------------------------------------ wire types
@dataclass(frozen=True)
class HeldLease:
    lease_id: str
    environment: str
    prompt_idx: int
    checkpoint: int
    nonce: str
    expires_at: float
    issued_at: float = 0.0

    @classmethod
    def from_wire(cls, body: Mapping[str, Any]) -> HeldLease:
        return cls(str(body["lease_id"]), str(body["environment"]), int(body["prompt_idx"]),
                   int(body["checkpoint"]), str(body["nonce"]), float(body["expires_at"]),
                   float(body.get("issued_at", 0.0)))

    def to_wire(self) -> dict:
        return {"lease_id": self.lease_id, "environment": self.environment, "prompt_idx": self.prompt_idx,
                "checkpoint": self.checkpoint, "nonce": self.nonce, "expires_at": self.expires_at,
                "issued_at": self.issued_at}


class RequestRefused(Exception):
    """A route answered with a refusal for the whole request. ``retryable``: 429/503/5xx or a transport error."""

    def __init__(self, reason: str, *, retryable: bool, retry_after: float | None = None) -> None:
        super().__init__(reason)
        self.reason, self.retryable, self.retry_after = reason, retryable, retry_after


@dataclass(frozen=True)
class EpisodeRollout:
    """One played episode of a lease (§4.10): its completion (everything after the validator's prompt render: model
    turns and tool observations), its model spans (completion-relative), the stop the validator admits and the
    gateway-signed transcript."""
    tokens: tuple[int, ...]
    spans: tuple[tuple[int, int], ...]
    stop: str
    transcript: Mapping[str, Any]

    def wire(self) -> dict:
        return {"spans": [[int(a), int(b)] for a, b in self.spans], "stop": str(self.stop),
                "transcript": dict(self.transcript)}


def submission_rollouts(rollouts: Sequence[Any]) -> list[dict]:
    """The ``rollouts`` field of ``POST /rl/submit`` (``protocol/rl_submit.py``): every index in order, completion
    token ids as ``encode_tokens``, and the BFT force span (completion-relative) where the generator reports one. A
    rollout is a token list, ``(tokens, force_span)``, or an ``EpisodeRollout`` (its ``episode`` field added)."""
    from reliquary.protocol.rl_submit import encode_tokens

    if len(rollouts) != ROLLOUTS:
        raise ValueError(f"a lease is submitted with all {ROLLOUTS} rollouts")
    out = []
    for index, rollout in enumerate(rollouts):
        if isinstance(rollout, EpisodeRollout):
            out.append({"index": index, "tokens": encode_tokens([int(t) for t in rollout.tokens]),
                        "force_span": None, "episode": rollout.wire()})
            continue
        tokens, span = (rollout if isinstance(rollout, tuple) else (rollout, None))
        out.append({"index": index, "tokens": encode_tokens([int(t) for t in tokens]),
                    "force_span": None if span is None else [int(span[0]), int(span[1])]})
    return out


def encode_submission(lease: HeldLease, rollouts: Sequence[Any], k_generated: int | None = None) -> dict:
    """The unsigned part of a submission: the lease id, its rollouts and, informational only (the lease's k is the
    generation checkpoint the validator uses; it ignores this field), the checkpoint the group was generated on.
    ``HttpRlTransport.submit`` adds the hotkey, the time and the signature. Tokens only: no logprob, no proof."""
    payload = {"lease_id": lease.lease_id, "rollouts": submission_rollouts(rollouts)}
    if k_generated is not None:
        payload["k_generated"] = int(k_generated)
    return payload


def submission_digest(payload: Mapping[str, Any]) -> bytes:
    """What the signature covers (``protocol.rl_submit.rollouts_digest`` of the rollouts as sent)."""
    from reliquary.protocol.rl_submit import rollouts_digest

    rollouts = sorted(payload["rollouts"], key=lambda r: int(r["index"]))
    return rollouts_digest([(int(r["index"]), r["tokens"], None if r.get("force_span") is None
                             else tuple(r["force_span"]), r.get("episode")) for r in rollouts])


class RlTransport(Protocol):
    async def prompt_state(self, since: int, environment: str) -> dict: ...
    async def quota(self) -> dict: ...
    async def lease(self, environment: str, checkpoint: int, prompt_ids: Sequence[int]) -> dict: ...
    async def submit(self, payload: dict) -> dict: ...
    async def my_leases(self) -> dict: ...
    async def checkpoint(self) -> dict | None: ...


class GroupGenerator(Protocol):
    """The narrow generation interface. ``generate`` returns, per lease, ``ROLLOUTS`` completions (token ids, stop
    token kept; or ``(token ids, force span)`` where the env forces one) drawn with the lease's nonce
    (``draw_context``) on the weights loaded last; ``load_weights`` replaces those weights
    (called between two ``generate`` calls only)."""
    def generate(self, leases: Sequence[HeldLease]) -> list[list[list[int]]]: ...
    def load_weights(self, state: Mapping[str, Any], version: int) -> None: ...


# ------------------------------------------------------------------------------------------- HTTP transport
class HttpRlTransport:
    """The validator's routes over ``httpx.AsyncClient``; lease and submit bodies are signed by the hotkey for the
    validator and the route."""

    def __init__(self, client, *, base_url: str, hotkey: str, validator_hotkey: str,
                 sign_binding: Callable[[bytes], str], prefix: str = "/rl",
                 clock: Callable[[], float] = time.time) -> None:
        self._client, self._base = client, base_url.rstrip("/")
        self.hotkey, self.validator_hotkey, self._prefix = hotkey, validator_hotkey, prefix
        self._sign, self._clock = sign_binding, clock

    async def _call(self, method: str, path: str, **kwargs) -> Any:
        import httpx

        try:
            response = await self._client.request(method, self._base + path, **kwargs)
        except httpx.HTTPError as exc:
            raise RequestRefused(type(exc).__name__, retryable=True) from exc
        if response.status_code == 200:
            return response.json()
        try:
            reason = str(response.json().get("reason") or response.status_code)
        except ValueError:
            reason = str(response.status_code)
        retry_after = response.headers.get("Retry-After")
        raise RequestRefused(reason, retryable=response.status_code in (429, 503) or response.status_code >= 500,
                             retry_after=float(retry_after) if retry_after and retry_after.isdigit() else None)

    async def prompt_state(self, since: int, environment: str) -> dict:
        return await self._call("GET", f"{self._prefix}/prompts/state",
                                params={"since": str(int(since)), "environment": environment})

    async def quota(self) -> dict:
        return await self._call("GET", f"{self._prefix}/quota", params={"hotkey": self.hotkey})

    async def lease(self, environment: str, checkpoint: int, prompt_ids: Sequence[int]) -> dict:
        from reliquary.protocol.signatures import build_rl_lease_binding

        path = f"{self._prefix}/lease"
        at, ids = int(self._clock()), [int(p) for p in prompt_ids]
        body = {"miner_hotkey": self.hotkey, "at": at, "environment": environment, "checkpoint": int(checkpoint),
                "prompt_ids": ids}
        body["signature"] = self._sign(build_rl_lease_binding(
            hotkey=self.hotkey, at=at, environment=environment, checkpoint=int(checkpoint), prompt_ids=ids,
            validator_hotkey=self.validator_hotkey, path=path))
        return await self._call("POST", path, json=body)

    async def submit(self, payload: dict) -> dict:
        from reliquary.protocol.signatures import build_rl_submit_binding

        path = f"{self._prefix}/submit"
        at = int(self._clock())                         # a fresh signature per attempt; the payload is the same
        body = {"miner_hotkey": self.hotkey, "at": at, **payload}
        body["signature"] = self._sign(build_rl_submit_binding(
            hotkey=self.hotkey, at=at, lease_id=str(payload["lease_id"]), rollouts_digest=submission_digest(payload),
            validator_hotkey=self.validator_hotkey, path=path))
        from reliquary.protocol.rl_submit import GZIP, HOTKEY_HEADER, gzip_body

        headers = {HOTKEY_HEADER: self.hotkey}          # the validator checks it before it reads the body
        if any(r.get("episode") is not None for r in payload.get("rollouts", ())):
            # 16 signed transcripts: sent gzipped (``protocol/rl_submit``: the validator inflates within its cap).
            raw = json.dumps(body, separators=(",", ":")).encode("utf-8")
            content = await asyncio.to_thread(gzip_body, raw)
            return await self._call("POST", path, content=content,
                                    headers={**headers, "Content-Type": "application/json", "Content-Encoding": GZIP})
        return await self._call("POST", path, json=body, headers=headers)

    async def episodes(self) -> dict:
        """``GET /rl/episodes``: the validator's episode envs (their blocks), ``{}`` when it serves none."""
        try:
            answer = await self._call("GET", f"{self._prefix}/episodes")
        except RequestRefused as exc:
            if exc.reason == "404":
                return {}
            raise
        return dict(answer.get("environments") or {})

    async def my_leases(self) -> dict:
        from reliquary.protocol.signatures import build_rl_lease_mine_binding

        path = f"{self._prefix}/lease/mine"
        at = int(self._clock())
        signature = self._sign(build_rl_lease_mine_binding(hotkey=self.hotkey, at=at,
                                                           validator_hotkey=self.validator_hotkey, path=path))
        return await self._call("GET", path, params={"hotkey": self.hotkey, "at": str(at), "signature": signature})

    async def forfeit(self, lease_id: str) -> dict:
        """``POST /rl/lease/forfeit``: give up an open lease (charged as its expiry, its quota slot freed now)."""
        from reliquary.protocol.signatures import build_rl_lease_forfeit_binding

        path = f"{self._prefix}/lease/forfeit"
        at = int(self._clock())
        signature = self._sign(build_rl_lease_forfeit_binding(hotkey=self.hotkey, at=at, lease_id=lease_id,
                                                              validator_hotkey=self.validator_hotkey, path=path))
        return await self._call("POST", path, json={"miner_hotkey": self.hotkey, "at": at, "lease_id": lease_id,
                                                    "signature": signature})

    async def checkpoint(self) -> dict | None:
        try:
            return await self._call("GET", f"{self._prefix}/checkpoint")
        except RequestRefused as exc:
            if exc.retryable:
                return None
            raise

    async def observations(self, after: int) -> dict | None:
        """The scan log's signed head and presigned URLs (``GET /rl/observations``); None while unpublished."""
        try:
            return await self._call("GET", f"{self._prefix}/observations", params={"after": str(int(after))})
        except RequestRefused as exc:
            if exc.retryable:
                return None
            raise

    async def fetch(self, url: str) -> bytes:
        response = await self._client.get(url)
        response.raise_for_status()
        return response.content


class DryRunTransport:
    """An in-process validator for ``--dry-run`` and tests: grants every asked prompt within its quota (unless told
    otherwise), derives nonces from (env, prompt, k), records submissions and serves no checkpoint."""

    def __init__(self, *, quota: int = 8, expiry_s: float = 600.0, latest: int = 0,
                 clock: Callable[[], float] = time.time) -> None:
        self.quota_value, self.expiry_s, self.latest, self._clock = quota, expiry_s, latest, clock
        self.open: dict[str, HeldLease] = {}
        self.granted: dict[str, HeldLease] = {}          # every lease handed out, by id
        self.lost_answers = 0                           # the next submits are recorded but their answer is lost
        self.submissions: list[dict] = []
        self.verdicts: dict[str, dict] = {}              # lease id -> the answer given (resends get it again)
        self.lease_calls: list[tuple[str, int, list[int]]] = []
        self.refuse: dict[int, str] = {}                # prompt -> refusal reason
        self.pages: list[dict] = []                     # prompt-state pages served in order, then empty ones
        self.submit_failures: list[RequestRefused] = [] # raised by the next submits, in order
        self.checkpoints: list[dict | None] = []
        self.state_calls: list[int] = []

    async def prompt_state(self, since: int, environment: str) -> dict:
        self.state_calls.append(int(since))
        if self.pages:
            return self.pages.pop(0)
        return {"latest": self.latest, "cursor": since, "more": False, "scans": [], "prompts": []}

    async def quota(self) -> dict:
        return {"quota": self.quota_value, "in_use": len(self.open), "latest": self.latest,
                "lease_expiry_s": {}}

    async def my_leases(self) -> dict:
        return {"leases": [lease.to_wire() for lease in self.open.values()], "latest": self.latest,
                "lease_expiry_s": {}}

    async def forfeit(self, lease_id: str) -> dict:
        self.open.pop(lease_id, None)
        return {"lease_id": lease_id, "state": "expired", "quota": self.quota_value}

    async def lease(self, environment: str, checkpoint: int, prompt_ids: Sequence[int]) -> dict:
        self.lease_calls.append((environment, int(checkpoint), [int(p) for p in prompt_ids]))
        now, leases, refused = self._clock(), [], []
        for prompt in prompt_ids:
            if prompt in self.refuse:
                refused.append({"prompt_idx": prompt, "reason": self.refuse[prompt]})
            elif len(self.open) >= self.quota_value:
                refused.append({"prompt_idx": prompt, "reason": QUOTA})
            else:
                nonce = hashlib.sha256(f"{environment}/{prompt}/{checkpoint}".encode()).hexdigest()
                lease = HeldLease(f"dry-{len(self.lease_calls)}-{prompt}", environment, int(prompt),
                                  int(checkpoint), nonce, now + self.expiry_s, now)
                self.open[lease.lease_id] = self.granted[lease.lease_id] = lease
                leases.append(lease.to_wire())
        return {"leases": leases, "refused": refused, "latest": self.latest, "quota": self.quota_value,
                "in_use": len(self.open), "lease_expiry_s": self.expiry_s}

    async def submit(self, payload: dict) -> dict:
        if self.submit_failures:
            raise self.submit_failures.pop(0)
        lease_id = payload["lease_id"]
        if lease_id in self.verdicts:                   # the validator's idempotent resend
            return self.verdicts[lease_id]
        if lease_id not in self.open:
            raise RequestRefused("lease_unknown", retryable=False)
        del self.open[lease_id]
        self.submissions.append(payload)
        self.verdicts[lease_id] = {"lease_id": lease_id, "verdict": "received", "rewards": []}
        if self.lost_answers:
            self.lost_answers -= 1
            raise RequestRefused("ReadTimeout", retryable=True)
        return self.verdicts[lease_id]

    async def checkpoint(self) -> dict | None:
        return self.checkpoints.pop(0) if self.checkpoints else None


# ------------------------------------------------------------------------------------------------ the miner
@dataclass(frozen=True)
class AsyncMinerConfig:
    environments: tuple[str, ...]
    lease_batch: int = LEASE_BATCH
    max_open: int = 64                  # the miner's own ceiling on open leases (its GPU); the quota binds first
    queue_groups: int = QUEUE_GROUPS
    batch_groups: int = BATCH_GROUPS
    state_poll_s: float = STATE_POLL_S
    quota_poll_s: float = QUOTA_POLL_S
    checkpoint_poll_s: float = CHECKPOINT_POLL_S
    lease_poll_s: float = LEASE_POLL_S
    submit_backoff_s: tuple[float, ...] = SUBMIT_BACKOFF_S
    state_dir: str | None = None        # where open leases are journaled (restart safety)

    def __post_init__(self) -> None:
        if not self.environments:
            raise ValueError("at least one environment is needed")
        if min(self.lease_batch, self.max_open, self.queue_groups, self.batch_groups) < 1:
            raise ValueError("lease batch, max open, queue and batch sizes must be positive")
        if self.batch_groups > self.queue_groups:
            # A batch is taken from the queue: it would silently stay at queue_groups.
            raise ValueError("batch_groups must not exceed queue_groups (--queue-groups)")


class AsyncMiner:
    def __init__(self, config: AsyncMinerConfig, *, transport: RlTransport, generator: GroupGenerator,
                 planners: Mapping[str, PromptPlanner], checkpoint_sync=None, loaded_version: int | None = None,
                 state: dict | None = None, rng: random.Random | None = None,
                 env_weights: Callable[[], Mapping[str, float] | None] | None = None,
                 clock: Callable[[], float] = time.time) -> None:
        missing = set(config.environments) - set(planners)
        if missing:
            raise ValueError(f"no planner for {sorted(missing)}")
        self.config, self.transport, self.generator = config, transport, generator
        self.planners, self._sync = dict(planners), checkpoint_sync
        self._rng, self._clock = rng or random.Random(), clock
        self._env_weights = env_weights
        self.loaded_version = loaded_version            # what the generator holds; leases are asked at it
        self._state, self._state_version = state, loaded_version   # the CPU copy the follower patches
        self.open: dict[str, HeldLease] = {}            # granted, not yet submitted nor given up
        self.queue: list[HeldLease] = []                # open and waiting for generation
        self.running: dict[str, HeldLease] = {}         # in a continuous-batching generator, by lease id
        self.quota: int | None = None
        self.stats = {"leased": 0, "submitted": 0, "submit_failed": 0, "expired_unsent": 0, "recovered": 0,
                      "refused": {}, "verdicts": {}}
        self._attempts: dict[str, int] = {}
        self._submits: set[asyncio.Task] = set()
        self._ready = asyncio.Event()                   # the follower holds new weights for the generator
        self._wake_generation = asyncio.Event()
        self._wake_lease = asyncio.Event()
        self._wake_checkpoint = asyncio.Event()
        self._stale = False                             # a ``stale`` refusal: lease nothing until newer weights
        self._env_turn = 0
        self._load_journal()

    # --- prompt state and quota ------------------------------------------------------------------------
    async def bootstrap_state(self) -> int:
        """At start, fold the run's published scan log (``protocol/rl_scan_log``: signed segments in the
        bucket, one download each) into the planners; ``/rl/prompts/state`` then continues from its last seq.
        Returns the scans read (0 when the transport or the validator has no log)."""
        from reliquary.protocol.rl_scan_log import read_scan_log, scans_as_page

        observations = getattr(self.transport, "observations", None)
        validator_hotkey = getattr(self.transport, "validator_hotkey", None)
        if observations is None or not validator_hotkey:
            return 0
        events, _ = await read_scan_log(observations, self.transport.fetch, validator_hotkey)
        if events:
            page = scans_as_page(events)
            for planner in self.planners.values():
                if planner.cursor < page["cursor"]:
                    planner.apply(page)
            logger.info("prompt table bootstrapped from %d published scans (seq %d)", len(events), page["cursor"])
        return len(events)

    async def refresh_state(self) -> None:
        for name, planner in self.planners.items():
            more = True
            while more:
                more = planner.apply(await self.transport.prompt_state(planner.cursor, name))

    def _note_expiries(self, answer: Mapping[str, Any], environment: str | None = None) -> None:
        """The env lease lifetimes a validator answer carries: a map (quota, own leases) or one env's (a lease)."""
        expiry = answer.get("lease_expiry_s")
        if isinstance(expiry, Mapping):
            for name, seconds in expiry.items():
                if name in self.planners:
                    self.planners[name].note_lease_expiry(seconds)
        elif expiry is not None and environment in self.planners:
            self.planners[environment].note_lease_expiry(expiry)

    async def refresh_quota(self) -> None:
        answer = await self.transport.quota()
        self.quota = int(answer["quota"])
        self._note_expiries(answer)
        logger.info("rl miner: k=%s quota=%d open=%d queued=%d %s", self.loaded_version, self.quota,
                    len(self.open), len(self.queue), self.stats)

    def lease_room(self) -> int:
        """How many leases to ask for now: within ``min(quota, max_open)`` open, and no more waiting for generation
        than ``queue_groups`` (a lease waiting long risks its expiry), at most ``lease_batch`` per request."""
        if self.loaded_version is None or self._stale or self.quota is None:
            return 0
        cap = min(self.quota, self.config.max_open) - len(self.open)
        return max(0, min(cap, self.config.queue_groups - len(self.queue), self.config.lease_batch))

    def _next_environment(self) -> str:
        names = list(self.config.environments)
        weights = self._env_weights() if self._env_weights is not None else None
        if weights and any(float(weights.get(n, 0.0)) > 0 for n in names):
            return self._rng.choices(names, [max(0.0, float(weights.get(n, 0.0))) for n in names])[0]
        self._env_turn = (self._env_turn + 1) % len(names)
        return names[self._env_turn]

    # --- leases --------------------------------------------------------------------------------------
    async def lease_more(self) -> int:
        """One lease request if there is room; the number of leases granted."""
        room = self.lease_room()
        if room <= 0:
            return 0
        environment = self._next_environment()
        planner = self.planners[environment]
        prompts = planner.pick(room)
        if not prompts:
            return 0
        k = int(self.loaded_version)
        try:
            answer = await self.transport.lease(environment, k, prompts)
        except RequestRefused as exc:
            logger.warning("lease request refused: %s", exc.reason)
            if exc.retryable:
                wait = self.config.lease_poll_s if exc.retry_after is None else exc.retry_after
            else:
                wait = 30.0                           # a refusal the miner must fix (registration, signature)
            await asyncio.sleep(min(wait, 60.0))
            return 0
        if answer.get("quota") is not None:
            self.quota = int(answer["quota"])
        self._note_expiries(answer, environment)
        granted = [HeldLease.from_wire(item) for item in answer.get("leases", ())]
        for lease in granted:
            if lease.lease_id in self.open:
                continue
            self.open[lease.lease_id] = lease
            self.queue.append(lease)
            planner.wait_for_outcome(lease.prompt_idx, lease.expires_at)
        for item in answer.get("refused", ()):
            self._note_refusal(planner, int(item["prompt_idx"]), str(item["reason"]))
        self.stats["leased"] += len(granted)
        if granted:
            self._journal()
            self._wake_generation.set()
            self._wake_lease.set()                    # there may be room for another request
        return len(granted)

    def _note_refusal(self, planner: PromptPlanner, prompt: int, reason: str) -> None:
        counts = self.stats["refused"]
        counts[reason] = counts.get(reason, 0) + 1
        if reason == TAKEN:
            planner.wait_for_outcome(prompt)          # a rival holds (prompt, k): its scan, or its expiry, comes
        elif reason in (COOLING, BLACKLISTED):
            planner.mark(prompt, reason)              # our table is behind: the next state poll catches up
        elif reason in (QUOTA, FULL):
            self.quota = len(self.open)               # the validator counts fewer, or is full: wait for its next quota
        elif reason in (FLEET_UNAVAILABLE, TASK_UNSERVABLE):
            planner.wait_for_outcome(prompt, self._clock() + (FLEET_RETRY_S if reason == FLEET_UNAVAILABLE
                                                              else UNSERVABLE_HOLD_S))
        elif reason == STALE:
            self._stale = True                        # our k is retired or not adopted yet: sync first
            self._wake_checkpoint.set()

    # --- generation and weights ----------------------------------------------------------------------
    def _swaps_running(self) -> bool:
        """The generator takes new weights while groups run (a rollout may then span two checkpoints: its draw is
        keyed by the lease, and the trainer judges every token against its own latest weights)."""
        return bool(getattr(self.generator, "swaps_running", False)) and not getattr(self, "_no_running_swap", False)

    async def swap_if_ready(self) -> bool:
        """Push the follower's weights into the generator: called between generation batches only."""
        if not self._ready.is_set():
            return False
        version = int(self._state_version)
        # A rollout is generated on one checkpoint: the swap waited for the running leases (the longest episode)
        # with no new lease started; that wait is the swap's cost in generation time.
        since = getattr(self, "_ready_since", None)
        waited = 0.0 if since is None else max(0.0, self._clock() - since)
        await asyncio.to_thread(self.generator.load_weights, self._state, version)
        self.stats["swaps"] = self.stats.get("swaps", 0) + 1
        self.stats["swap_wait_s"] = round(self.stats.get("swap_wait_s", 0.0) + waited, 1)
        self.stats["swap_wait_max_s"] = round(max(self.stats.get("swap_wait_max_s", 0.0), waited), 1)
        logger.info("generator now on checkpoint %d (was %s) after %.0f s waiting for running leases", version,
                    self.loaded_version, waited)
        self.loaded_version, self._stale = version, False
        self._ready.clear()
        self._wake_lease.set()
        return True

    def _take_batch(self, n: int | None = None) -> list[HeldLease]:
        now = self._clock()
        self.queue.sort(key=lambda lease: lease.expires_at)        # earliest expiry first
        batch: list[HeldLease] = []
        while self.queue and len(batch) < (self.config.batch_groups if n is None else n):
            lease = self.queue.pop(0)
            if lease.expires_at <= now:
                logger.warning("lease %s expired before generation", lease.lease_id[:12])
                self.stats["expired_unsent"] += 1
                self._close(lease)
                continue
            batch.append(lease)
        return batch

    def _retry(self, leases: Sequence[HeldLease]) -> None:
        """Leases whose generation failed: back in the queue, or given up after ``GENERATION_ATTEMPTS``."""
        for lease in leases:
            tries = self._attempts[lease.lease_id] = self._attempts.get(lease.lease_id, 0) + 1
            if tries < GENERATION_ATTEMPTS:
                self.queue.append(lease)
            else:
                logger.error("lease %s given up after %d generation failures", lease.lease_id[:12], tries)
                self._close(lease)
                self._forfeit(lease)

    def _forfeit(self, lease: HeldLease) -> None:
        """Tell the validator a given-up lease will not be submitted: charged as its expiry, its quota slot freed
        now instead of at its end (an episode lease's end is over an hour away)."""
        call = getattr(self.transport, "forfeit", None)
        if call is None:
            return

        async def send() -> None:
            try:
                await call(lease.lease_id)
                self.stats["forfeited"] = self.stats.get("forfeited", 0) + 1
            except Exception as exc:                     # the lease then expires as before
                logger.warning("forfeit of lease %s failed (%s)", lease.lease_id[:12], exc)

        task = asyncio.create_task(send())
        self._submits.add(task)
        task.add_done_callback(self._submits.discard)

    def _hand_over(self, lease: HeldLease, rollouts, version: int) -> None:
        task = asyncio.create_task(self.submit(lease, rollouts, version))
        self._submits.add(task)
        task.add_done_callback(self._submits.discard)

    async def _stream_once(self) -> int:
        """Continuous batching (a generator with ``advance``): keep ``batch_groups`` leases running, start a queued
        one as soon as a group completes and submit that group at once, so no lease waits for a batch's longest
        rollout. A generator that ``swaps_running`` takes new weights at once, its running groups resumed on them;
        otherwise new weights wait until the running groups are done and no lease starts while a swap is pending. Returns the groups completed, or 1 while the engine is busy."""
        if self._ready.is_set() and (not self.running or self._swaps_running()):
            try:
                await self.swap_if_ready()
            except Exception:
                logger.exception("weight swap with %d leases running failed; draining before the next try",
                                 len(self.running))
                self._no_running_swap = True
        if self.loaded_version is None:
            return 0
        new = [] if self._ready.is_set() and not self._swaps_running() else self._take_batch(
            self.config.batch_groups - len(self.running))
        if not new and not self.running:
            return 0
        version = int(self.loaded_version)
        self.running.update((lease.lease_id, lease) for lease in new)
        if new:
            self._wake_lease.set()                                 # the queue has room again
        try:
            aadvance = getattr(self.generator, "aadvance", None)          # episodes: played on the event loop
            done = (await aadvance(new, STREAM_STEP_S) if callable(aadvance)
                    else await asyncio.to_thread(self.generator.advance, new, STREAM_STEP_S))
        except Exception:
            logger.exception("continuous generation of %d running leases failed", len(self.running))
            failed = list(self.running.values())
            self.running.clear()
            with contextlib.suppress(Exception):
                if callable(aadvance):
                    self.generator.abort()                 # its tasks live on this loop
                else:
                    await asyncio.to_thread(self.generator.abort)
            self._retry(failed)
            return 0
        for lease_id, rollouts in done:
            lease = self.running.pop(lease_id, None)
            if lease is None:
                continue
            if isinstance(rollouts, BaseException):          # that lease alone failed, for good (e.g. episodes)
                logger.error("lease %s given up: %s", lease_id[:12], rollouts)
                self._close(lease)
                self._forfeit(lease)
                continue
            if len(rollouts) != ROLLOUTS:
                logger.error("lease %s came back with %d rollouts", lease_id[:12], len(rollouts))
                self._retry([lease])
                continue
            self._hand_over(lease, rollouts, version)
        return max(len(done), 1)

    async def generate_once(self) -> int:
        """Swap weights if new ones are ready, then generate one batch and hand each group to its submission. A
        generator with ``advance`` is driven continuously instead (``_stream_once``)."""
        if callable(getattr(self.generator, "advance", None)) or callable(getattr(self.generator, "aadvance", None)):
            return await self._stream_once()
        await self.swap_if_ready()
        if self.loaded_version is None:
            return 0
        batch = self._take_batch()
        if not batch:
            return 0
        version = int(self.loaded_version)
        self._wake_lease.set()                                     # the queue has room again
        try:
            agenerate = getattr(self.generator, "agenerate", None)       # episodes: played on the event loop
            groups = (await agenerate(batch) if agenerate is not None
                      else await asyncio.to_thread(self.generator.generate, batch))
            if len(groups) != len(batch) or any(not isinstance(g, BaseException) and len(g) != ROLLOUTS
                                                for g in groups):
                raise ValueError(f"the generator must return {ROLLOUTS} rollouts per lease")
        except Exception:
            logger.exception("generation of %d leases failed", len(batch))
            self._retry(batch)
            return 0
        done = 0
        for lease, rollouts in zip(batch, groups):
            if isinstance(rollouts, BaseException):          # that lease alone failed, for good (e.g. episodes)
                logger.error("lease %s given up: %s", lease.lease_id[:12], rollouts)
                self._close(lease)
                self._forfeit(lease)
                continue
            self._hand_over(lease, rollouts, version)
            done += 1
        return done

    # --- submission ----------------------------------------------------------------------------------
    async def submit(self, lease: HeldLease, rollouts: Sequence[Sequence[int]], k_generated: int) -> bool:
        """Submit one group until it is received, refused for good, or its lease has expired. The payload is built
        once: a resend after a lost answer is the same submission (the lease id is the idempotency key)."""
        payload = encode_submission(lease, rollouts, k_generated)
        attempt = 0
        try:
            while True:
                try:
                    answer = await self.transport.submit(payload)
                    self.stats["submitted"] += 1
                    verdict = str((answer or {}).get("verdict", "unknown"))
                    self.stats["verdicts"][verdict] = self.stats["verdicts"].get(verdict, 0) + 1
                    return True
                except RequestRefused as exc:
                    if not exc.retryable:
                        logger.warning("submission of lease %s refused: %s", lease.lease_id[:12], exc.reason)
                        self.stats["submit_failed"] += 1
                        return False
                    delay = exc.retry_after
                backoff = self.config.submit_backoff_s
                delay = delay if delay is not None else backoff[min(attempt, len(backoff) - 1)]
                attempt += 1
                left = lease.expires_at - self._clock()
                if left <= 0:
                    logger.warning("lease %s expired while its submission was retried", lease.lease_id[:12])
                    self.stats["submit_failed"] += 1
                    return False
                await asyncio.sleep(min(delay, max(0.0, left - 0.5)))
        finally:
            self._close(lease)
            self.planners[lease.environment].wait_for_outcome(lease.prompt_idx)   # its scan is on the way

    def _close(self, lease: HeldLease) -> None:
        self.open.pop(lease.lease_id, None)
        self._attempts.pop(lease.lease_id, None)
        self._journal()
        self._wake_lease.set()

    async def drain(self) -> None:
        """Wait for every submission in flight."""
        while self._submits:
            await asyncio.gather(*list(self._submits), return_exceptions=True)

    # --- checkpoints ---------------------------------------------------------------------------------
    async def follow_checkpoint_once(self) -> int | None:
        """One poll of ``GET /rl/checkpoint``: when the index is ahead of our CPU copy, bring the copy there (off the
        event loop) and offer it to the generator; returns the new version. The copy is not touched again until the
        generator has taken it (``swap_if_ready``)."""
        if self._sync is None or self._ready.is_set():
            return None
        response = await self.transport.checkpoint()
        if not response:
            return None
        latest = int(response["index"]["latest"]["version"])
        if self._state_version is not None and latest <= self._state_version:
            self._stale = False                       # nothing newer to fetch: let the next lease request retry
            return None
        state, version, plan = await asyncio.to_thread(self._sync.sync, response, self._state, self._state_version)
        logger.info("checkpoint %d rebuilt (%s, %d bytes)", version,
                    "anchor" if plan.anchor is not None else f"{len(plan.deltas)} deltas", plan.bytes)
        self._state, self._state_version = state, version
        self._ready_since = self._clock()
        self._ready.set()
        self._wake_generation.set()
        return version

    # --- journal -------------------------------------------------------------------------------------
    def _journal_path(self) -> Path | None:
        return None if not self.config.state_dir else Path(self.config.state_dir) / "rl-open-leases.json"

    def _journal(self) -> None:
        path = self._journal_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps([lease.to_wire() for lease in self.open.values()]))
        os.replace(tmp, path)

    def _load_journal(self) -> None:
        path = self._journal_path()
        if path is None or not path.exists():
            return
        now = self._clock()
        try:
            leases = [HeldLease.from_wire(item) for item in json.loads(path.read_text())]
        except (ValueError, KeyError, TypeError):
            logger.warning("unreadable lease journal %s ignored", path)
            return
        for lease in leases:
            self._adopt(lease, now)
        if self.queue:
            logger.info("%d open leases resumed from the journal", len(self.queue))

    def _adopt(self, lease: HeldLease, now: float) -> bool:
        """Take an open lease found after a restart (journal or validator) back into the queue."""
        if lease.lease_id in self.open or lease.expires_at <= now or lease.environment not in self.planners:
            return False
        self.open[lease.lease_id] = lease
        self.queue.append(lease)
        self.planners[lease.environment].wait_for_outcome(lease.prompt_idx, lease.expires_at)
        return True

    async def recover_leases(self) -> int:
        """Ask the validator for this hotkey's open leases (``GET /rl/lease/mine``) and queue those the journal did
        not have (a lost or foreign state dir): they are generated and submitted like any other. Returns how many."""
        answer = await self.transport.my_leases()
        self._note_expiries(answer)
        now = self._clock()
        added = sum(self._adopt(HeldLease.from_wire(item), now) for item in answer.get("leases", ()))
        if added:
            self.stats["recovered"] += added
            logger.info("%d open leases recovered from the validator", added)
            self._journal()
            self._wake_generation.set()
        return added

    # --- loops ---------------------------------------------------------------------------------------
    async def _every(self, step: Callable[[], Any], period: float, wake: asyncio.Event | None,
                     stop: asyncio.Event, name: str) -> None:
        while not stop.is_set():
            try:
                await step()
            except RequestRefused as exc:
                logger.warning("%s: %s", name, exc.reason)
            except Exception:
                logger.exception("%s failed", name)
            waits = [asyncio.ensure_future(stop.wait())]
            if wake is not None:
                waits.append(asyncio.ensure_future(wake.wait()))
            done, pending = await asyncio.wait(waits, timeout=period, return_when=asyncio.FIRST_COMPLETED)
            for future in pending:
                future.cancel()
            if wake is not None:
                wake.clear()

    async def _generation_loop(self, stop: asyncio.Event) -> None:
        try:
            while not stop.is_set():
                if await self.generate_once():
                    continue
                self._wake_generation.clear()
                if (self.queue and self.loaded_version is not None) or self._ready.is_set():
                    continue
                with contextlib.suppress(TimeoutError):
                    await asyncio.wait_for(self._wake_generation.wait(), timeout=self.config.lease_poll_s)
        except Exception:
            # A generator that cannot load weights must not leave the other loops leasing for it: the leases
            # would expire and halve the quota. Stop the miner (run() re-raises it).
            logger.exception("generation stopped; the miner stops")
            stop.set()
            raise

    async def run(self, stop: asyncio.Event | None = None) -> None:
        stop = stop or asyncio.Event()
        config = self.config
        try:
            await self.recover_leases()
        except RequestRefused as exc:
            logger.warning("open leases not recovered from the validator: %s", exc.reason)
        try:
            await self.bootstrap_state()
        except Exception as exc:          # the incremental route covers everything the log would have
            logger.warning("scan log not read (%s); paging /rl/prompts/state from the start", exc)
        tasks = [
            asyncio.create_task(self._every(self.refresh_state, config.state_poll_s, None, stop, "prompt state")),
            asyncio.create_task(self._every(self.refresh_quota, config.quota_poll_s, None, stop, "quota")),
            asyncio.create_task(self._every(self.follow_checkpoint_once, config.checkpoint_poll_s,
                                            self._wake_checkpoint, stop, "checkpoint")),
            asyncio.create_task(self._every(self.lease_more, config.lease_poll_s, self._wake_lease, stop, "lease")),
            asyncio.create_task(self._generation_loop(stop)),
        ]
        try:
            await stop.wait()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            # Leases already generated are still submitted: every lease ends in a submission when it can.
            await self.drain()
        generation = tasks[-1]
        if not generation.cancelled() and generation.exception() is not None:
            raise generation.exception()


# ------------------------------------------------------------------------------------ vLLM generation adapter
def draw_context(lease: HeldLease) -> dict:
    """What the existing generator's draw is keyed with for one lease: ``protocol.rl_draw.lease_draw_context``, the
    keying the validator's audits and the trainer recompute from the lease (``SubmittedGroup.draw_uniforms``)."""
    from reliquary.protocol.rl_draw import lease_draw_context

    return lease_draw_context(lease.nonce, lease.prompt_idx, lease.checkpoint)


def write_hf_dir(state: Mapping[str, Any], base_model_dir: str | Path, dest: str | Path) -> Path:
    """A loadable HF directory: the base model's config and tokenizer files plus ``state`` as safetensors shards."""
    from reliquary.shared.ckpt_delta import write_anchor_shards

    base, dest = Path(base_model_dir), Path(dest)
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    for item in base.iterdir():
        if item.is_file() and not item.name.endswith((".safetensors", ".safetensors.index.json", ".bin")):
            shutil.copy2(item, dest / item.name)
    from safetensors import safe_open

    # transformers loads several shards only through their index (vLLM globs them): the trainer's resume and the
    # Hugging Face backups load this directory with transformers.
    weight_map = {}
    for path in write_anchor_shards(state, dest):
        with safe_open(str(path), framework="pt") as shard:
            weight_map.update(dict.fromkeys(shard.keys(), path.name))
    (dest / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": weight_map}))
    return dest


def _load_weights_from_files(worker, paths: list[str]) -> bool:
    """Runs inside each vLLM worker (``LLM.collective_rpc``): the model's own ``load_weights`` over the shards."""
    from safetensors.torch import load_file

    model = worker.model_runner.model
    for path in paths:
        model.load_weights(weights=load_file(path).items())
    return True


class VllmGroupGenerator:
    """``GroupGenerator`` over the existing ``VLLMRolloutGenerator`` (its draw is the active profile's).

    Weights: ``swap="push"`` writes the state to shards in ``scratch_dir`` (``/dev/shm`` by default) and has every
    worker load them through the model's ``load_weights`` (``LLM.collective_rpc``), then resets the prefix cache;
    a failure falls back to ``"reload"``, which writes an HF directory and rebuilds the engine (a full model load).

    ``base_model_dir`` holds the HF config and tokenizer files. It may be the directory ``CheckpointSync`` keeps the
    anchors' model files in: it is read on the first ``load_weights``, after the first checkpoint arrived.
    """

    def __init__(self, *, base_model_dir: str, work_dir: str, swap: str = "push", scratch_dir: str | None = None,
                 engine_kwargs: Mapping[str, Any] | None = None) -> None:
        if swap not in ("push", "reload"):
            raise ValueError("swap must be push or reload")
        self.base, self.work, self.swap = Path(base_model_dir), Path(work_dir), swap
        # One per process: miners sharing a host swap at the same moment and would clear each other's shards.
        self.scratch = Path(scratch_dir or ("/dev/shm" if Path("/dev/shm").is_dir() else work_dir)) / f"rl-push-{os.getpid()}"
        self.tokenizer = None
        self.eos_ids: list[int] = []
        if (self.base / "config.json").is_file():
            self._load_model_files()
        self._engine_kwargs = dict(engine_kwargs or {})
        self._engine = None
        self._dir: Path | None = None
        self._envs: dict[str, Any] = {}
        if swap == "push":
            # collective_rpc ships a function to the workers: allowed before the engine starts them.
            os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    def _load_model_files(self) -> None:
        from transformers import AutoTokenizer, GenerationConfig

        from reliquary.shared.modeling import resolve_eos_token_ids

        self.tokenizer = AutoTokenizer.from_pretrained(str(self.base))
        try:
            generation = GenerationConfig.from_pretrained(str(self.base))
        except OSError:
            generation = None
        model_meta = None if generation is None else type("Meta", (), {"generation_config": generation})()
        self.eos_ids = sorted(resolve_eos_token_ids(model_meta, self.tokenizer))

    def _prompt(self, environment: str, prompt_idx: int) -> list[int]:
        from reliquary.constants import thinking_for_environment
        from reliquary.environment import load_environment
        from reliquary.protocol.tokens import encode_prompt

        env = self._envs.get(environment)
        if env is None:
            env = self._envs[environment] = load_environment(environment)
        problem = env.get_problem(prompt_idx)
        return encode_prompt(self.tokenizer, problem["prompt"], thinking=thinking_for_environment(environment))

    def generate(self, leases: Sequence[HeldLease]) -> list[list[list[int]]]:
        from reliquary.constants import max_new_tokens_for_environment

        from reliquary.protocol.rl_draw import lease_extra_args

        if self._engine is None:
            raise RuntimeError("no weights loaded")
        # One engine call for the whole batch (leases x ROLLOUTS requests): each request carries its own lease's
        # draw (nonce, prompt, k, rollout index), read by request id in the runner-keyed processor.
        return self._engine.generate_batch(
            [(self._prompt(lease.environment, lease.prompt_idx),
              [lease_extra_args(lease.nonce, lease.prompt_idx, lease.checkpoint, r) for r in range(ROLLOUTS)],
              max_new_tokens_for_environment(lease.environment)) for lease in leases],
            eos_ids=self.eos_ids)

    def advance(self, leases: Sequence[HeldLease], budget_s: float) -> list[tuple[str, list[list[int]]]]:
        """Continuous batching: start ``leases`` in the running engine, step it for about ``budget_s``, return the
        groups that completed as ``(lease id, rollouts)``."""
        from reliquary.constants import max_new_tokens_for_environment

        from reliquary.protocol.rl_draw import lease_extra_args

        if self._engine is None:
            raise RuntimeError("no weights loaded")
        for lease in leases:
            self._engine.add_group(
                lease.lease_id, self._prompt(lease.environment, lease.prompt_idx),
                [lease_extra_args(lease.nonce, lease.prompt_idx, lease.checkpoint, r) for r in range(ROLLOUTS)],
                max_new_tokens_for_environment(lease.environment), eos_ids=self.eos_ids)
        return self._engine.step_groups(budget_s)

    def abort(self) -> None:
        if self._engine is not None:
            self._engine.abort_groups()

    def _reload(self, state: Mapping[str, Any], version: int) -> None:
        from reliquary.miner.vllm_generation import VLLMRolloutGenerator

        new_dir = write_hf_dir(state, self.base, self.work / f"weights-{version:08d}")
        if self._engine is None:
            self._engine = VLLMRolloutGenerator(str(new_dir), **self._engine_kwargs)
        else:
            self._engine.reload(str(new_dir))
        if self._dir is not None and self._dir != new_dir:
            shutil.rmtree(self._dir, ignore_errors=True)
        self._dir = new_dir

    def _push(self, state: Mapping[str, Any]) -> None:
        from reliquary.shared.ckpt_delta import write_anchor_shards

        shutil.rmtree(self.scratch, ignore_errors=True)
        try:
            paths = [str(p) for p in write_anchor_shards(state, self.scratch)]
            llm = self._engine._llm
            llm.collective_rpc(_load_weights_from_files, args=(paths,))
            # Running requests are preempted and resumed on the new weights (their KV recomputed, their draw
            # placed by the runner's own token count): a swap never waits for the longest rollout.
            llm.reset_prefix_cache(reset_running_requests=True)
        finally:
            shutil.rmtree(self.scratch, ignore_errors=True)

    @property
    def swaps_running(self) -> bool:
        """New weights go in while groups are generating (``swap="push"``): the miner never drains for a swap."""
        return self.swap == "push"

    def load_weights(self, state: Mapping[str, Any], version: int) -> None:
        if self.tokenizer is None:
            if not (self.base / "config.json").is_file():
                raise RuntimeError(f"no model files (config, tokenizer) in {self.base}")
            self._load_model_files()
        if self._engine is None or self.swap == "reload":
            self._reload(state, version)
            return
        try:
            self._push(state)
        except Exception:
            if self._engine.has_running_groups():
                raise                         # a rebuild would drop the running groups: the miner drains first
            logger.exception("in-place weight push failed; rebuilding the engine on checkpoint %d", version)
            self._reload(state, version)


class FakeGroupGenerator:
    """A CPU generator for ``--dry-run --fake-generator`` and tests: completions derived from the nonce."""

    def __init__(self, length: int = 8) -> None:
        self.length = length
        self.calls: list[tuple[str, ...]] = []
        self.loads: list[int] = []
        self.version: int | None = None
        self.versions_used: list[int | None] = []

    def generate(self, leases: Sequence[HeldLease]) -> list[list[list[int]]]:
        self.calls.append(tuple(lease.lease_id for lease in leases))
        self.versions_used.append(self.version)
        return [[[int(lease.nonce[(r + t) % len(lease.nonce)], 16) for t in range(self.length)]
                 for r in range(ROLLOUTS)] for lease in leases]

    def load_weights(self, state, version: int) -> None:
        self.loads.append(int(version))
        self.version = int(version)


class FakeStreamGenerator(FakeGroupGenerator):
    """``FakeGroupGenerator`` with continuous batching: a lease completes ``steps[prompt_idx]`` (default 1)
    ``advance`` calls after it started."""

    def __init__(self, length: int = 8, steps: Mapping[int, int] | None = None) -> None:
        super().__init__(length)
        self.steps = dict(steps or {})
        self.running: dict[str, tuple[HeldLease, int]] = {}
        self.started: list[str] = []
        self.fail_next = False

    def advance(self, leases: Sequence[HeldLease], budget_s: float) -> list[tuple[str, list[list[int]]]]:
        if self.fail_next:
            self.fail_next = False
            raise RuntimeError("engine step failed")
        for lease in leases:
            self.started.append(lease.lease_id)
            self.running[lease.lease_id] = (lease, self.steps.get(lease.prompt_idx, 1))
        done = []
        for lease_id, (lease, left) in list(self.running.items()):
            if left <= 1:
                del self.running[lease_id]
                done.append((lease_id, self.generate([lease])[0]))
            else:
                self.running[lease_id] = (lease, left - 1)
        return done

    def abort(self) -> None:
        self.running.clear()


# ------------------------------------------------------------------------------------------------ entry point
async def run_async_miner(config: AsyncMinerConfig, *, validator_url: str, validator_hotkey: str, wallet=None,
                          base_model_dir: str | None = None, work_dir: str, initial_dir: str | None = None,
                          initial_version: int | None = None, swap: str = "push", dry_run: bool = False,
                          fake_generator: bool = False, engine_kwargs: Mapping[str, Any] | None = None,
                          prompt_ranges: Mapping[str, tuple[int, int]] | None = None,
                          episode_options: Mapping[str, Any] | None = None) -> None:
    """Build the transport, generator, planners and checkpoint follower, and run until interrupted."""
    import httpx

    from reliquary.miner.ckpt_sync import CheckpointSync
    from reliquary.miner.rl_pay_client import PayWeights
    from reliquary.shared.ckpt_delta import load_anchor_shards

    model_files = Path(work_dir) / "model-files"     # the anchors' HF config and tokenizer (CheckpointSync)
    # The config and tokenizer files: --base-model, else the initial checkpoint's directory, else the ones the
    # anchors carry (synced on the first checkpoint).
    if not base_model_dir and initial_dir is not None and (Path(initial_dir) / "config.json").is_file():
        base_model_dir = initial_dir
    if initial_dir is not None and initial_version is None:
        raise ValueError("--initial-version is required with --initial-checkpoint")
    if dry_run and not fake_generator and initial_dir is None:
        raise ValueError("a dry run generates on --initial-checkpoint (or --fake-generator): it fetches nothing")
    async with contextlib.AsyncExitStack() as stack:
        if dry_run:
            transport: RlTransport = DryRunTransport()
            sync, weights, blocks = None, None, {}
        else:
            client = await stack.enter_async_context(httpx.AsyncClient(timeout=60.0))
            hotkey = wallet.hotkey.ss58_address
            transport = HttpRlTransport(client, base_url=validator_url, hotkey=hotkey,
                                        validator_hotkey=validator_hotkey,
                                        sign_binding=lambda binding: wallet.hotkey.sign(binding).hex())
            sync = CheckpointSync(validator_hotkey=validator_hotkey, work_dir=Path(work_dir) / "ckpt-download",
                                  model_dir=model_files)
            weights = PayWeights(validator_url).weights
            # The validator's signed-episode envs (their blocks: policy, catalogue); played by the episode stack. The
            # answer adds each env's lease lifetime, which is no part of the block.
            blocks = {name: {k: v for k, v in block.items() if k != "lease_expiry_s"}
                      for name, block in (await transport.episodes()).items() if name in config.environments}
        generator = await asyncio.to_thread(
            miner_generator, config, blocks=blocks, fake_generator=fake_generator, dry_run=dry_run,
            base_model_dir=base_model_dir or str(model_files), work_dir=work_dir, swap=swap,
            engine_kwargs=engine_kwargs, validator_url=validator_url, validator_hotkey=validator_hotkey,
            wallet=wallet, episode_options=episode_options)
        closer = getattr(getattr(getattr(generator, "episodes", None), "host", None), "close", None)
        if closer is not None:
            stack.push_async_callback(closer)
        ranges = dict(prompt_ranges or {})
        excluded: dict[str, frozenset[int]] = {}
        sources = getattr(getattr(generator, "episodes", None), "sources", None) or {}
        for name in config.environments:
            if name in ranges:
                continue
            if name in sources:
                ranges[name] = (0, len(sources[name]))         # an episode env: its whole taskset (the validator's)
                continue
            from reliquary.environment import load_environment
            from reliquary.protocol.rl_prompt_ranges import rl_excluded_prompts, rl_prompt_range

            env = load_environment(name)
            ranges[name] = rl_prompt_range(name, len(env))
            excluded[name] = rl_excluded_prompts(name, env)     # the validator's: it refuses a lease of these
        planners = {name: PromptPlanner(name, ranges[name], excluded=excluded.get(name, frozenset()))
                    for name in config.environments}
        state, loaded = None, None
        if initial_dir is not None:
            state = load_anchor_shards(sorted(Path(initial_dir).glob("*.safetensors")))
            await asyncio.to_thread(generator.load_weights, state, int(initial_version))
            loaded = int(initial_version)
        elif dry_run:
            generator.load_weights({}, 0)
            loaded = 0
        miner = AsyncMiner(config, transport=transport, generator=generator, planners=planners,
                           checkpoint_sync=sync, loaded_version=loaded, state=state, env_weights=weights)
        await miner.run()


def miner_generator(config: AsyncMinerConfig, *, blocks: Mapping[str, Any], fake_generator: bool = False,
                    dry_run: bool = False, base_model_dir: str, work_dir: str, swap: str = "push",
                    engine_kwargs: Mapping[str, Any] | None = None, validator_url: str = "",
                    validator_hotkey: str = "", wallet=None, episode_options: Mapping[str, Any] | None = None,
                    build_episodes=None, build_single=None) -> GroupGenerator:
    """The miner's generator: the vLLM group generator for its single-turn envs, the episode stack
    (``rl_async_episodes.build_episode_generator``) for the validator's signed-episode envs (``blocks``) and the
    replayed ones (``rl_suite_episodes``: registry episode envs, no block), both under
    one ``MixedGenerator`` when the miner mines both (they share the GPU: split ``--gpu-memory-utilization``).
    Blocking (an episode env loads its package and its taskset)."""
    if fake_generator:
        return FakeGroupGenerator()
    if dry_run and not (Path(base_model_dir) / "config.json").is_file():
        raise ValueError("a dry run needs --base-model or a full --initial-checkpoint (config and tokenizer)")
    from reliquary.validator.rl_suite_episodes import is_replayed_env

    # Replayed episode envs (deterministic tool worlds): played in this process on the episode engine.
    suites = [name for name in config.environments if name not in blocks and is_replayed_env(name)]
    singles = [name for name in config.environments if name not in blocks and name not in suites]
    single = None
    if singles:
        single = (build_single or VllmGroupGenerator)(base_model_dir=base_model_dir, work_dir=work_dir, swap=swap,
                                                      engine_kwargs=engine_kwargs)
        if not blocks and not suites:
            return single
    from reliquary.miner.rl_async_episodes import MixedGenerator, build_episode_generator

    options = dict(episode_options or {})
    if suites:
        options["suites"] = tuple(suites)
    episodes = (build_episodes or build_episode_generator)(
        dict(blocks), validator_url=validator_url, validator_hotkey=validator_hotkey,
        hotkey=wallet.hotkey.ss58_address, sign_binding=lambda binding: wallet.hotkey.sign(binding).hex(),
        base_model_dir=base_model_dir, work_dir=work_dir, **options)
    logger.info("episode envs %s played on the forced episode engine%s", sorted([*blocks, *suites]),
                f"; single-turn envs {singles} on vLLM" if singles else "")
    return MixedGenerator(episodes=episodes, single=single)


__all__ = [
    "AsyncMiner", "AsyncMinerConfig", "DryRunTransport", "FakeGroupGenerator", "FakeStreamGenerator", "GroupGenerator",
    "HeldLease", "HttpRlTransport", "PromptPlanner", "PromptRecord", "RequestRefused", "RlTransport",
    "VllmGroupGenerator", "default_priority", "draw_context", "encode_submission", "miner_generator", "run_async_miner", "submission_digest",
    "submission_rollouts", "write_hf_dir",
]
