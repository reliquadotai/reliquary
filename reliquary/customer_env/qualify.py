"""Qualifying one customer env version: cheapest step first, stop at the first failure with its reason.

1. the package's shape (``package.read_package``)            -> package_invalid
2. the image (``image.write_context`` + a builder)           -> build_failed
3. the worker starts in its sandbox and loads the taskset    -> start_failed
4. the prompts freeze into rows                              -> freeze_failed
5. every example scores as the customer says                 -> examples_failed
6. a second, fresh sandbox scores the examples identically   -> else nondeterministic: eval only
7. scoring time per item, for the price

No model runs and no GPU is used. A failure of ours (bucket, registry, sandbox capacity) is ``infrastructure``:
retried, never billed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable

from reliquary.customer_env import USES, parse_env_id
from reliquary.customer_env.image import BuildFailed
from reliquary.customer_env.package import PackageInvalid, read_package
from reliquary.customer_env.registry import Record, Registry, subnet_key
from reliquary.customer_env.runner import RewardClient, RewardInfrastructureError

logger = logging.getLogger(__name__)

SCORE_TOLERANCE = 1e-9
MAX_ROWS = 1_000_000


class _Refused(Exception):
    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(detail)
        self.reason, self.detail = reason, detail[:2000]


def rows_body(rows: list[dict]) -> bytes:
    return b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in rows)


class Qualifier:
    """``read_object(key) -> bytes``; ``builder.build_archive(data, env) -> image digest``; ``open_worker(image) ->
    RewardClient`` (a fresh sandbox each call); ``store.create_bytes(key, body)`` (the subnet bucket)."""

    def __init__(self, registry: Registry, *, read_object: Callable[[str], bytes], builder,
                 open_worker: Callable[[str], RewardClient], store, clock=time.monotonic) -> None:
        self._registry, self._read, self._builder = registry, read_object, builder
        self._open, self._store, self._clock = open_worker, store, clock

    def run(self, env_id: str) -> Record:
        record = self._registry.get(env_id)
        env = parse_env_id(env_id)
        try:
            return self._qualify(record, env)
        except _Refused as refused:
            return self._registry.update(env_id, status="refused", reason=refused.reason, detail=refused.detail)
        except Exception as exc:  # ours: retried, never billed
            logger.warning("customer env %s: infrastructure failure", env_id, exc_info=True)
            return self._registry.update(env_id, status="refused", reason="infrastructure",
                                         detail=f"{type(exc).__name__}: {exc}"[:500])

    def _qualify(self, record: Record, env) -> Record:
        data = self._read(record.object_key)
        if hashlib.sha256(data).hexdigest() != env.sha256:
            raise _Refused("package_invalid", "the stored archive is not the version's sha256")
        try:
            package = read_package(data)
        except PackageInvalid as exc:
            raise _Refused("package_invalid", str(exc)) from None
        self._registry.update(record.env_id, status="building", kind=package.kind)
        if package.kind == "agentic":
            raise _Refused("agentic_not_supported",
                           "agentic environments run on the order's gateways: not served yet")
        try:
            from reliquary.customer_env.image import check_lock

            check_lock(package)
        except PackageInvalid as exc:
            raise _Refused("package_invalid", str(exc)) from None
        try:
            image = self._builder.build_archive(data, env)
        except BuildFailed as exc:
            raise _Refused("build_failed", str(exc)) from None
        self._registry.update(record.env_id, status="qualifying", image_digest=image)

        first = self._open(image)
        try:
            started = self._clock()
            try:
                rows, refused = first.freeze_all(max_rows=MAX_ROWS)
            except RewardInfrastructureError as exc:
                if str(exc).startswith("spawn_failed"):
                    raise  # ours
                raise _Refused("start_failed" if "start_failed" in str(exc) else "freeze_failed", str(exc)) from None
            freeze_s = self._clock() - started
            if not rows:
                raise _Refused("freeze_failed", f"no task can be served: {refused[:5]}")
            # An example names its task by key, or by the task's own name (verifiers keys are content hashes).
            names = {r["name"]: r for r in rows if r.get("name")}
            by_key = {**({} if len(names) != len({r.get("name") for r in rows}) else names),
                      **{r["key"]: r for r in rows}}
            missing = sorted({e.key for e in package.examples} - set(by_key))
            if missing:
                raise _Refused("examples_failed", f"examples name tasks the taskset does not serve: {missing[:10]}")
            items = [{"key": by_key[e.key]["key"], "prompt_sha256": by_key[e.key]["prompt_sha256"], "answer": e.answer}
                     for e in package.examples]
            scored = self._score(first, items)
            # Timed on the warm worker: the first pass pays the worker's own start (imports, the taskset).
            started = self._clock()
            warm = self._score(first, items)
            score_s = self._clock() - started
        finally:
            first.close()
        report, wrong = [], []
        for example, result in zip(package.examples, scored):
            report.append({"key": example.key, "expected_score": example.expected_score,
                           "score": result.score, "error": result.error})
            if result.score is None or abs(result.score - example.expected_score) > SCORE_TOLERANCE:
                wrong.append(report[-1])
        if wrong:
            self._registry.update(record.env_id, examples=report)
            raise _Refused("examples_failed", json.dumps(wrong[:10]))

        second = self._open(image)
        try:
            again = self._score(second, items)
        finally:
            second.close()
        deterministic = [r.score for r in again] == [r.score for r in scored] == [r.score for r in warm]

        body = rows_body(rows)
        rows_sha = hashlib.sha256(body).hexdigest()
        self._store.create_bytes(subnet_key(env, "prompts.jsonl"), body)
        uses = list(USES) if deterministic else ["eval"]
        if any(r.get("system") for r in rows) and "rl" in uses:
            # An RL problem is one user turn: a system prompt would be silently dropped.
            uses.remove("rl")
        final = self._registry.update(
            record.env_id, status="qualified", reason=None if deterministic else "nondeterministic",
            detail=None if deterministic else "two fresh sandboxes scored the examples differently",
            uses=uses, examples=report, prompt_set={"count": len(rows), "sha256": rows_sha, "refused": len(refused)},
            cost={"qualification_seconds": round(freeze_s + 3 * score_s, 3),
                  "scoring_ms_per_item": round(1000 * score_s / max(1, len(items)), 3)})
        self._store.create_bytes(subnet_key(env, "record.json"),
                                 json.dumps(final.public(), sort_keys=True).encode())
        return final

    @staticmethod
    def _score(client: RewardClient, items: list[dict]):
        try:
            return client.score(items)
        except RewardInfrastructureError as exc:
            if str(exc).startswith("spawn_failed"):
                raise
            raise _Refused("examples_failed", f"the worker failed on the examples: {exc}") from None


__all__ = ["Qualifier", "rows_body"]
