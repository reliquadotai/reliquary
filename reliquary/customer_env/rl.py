"""A qualified customer env version as an RL (or batch) environment, by its id as the environment name.

``get_problem`` serves the frozen rows; ``compute_reward`` asks the reward host. The validator's process
never imports the customer's code. A failure of the reward host, or of the customer's grader on an item,
raises ``GraderInfrastructureError``: the completion is not scored and its miner is not faulted.

Rows are read from ``RELIQUARY_CUSTOMER_ENV_ROWS_DIR`` (``<dir>/<account>/<name>/<sha256>/prompts.jsonl``, as the
controller mirrors them to its miners) or else from the subnet bucket, and checked against the version's
qualified record.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path

from reliquary.customer_env import is_customer_env, parse_env_id
from reliquary.customer_env.registry import subnet_key

CONTRACT_VERSION = "reliquary/customer-env/v1"
_SCORER: dict[str, object] = {}
_SCORER_LOCK = threading.Lock()


def _bucket_get(key: str) -> bytes | None:
    from reliquary.eval.storage import SubnetEvalStore

    return asyncio.run(SubnetEvalStore().get_bytes(key))


ROWS_URL_ENV = "RELIQUARY_CUSTOMER_ENV_ROWS_URL"
PROMPT_SET_HEADER = "X-Prompt-Set-Sha256"


def _rows_from_controller(base_url: str, env_id: str, *, get=None) -> list[dict]:
    """A miner's rows: its task's controller serves them (the miner has no bucket access), hashed as qualified."""
    if get is None:
        import httpx

        def get(url, params):
            return httpx.get(url, params=params, timeout=120.0)

    response = get(f"{base_url.rstrip('/')}/customer-envs/rows", {"env": env_id})
    if response.status_code != 200:
        raise RuntimeError(f"{env_id}: the controller serves no rows ({response.status_code})")
    body = response.content
    if hashlib.sha256(body).hexdigest() != response.headers.get(PROMPT_SET_HEADER):
        raise RuntimeError(f"{env_id}: the served rows do not hash to the qualified prompt set")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


def task_environments() -> list[str]:
    """The customer envs this controller's contract names (its process profile)."""
    from reliquary.protocol.profiles import ACTIVE_PROTOCOL_PROFILE

    return sorted(name for name in ACTIVE_PROTOCOL_PROFILE.environments if is_customer_env(name))


_ROWS_CACHE: dict[str, tuple[bytes, str]] = {}


def rows_body(env_id: str, *, read=_bucket_get) -> tuple[bytes, str]:
    """The qualified rows' bytes and sha256, checked against the version's record (cached)."""
    if env_id not in _ROWS_CACHE:
        rows = load_rows(env_id, read=read, rows_dir="")
        body = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in rows)
        _ROWS_CACHE[env_id] = (body, hashlib.sha256(body).hexdigest())
    return _ROWS_CACHE[env_id]


def mount_rows_routes(router) -> None:
    """``GET /rl/customer-envs``: the task's customer envs; ``GET /rl/customer-envs/rows?env=``: one's rows."""
    from fastapi.responses import JSONResponse, Response

    @router.get("/rl/customer-envs")
    async def customer_envs():
        return {"environments": task_environments()}

    @router.get("/rl/customer-envs/rows")
    async def customer_env_rows(env: str = ""):
        if env not in task_environments():
            return JSONResponse(status_code=404, content={"reason": "unknown_environment"})
        try:
            body, sha = await asyncio.to_thread(rows_body, env)
        except RuntimeError as exc:
            return JSONResponse(status_code=503, content={"reason": str(exc)[:300]})
        return Response(content=body, media_type="application/x-ndjson",
                        headers={PROMPT_SET_HEADER: sha, "Cache-Control": "max-age=300"})


def load_rows(env_id: str, *, read=_bucket_get, rows_dir: str | None = None, rows_url: str | None = None,
              get=None) -> list[dict]:
    rows_url = rows_url if rows_url is not None else os.environ.get(ROWS_URL_ENV)
    if rows_url:
        return _rows_from_controller(rows_url, env_id, get=get)
    env = parse_env_id(env_id)
    record_body = read(subnet_key(env, "record.json"))
    if record_body is None:
        raise RuntimeError(f"{env_id}: no qualified record in the subnet bucket")
    record = json.loads(record_body)
    if record.get("status") != "qualified" or "rl" not in record.get("uses", ()):
        raise RuntimeError(f"{env_id} is not qualified for RL")
    rows_dir = rows_dir if rows_dir is not None else os.environ.get("RELIQUARY_CUSTOMER_ENV_ROWS_DIR")
    if rows_dir:
        body = (Path(rows_dir) / env.account / env.name / env.sha256 / "prompts.jsonl").read_bytes()
    else:
        body = read(subnet_key(env, "prompts.jsonl"))
    if body is None or hashlib.sha256(body).hexdigest() != record["prompt_set"]["sha256"]:
        raise RuntimeError(f"{env_id}: its frozen rows are missing or changed")
    return [json.loads(line) for line in body.splitlines() if line.strip()]


class CustomerEnvironment:
    def __init__(self, name: str, *, rows: list[dict] | None = None, scorer=None) -> None:
        if not is_customer_env(name):
            raise ValueError(f"{name!r} is not a customer env id")
        self.name = name
        self._rows = rows
        self._scorer = scorer
        self._lock = threading.Lock()

    def _all(self) -> list[dict]:
        with self._lock:
            if self._rows is None:
                self._rows = load_rows(self.name)
            return self._rows

    def __len__(self) -> int:
        return len(self._all())

    def get_problem(self, index: int) -> dict:
        rows = self._all()
        row = rows[int(index) % len(rows)]
        problem = {"prompt": row["prompt"], "ground_truth": "", "id": row["prompt_sha256"][:16],
                   "environment": self.name, "key": row["key"], "prompt_sha256": row["prompt_sha256"]}
        if row.get("system"):
            problem["system"] = row["system"]
        return problem

    def compute_reward(self, problem: dict, completion: str) -> float:
        return score_customer_answers(problem, [completion], scorer=self._scorer)[0]


def _default_scorer():
    with _SCORER_LOCK:
        if "default" not in _SCORER:
            from reliquary.customer_env.scoring import from_environment

            _SCORER["default"] = from_environment()
        return _SCORER["default"]


def score_customer_answers(problem: dict, completions: list[str], reward_materials=None, *,
                           scorer=None) -> list[float]:
    """The reward host's scores for one problem's completions, or ``GraderInfrastructureError``."""
    from reliquary.customer_env.runner import RewardInfrastructureError
    from reliquary.environment.grader_client import GraderInfrastructureError
    from reliquary.eval.grading import answer_text

    scorer = scorer or _default_scorer()
    if scorer is None:
        raise GraderInfrastructureError("customer_env_unavailable: no reward host configured")
    ref = {"key": problem["key"], "prompt_sha256": problem["prompt_sha256"]}
    answers = [answer_text("text", c) for c in completions]
    try:
        results = scorer.score_many(problem["environment"], [ref] * len(answers), answers)
    except (RewardInfrastructureError, LookupError) as exc:
        raise GraderInfrastructureError(f"customer_env: {exc}") from exc
    failed = [r.error for r in results if r.error is not None]
    if failed:
        raise GraderInfrastructureError(f"customer_grader: {failed[0]}")
    return [float(r.score) for r in results]


def _spec_class():
    from reliquary.environment.registry import EnvironmentSpec

    class CustomerSpec(EnvironmentSpec):
        """One spec per version, all served by ``CustomerEnvironment`` built with the version id."""

        __slots__ = ()

        def create(self, split: str = "train") -> CustomerEnvironment:
            if split != "train":
                raise ValueError(f"{self.name} has no split {split}: its rows are one index space")
            return CustomerEnvironment(self.name)

    return CustomerSpec


_CUSTOMER_SPEC = None


def customer_spec(name: str):
    global _CUSTOMER_SPEC
    parse_env_id(name)
    if _CUSTOMER_SPEC is None:
        _CUSTOMER_SPEC = _spec_class()
    return _CUSTOMER_SPEC(
        name=name, factory_path="reliquary.customer_env.rl:CustomerEnvironment",
        scorer_path="reliquary.customer_env.rl:score_customer_answers", validator_authoritative_reward=True,
        admission_resource_class="cpu", termination_policy="eos_or_cap", final_answer_policy="text",
        reward_lattice_policy="customer-continuous-v1", attainable_rewards=(), contract_version=CONTRACT_VERSION)


def register_customer_envs(names) -> list[str]:
    """Make each well-formed customer env id among ``names`` known to the environment registry (its spec is its
    id), where a task's contract or lease names it. ``environment/registry.py`` is left as it is: its source is
    part of the grader fingerprint the sealed catalog sets carry (``eval.protocol.grader_sha256``), so the spec
    goes into its run-time table from here."""
    from reliquary.environment import registry

    known = []
    for name in names:
        name = str(name)
        if not is_customer_env(name):
            continue
        if registry.find_environment_spec(name) is None:
            try:
                spec = customer_spec(name)
            except ValueError:
                continue  # not a well-formed version id: nothing to know it by
            with registry._CATALOGUE_LOCK:
                registry._CATALOGUE_SPECS.setdefault(name, spec)
        known.append(name)
    return known


__all__ = ["CONTRACT_VERSION", "PROMPT_SET_HEADER", "ROWS_URL_ENV", "CustomerEnvironment", "customer_spec",
           "load_rows", "register_customer_envs", "mount_rows_routes", "rows_body", "score_customer_answers", "task_environments"]
