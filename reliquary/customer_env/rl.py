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


def load_rows(env_id: str, *, read=_bucket_get, rows_dir: str | None = None) -> list[dict]:
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
        return {"prompt": row["prompt"], "ground_truth": "", "id": row["prompt_sha256"][:16],
                "environment": self.name, "key": row["key"], "prompt_sha256": row["prompt_sha256"]}

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


def customer_spec(name: str):
    from reliquary.environment.registry import EnvironmentSpec

    parse_env_id(name)
    return EnvironmentSpec(
        name=name, factory_path="reliquary.customer_env.rl:CustomerEnvironment",
        scorer_path="reliquary.customer_env.rl:score_customer_answers", validator_authoritative_reward=True,
        admission_resource_class="cpu", termination_policy="eos_or_cap", final_answer_policy="text",
        reward_lattice_policy="customer-continuous-v1", attainable_rewards=(), contract_version=CONTRACT_VERSION)


__all__ = ["CONTRACT_VERSION", "CustomerEnvironment", "customer_spec", "load_rows", "score_customer_answers"]
