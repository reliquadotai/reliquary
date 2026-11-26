import hashlib
import json

import pytest

from reliquary.customer_env import env_id
from reliquary.customer_env.registry import subnet_key
from reliquary.customer_env.rl import CustomerEnvironment, load_rows
from reliquary.customer_env.runner import RewardInfrastructureError, ScoreResult
from reliquary.customer_env.scoring import CustomerScorer
from reliquary.environment.grader_client import GraderInfrastructureError

ENV = env_id("acct", "my-env", "a" * 64)
ROWS = [{"key": f"t{i}", "system": None, "prompt": f"{i}+{i}=", "prompt_sha256": f"{i}" * 64} for i in range(3)]
BODY = b"".join(json.dumps(r).encode() + b"\n" for r in ROWS)


def bucket(record=None, body=BODY):
    record = record or {"status": "qualified", "uses": ["rl", "batch", "eval"],
                        "prompt_set": {"sha256": hashlib.sha256(BODY).hexdigest()}}
    objects = {subnet_key(ENV, "record.json"): json.dumps(record).encode(), subnet_key(ENV, "prompts.jsonl"): body}
    return objects.get


class Client:
    def __init__(self, fail=None):
        self.fail, self.items = fail, []

    def score(self, items):
        self.items += items
        if self.fail == "host":
            raise RewardInfrastructureError("worker_hung")
        if self.fail == "item":
            return [ScoreResult(None, "grader_error: KeyError") for _ in items]
        return [ScoreResult(1.0 if i["answer"].strip() == "2" else 0.0) for i in items]

    def close(self):
        pass


def env_with(client):
    return CustomerEnvironment(str(ENV), rows=load_rows(str(ENV), read=bucket(), rows_dir=""),
                               scorer=CustomerScorer(lambda e: "img@sha256:" + "c" * 64, lambda image: client))


def test_problems_come_from_the_frozen_rows_and_rewards_from_the_reward_host():
    client = Client()
    env = env_with(client)
    assert len(env) == 3
    problem = env.get_problem(4)
    assert (problem["prompt"], problem["key"], problem["environment"]) == ("1+1=", "t1", str(ENV))
    assert env.compute_reward(problem, "<think>x</think>2") == 1.0
    assert env.compute_reward(problem, "3") == 0.0
    assert client.items[0] == {"key": "t1", "prompt_sha256": "1" * 64, "answer": "2"}


@pytest.mark.parametrize("fail", ["host", "item"])
def test_a_failure_is_infrastructure_never_a_zero(fail):
    env = env_with(Client(fail))
    with pytest.raises(GraderInfrastructureError):
        env.compute_reward(env.get_problem(0), "0")


@pytest.mark.parametrize("record, body, reason", [
    ({"status": "qualified", "uses": ["eval"], "prompt_set": {"sha256": "x"}}, BODY, "not qualified for RL"),
    ({"status": "disabled", "uses": ["rl"], "prompt_set": {"sha256": "x"}}, BODY, "not qualified for RL"),
    (None, BODY + b"\n{}", "missing or changed"),
])
def test_rows_must_match_a_version_qualified_for_rl(record, body, reason):
    with pytest.raises(RuntimeError, match=reason):
        load_rows(str(ENV), read=bucket(record, body), rows_dir="")


def test_the_registry_knows_a_customer_env_by_its_id():
    from reliquary.environment.registry import find_environment_spec, get_environment_spec

    spec = get_environment_spec(str(ENV))
    assert (spec.contract_version, spec.attainable_rewards, spec.validator_authoritative_reward) == (
        "reliquary/customer-env/v1", (), True)
    assert find_environment_spec("customer:bad") is None
    env = spec.create()
    assert isinstance(env, CustomerEnvironment) and env.name == str(ENV)
