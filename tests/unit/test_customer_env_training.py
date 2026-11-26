"""A training on a customer env: its task carries a contract with the env's settings; controller, leases and
miners follow it; the miner reads the rows from the task's controller."""
import hashlib
import json

import pytest

from reliquary.admin.training import CreateTraining, order_digest, training_entry, training_refusal
from reliquary.customer_env import env_id
from reliquary.environment.abi import canonical_sha256
from reliquary.protocol.profiles import PROFILES, entry_profile, profile_from_contract

PROFILE = "qwen35-4b-reliquary-suite-v10-gumbel-dev1"
ENV = str(env_id("acct", "my-env", "a" * 64))


def body(**over):
    order = {"training_id": "t1", "profile_id": PROFILE, "account": "acct",
             "environments": [{"name": "reliquary_dapo_math_v1", "share_bps": 6000},
                              {"name": ENV, "share_bps": 4000, "max_new_tokens": 2048, "thinking": False}],
             "limits": {"max_steps": 5, "deadline_seconds": 3600}, "max_price_usd_per_mtoken": 1.0,
             "batch_groups": 8}
    return CreateTraining(**{**order, **over})


def ok(env, account):
    if account != "acct":
        raise ValueError("customer_env_not_yours")


def prices():
    from reliquary.validator.corpus_periods import usd_price_params

    return usd_price_params(usd_per_mtoken_input=0., usd_per_mtoken_output=1., alpha_usd=1., tao_usd=1.,
                            miner_alpha_per_period=1., set_at=1000.)


@pytest.mark.parametrize("over, why", [
    ({"account": None}, "names its account"),
    ({"environments": [{"name": ENV, "share_bps": 10000}]}, "needs max_new_tokens"),
    ({"account": "other"}, "not_yours"),
    ({"environments": [{"name": "reliquary_dapo_math_v1", "share_bps": 10000, "max_new_tokens": 99}]},
     "keeps its profile's settings"),
])
def test_a_customer_env_order_is_refused_with_its_reason(over, why):
    assert why in training_refusal(body(**over), customer_env=ok)


def test_without_customer_envs_on_this_admin_it_is_refused():
    assert "unavailable" in training_refusal(body(), customer_env=None)


def test_the_entry_carries_the_profile_contract_with_the_customer_env_added():
    entry = training_entry(body(), task_prefix="order-", created_at=1000, price_params=prices(), customer_env=ok,
                           model_architecture="Qwen3_5ForConditionalGeneration")
    assert entry.profile_id == PROFILE and entry.contract["profile_id"] == PROFILE
    assert entry.profile_sha256 == canonical_sha256(entry.contract)
    assert set(entry.contract["environments"]) == {*PROFILES[PROFILE].environments, ENV}
    assert entry.contract["environments"][ENV] == {"max_new_tokens": 2048, "answer_format": None, "bft": None,
                                                    "thinking": False}
    assert entry.env_split[ENV] == 0.4 and entry.env_split["reliquary_dapo_math_v1"] == 0.6
    profile = entry_profile(entry)
    assert profile.environments[ENV].max_new_tokens == 2048 and profile.environments[ENV].thinking is False
    assert profile_from_contract(entry.contract).model_architecture == "Qwen3_5ForConditionalGeneration"
    # The profile's own envs are the profile's, byte for byte.
    template = PROFILES[PROFILE].to_generation_contract()["environments"]
    assert all(entry.contract["environments"][n] == template[n] for n in template)


def test_a_customer_entry_needs_the_model_architecture():
    with pytest.raises(ValueError, match="architecture"):
        training_entry(body(), task_prefix="order-", created_at=1000, price_params=prices(), customer_env=ok)


def test_orders_without_the_new_fields_keep_their_digest():
    plain = {"training_id": "t1", "profile_id": PROFILE,
             "environments": [{"name": "reliquary_dapo_math_v1", "share_bps": 6000},
                              {"name": "envscaler_tools_v2", "share_bps": 4000}],
             "limits": {"max_steps": 200, "deadline_seconds": 86400}, "max_price_usd_per_mtoken": 2.0,
             "batch_groups": 128}
    assert order_digest(CreateTraining(**plain)) == "33a31835ba45c4f5eaa0904cd701b016eabd6377351a608f00edd735ce688bb0"
    assert order_digest(body()) != order_digest(body(account="acct2"))


def test_the_controller_starts_on_a_contract_naming_a_customer_env():
    from reliquary.validator.task_config import resolve_task_config

    entry = training_entry(body(), task_prefix="order-", created_at=1000, price_params=prices(), customer_env=ok,
                           model_architecture="Qwen3_5ForConditionalGeneration")
    config = resolve_task_config({entry.task_id: entry}, entry.task_id, profile_id=PROFILE,
                                 generation_contract=entry.contract)
    assert set(config.env_caps) >= {ENV}


def test_a_lease_on_a_customer_env_carries_its_settings_and_the_miner_reads_them(monkeypatch):
    from reliquary.miner.rl_async import HeldLease
    from reliquary.validator.rl_leases import Lease

    monkeypatch.setattr("reliquary.constants.max_new_tokens_for_environment", lambda env: 2048)
    monkeypatch.setattr("reliquary.constants.thinking_for_environment", lambda env: False)
    fields = {f: None for f in Lease.__dataclass_fields__}
    fields.update(lease_id="L1", environment=ENV, prompt_idx=3, checkpoint=2, nonce="ab" * 32, issued_at=1.0,
                  expires_at=2.0, state="open")
    public = Lease(**{k: v for k, v in fields.items()}).to_public()
    assert public["env_settings"] == {"max_new_tokens": 2048, "thinking": False}
    held = HeldLease.from_wire(public)
    monkeypatch.setattr("reliquary.constants.max_new_tokens_for_environment", lambda env: 99)
    assert (held.max_new_tokens(), held.thinking()) == (2048, False)       # the lease's, not the miner's profile
    assert HeldLease.from_wire(held.to_wire()) == held
    plain = {k: v for k, v in public.items() if k != "env_settings"}
    assert HeldLease.from_wire(plain).env_settings is None
    with pytest.raises(ValueError):
        HeldLease.from_wire({**public, "env_settings": {"max_new_tokens": "x", "thinking": False}})


def test_a_lease_on_a_profile_env_carries_nothing_new():
    from reliquary.validator.rl_leases import Lease

    fields = {f: None for f in Lease.__dataclass_fields__}
    fields.update(lease_id="L1", environment="reliquary_dapo_math_v1", prompt_idx=3, checkpoint=2, nonce="ab" * 32,
                  issued_at=1.0, expires_at=2.0, state="open")
    assert "env_settings" not in Lease(**fields).to_public()


def test_an_opted_in_miner_serves_a_tasks_customer_envs():
    from reliquary.miner.rl_task_choice import expected_pay, served_in

    task = {"task_id": "t", "auction": {"environments": {
        ENV: {"budget": 1, "share": 1.0, "price_alpha_per_mtoken": 2.0}}}}
    assert served_in(task, ["math"]) == ["math"]
    assert served_in(task, ["math"], customer_envs=True) == ["math", ENV]
    assert expected_pay(task, served_envs=["math"], asks={}) is None
    assert expected_pay(task, served_envs=["math"], asks={}, customer_envs=True) == 2.0


def test_the_controller_serves_the_rows_and_the_miner_checks_them(monkeypatch):
    from fastapi import APIRouter, FastAPI
    from fastapi.testclient import TestClient

    from reliquary.customer_env import rl

    rows = [{"key": "t0", "system": None, "prompt": "1+1=", "prompt_sha256": "0" * 64}]
    body = b"".join(json.dumps(r, sort_keys=True, separators=(",", ":")).encode() + b"\n" for r in rows)
    monkeypatch.setattr(rl, "task_environments", lambda: [ENV])
    monkeypatch.setattr(rl, "_ROWS_CACHE", {ENV: (body, hashlib.sha256(body).hexdigest())})
    router = APIRouter()
    rl.mount_rows_routes(router)
    app = FastAPI()
    app.include_router(router)
    client = TestClient(app)
    assert client.get("/rl/customer-envs").json() == {"environments": [ENV]}
    assert client.get("/rl/customer-envs/rows", params={"env": "customer:x/y@" + "b" * 64}).status_code == 404

    def get(url, params):
        return client.get(url.replace("http://controller", ""), params=params)

    assert rl.load_rows(ENV, rows_url="http://controller/rl", get=get) == rows

    def tampered(url, params):
        response = get(url, params)
        response.headers[rl.PROMPT_SET_HEADER] = "f" * 64
        return response

    with pytest.raises(RuntimeError, match="do not hash"):
        rl.load_rows(ENV, rows_url="http://controller/rl", get=tampered)


def test_a_validator_finds_the_image_in_the_versions_bucket_record():
    from reliquary.customer_env.registry import subnet_key
    from reliquary.customer_env.scoring import bucket_image_for

    key = subnet_key(env_id("acct", "my-env", "a" * 64), "record.json")
    image_for = bucket_image_for({key: json.dumps({"status": "qualified", "image_digest": "img@sha256:" + "c" * 64})
                                  .encode()}.get)
    assert image_for(ENV) == "img@sha256:" + "c" * 64
    with pytest.raises(LookupError):
        bucket_image_for({key: json.dumps({"status": "disabled"}).encode()}.get)(ENV)
