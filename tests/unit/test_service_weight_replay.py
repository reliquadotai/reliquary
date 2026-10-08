"""Weight replay of service-contract/v2 archives: validated from their own fields, abstain on a forgery."""
import copy
import json
from types import SimpleNamespace

import pytest

from reliquary.services.settlement import SettlementError, settle_window
from reliquary.shared.task_registry import MECHANISM_SERVICE_RL
from reliquary.validator import weight_only
from reliquary.validator.weight_only import WeightOnlyValidator
from tests.unit.service_v2_fixtures import CODE, MATH, contract_v2
from tests.unit.test_service_runtime_v2 import archive as runtime_archive, audited, explore, runtime, train
from tests.unit.test_service_settlement_v2 import PICKS, SLOTS, archive, envelope

TASK = "next-rl"
GEOM = (PICKS, SLOTS)
G = 0.25 / (PICKS * SLOTS)


def declared(cap=0.5, extra=None):
    entry = SimpleNamespace(mechanism=MECHANISM_SERVICE_RL, service_contract=contract_v2().to_dict(),
                            params={"cap": cap})
    legacy = SimpleNamespace(mechanism="auction", service_contract=None, params={"cap": 0.5})
    return {TASK: entry, "default": legacy, **(extra or {})}


def settled(rows=(("a", MATH),), exploration=None, window=3, **kw):
    contract = contract_v2()
    exploration = {CODE: {"x": 1}} if exploration is None else exploration
    record = settle_window(archive={**archive(list(rows)), "window_start": window}, contract=contract,
                           envelope=envelope(contract, {MATH: 0.25, CODE: 0.25}), exploration=exploration,
                           aborted=kw.pop("aborted", False))
    return {**json.loads(json.dumps(record)), "task_id": TASK, **kw}


def check(records, decl=None):
    return WeightOnlyValidator._validated_service_archives(records, decl or declared(), geometry=GEOM)


def test_replay_of_a_v2_service_archive_is_deterministic():
    record = settled()
    first = WeightOnlyValidator._replay_ema(check([record]), caps={TASK: 0.5}, service_tasks=frozenset({TASK}))
    second = WeightOnlyValidator._replay_ema(check([dict(record)]), caps={TASK: 0.5}, service_tasks=frozenset({TASK}))
    assert first == second and first["x"] > 0 and first["a"] > 0


def test_service_fields_listed_for_fetch():
    fields = weight_only.SERVICE_ARCHIVE_FIELDS
    for needed in ("batch", "service_scale_by_environment", "service_training_by_environment",
                   "service_exploration_by_environment", "service_pools_by_environment"):
        assert needed in fields
    for banned in ("service_context_contract", "randomness", "service_cooldown_advice",
                   "service_training_recomputed_delta", "service_order_contract"):
        assert banned not in fields


def test_replay_pays_the_recomputed_map_from_one_source():
    record = settled()
    (out,) = check([record])
    assert set(out) == {"task_id", "window_start", "window_status", "rewards_by_hotkey"}   # batch dropped
    assert out["rewards_by_hotkey"]["a"] == pytest.approx(G) and out["rewards_by_hotkey"]["x"] == pytest.approx(0.15 * G)


def test_runtime_archive_round_trips_through_json_and_replays_to_the_runtime_map(tmp_path):
    rt = runtime(tmp_path)
    paid = explore(rt, hotkey="x")
    audited(rt, paid)
    train(rt, prompt=20, hotkey="a")
    train(rt, prompt=21, hotkey="b", env=CODE)
    rt.finalize_exploration(1, environment=MATH, now=10_050.0)
    result = rt.reconcile_archive(runtime_archive(batch=[("a", MATH, 20), ("b", CODE, 21)]), now=10_100.0)
    record = {**json.loads(json.dumps(result)), "task_id": TASK}
    (out,) = WeightOnlyValidator._validated_service_archives([record], declared())   # protocol geometry
    assert out["rewards_by_hotkey"] == pytest.approx(result["rewards_by_hotkey"], abs=1e-12)
    assert set(out["rewards_by_hotkey"]) == {"a", "b", "x"}
    assert "service_cooldown_advice" in result       # informational: present, ignored


def test_recovery_artifacts_are_tolerated():
    record = settled(window_status="recovered_partial", service_training_recomputed_delta=0.7,
                     service_cooldown_advice={"garbage": [float("1e9")]}, randomness="not-a-beacon")
    (out,) = check([record])
    assert out["window_status"] == "recovered_partial" and out["rewards_by_hotkey"]["a"] > 0


def test_aborted_archive_pays_nothing_at_all_training_included():
    record = settled(aborted=True, exploration={})
    assert record["rewards_by_hotkey"]["a"] > 0           # the archive still carries the training amounts
    out = check([record])
    ema = WeightOnlyValidator._replay_ema(out, caps={TASK: 0.5}, service_tasks=frozenset({TASK}))
    assert ema == {}


def test_aborted_archive_awarding_exploration_is_refused():
    record = settled()
    record["window_status"] = "aborted"
    with pytest.raises(SettlementError):
        check([record])


def test_empty_non_aborted_window_pays_nothing_and_is_not_a_refusal():
    record = settled(rows=(), exploration={})
    (out,) = check([record])
    assert out["rewards_by_hotkey"] == {}


def mutate(path_fn):
    record = settled(rows=(("a", MATH), ("a", MATH)))
    path_fn(record)
    return record


FORGERIES = {
    "geometry": lambda r: r.update(service_picks_target=1, service_batch_slots=1),
    "scale": lambda r: r["service_scale_by_environment"].update({MATH: 0.5}),
    "training_map": lambda r: r["service_training_by_environment"][MATH].update(a=0.25),
    "rewards_inflated": lambda r: r["rewards_by_hotkey"].update(a=0.25),
    "rewards_extra_hotkey": lambda r: r["rewards_by_hotkey"].update(mallory=0.01),
    "exploration_over_cap": lambda r: r.update(service_exploration_by_environment={CODE: {"x": 10}}),
    "exploration_float": lambda r: r.update(service_exploration_by_environment={CODE: {"x": 1.5}}),
    "exploration_env_outside": lambda r: r.update(service_exploration_by_environment={"other": {"x": 1}}),
    "wrong_order": lambda r: r.update(service_order_sha256="0" * 64),
    "schedule_hash": lambda r: r.update(service_schedule_sha256="1" * 64),
    "schedule_body": lambda r: r["service_schedule"].update(revision=99),
    "policy": lambda r: r.update(service_payment_policy="service-first-pays-all/v9"),
    "pool_over_cap_share": lambda r: r["service_pools_by_environment"].update({MATH: 0.26}),
    "nan_pool": lambda r: r["service_pools_by_environment"].update({MATH: float("nan")}),
    "huge_pool": lambda r: r["service_pools_by_environment"].update({MATH: 1e300}),
    "nan_reward": lambda r: r["rewards_by_hotkey"].update(a=float("nan")),
    "huge_count": lambda r: r.update(service_exploration_by_environment={CODE: {"x": 10**400}}),
    "missing_prompt_idx": lambda r: r["batch"][0].pop("prompt_idx"),
    "batch_env_outside": lambda r: r["batch"][0].update(env_name="other"),
    "batch_not_list": lambda r: r.update(batch={"a": 1}),
    "missing_scale": lambda r: r.pop("service_scale_by_environment"),
    "missing_batch_pays_rows": lambda r: r.pop("batch"),
}


@pytest.mark.parametrize("name", sorted(FORGERIES))
def test_forged_archive_is_refused_without_raising_anything_but_valueerror(name):
    record = mutate(FORGERIES[name])
    with pytest.raises(ValueError):
        check([record])


def test_a_forgery_in_a_mixed_stream_refuses_the_service_task_only_through_valueerror():
    legacy = {"task_id": "default", "window_start": 3, "rewards_by_hotkey": {"l": 0.1}}
    good = settled()
    assert check([legacy, good])[0] is legacy                           # legacy passes through
    with pytest.raises(ValueError):
        check([legacy, mutate(FORGERIES["scale"])])


def test_legacy_archive_is_passed_through_untouched():
    legacy = [{"task_id": "default", "window_start": w, "rewards_by_hotkey": {"l": 0.1}, "junk": object()}
              for w in (1, 2)]
    out = check(legacy)
    assert out == legacy and all(a is b for a, b in zip(out, legacy))


def test_task_cap_is_the_registry_cap_not_one():
    record = settled()
    check([record], declared(cap=0.5))
    with pytest.raises(SettlementError, match="cap"):
        check([record], declared(cap=0.4))          # pools 0.25 each exceed 0.4 * 50 %
    for bad in (0, float("nan"), 2.0, "x", None):
        with pytest.raises(ValueError):
            check([record], declared(cap=bad))


def test_unusable_contract_in_the_registry_is_a_refusal():
    decl = declared()
    decl[TASK].service_contract = {"schema": "nope"}
    with pytest.raises(ValueError):
        check([settled()], decl)


def test_a_window_settled_twice_leaves_one_archive_deterministically():
    one = settled(rows=(("a", MATH),))
    two = settled(rows=(("b", MATH),))
    forward, backward = check([one, two]), check([two, one])
    assert len(forward) == 1 and forward == backward
    ema = WeightOnlyValidator._replay_ema(forward, caps={TASK: 0.5}, service_tasks=frozenset({TASK}))
    assert sum(ema.values()) < 0.2          # never both summed
    assert len(check([one, copy.deepcopy(one), settled(window=4)])) == 2


# ---- the whole submit path: a refusal abstains, a mixed epoch is paid

def wire(monkeypatch, service_records, legacy_records, decl):
    async def read_registry():
        return decl, "etag"

    async def task_ids(strict=False):
        return [TASK, "default"]

    async def window_keys(task_id=None, strict=False):
        return [3]

    async def recent(current_window, n, *, task_id=None, fields=None, **kw):
        records = service_records if task_id == TASK else legacy_records
        return [{k: v for k, v in r.items() if k != "task_id"} for r in records]

    async def periods(declared):
        return {}

    async def get_subtensor():
        return object()

    async def close_subtensor(s):
        return None

    monkeypatch.setattr(weight_only, "read_registry", read_registry)
    monkeypatch.setattr(weight_only.storage, "list_task_ids", task_ids)
    monkeypatch.setattr(weight_only.storage, "list_all_window_keys", window_keys)
    monkeypatch.setattr(weight_only.storage, "list_recent_datasets", recent)
    monkeypatch.setattr(weight_only.chain, "get_subtensor", get_subtensor)
    monkeypatch.setattr(weight_only.chain, "close_subtensor", close_subtensor)
    monkeypatch.setattr(WeightOnlyValidator, "_period_weights", staticmethod(periods))
    monkeypatch.setattr("reliquary.services.runtime.protocol_slot_geometry", lambda: GEOM)
    sent = []

    async def submit(self, subtensor, weights):
        sent.append(weights)
        return True

    monkeypatch.setattr(WeightOnlyValidator, "_submit_weights", submit)
    wallet = SimpleNamespace(hotkey=SimpleNamespace(ss58_address="validator"))
    return WeightOnlyValidator(wallet, 81), sent


@pytest.mark.asyncio
async def test_submit_once_pays_a_v2_task_and_a_legacy_task_in_one_epoch(monkeypatch):
    legacy = [{"task_id": "default", "window_start": 3, "window_status": "complete", "rewards_by_hotkey": {"l": 0.4}}]
    wov, sent = wire(monkeypatch, [settled()], legacy, declared())
    assert await wov.submit_once() is True
    (weights,) = sent
    assert set(weights) == {"a", "x", "l"} and weights["l"] > 0 and weights["a"] > 0


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ["scale", "training_map", "missing_prompt_idx", "nan_pool", "exploration_over_cap"])
async def test_submit_once_abstains_on_a_forged_service_archive(monkeypatch, name):
    legacy = [{"task_id": "default", "window_start": 3, "rewards_by_hotkey": {"l": 0.4}}]
    wov, sent = wire(monkeypatch, [mutate(FORGERIES[name])], legacy, declared())
    assert await wov.submit_once() is False
    assert sent == []


@pytest.mark.asyncio
async def test_submit_once_fetches_the_service_fields_only_for_service_tasks(monkeypatch):
    seen = {}
    wov, sent = wire(monkeypatch, [settled()], [{"task_id": "default", "window_start": 3, "rewards_by_hotkey": {}}],
                     declared())

    async def recent(current_window, n, *, task_id=None, fields=None, **kw):
        seen[task_id] = fields
        return [settled()] if task_id == TASK else []

    monkeypatch.setattr(weight_only.storage, "list_recent_datasets", recent)
    await wov.submit_once()
    assert seen["default"] == ("window_start", "window_status", "rewards_by_hotkey")
    assert seen[TASK] == ("window_start", "window_status", "rewards_by_hotkey") + weight_only.SERVICE_ARCHIVE_FIELDS


def test_replay_pays_the_recomputed_value_even_when_the_archived_one_is_within_tolerance():
    record = settled()
    exact = check([copy.deepcopy(record)])[0]["rewards_by_hotkey"]
    record["rewards_by_hotkey"]["a"] += 5e-13                  # inside the 1e-12 tolerance: accepted...
    assert check([record])[0]["rewards_by_hotkey"] == exact    # ...but never what is paid
