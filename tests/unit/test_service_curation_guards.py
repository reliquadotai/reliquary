import asyncio
import hashlib
import json
import re
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.curation import curate_rows
from reliquary.services.exploration import ExplorationLedger
from reliquary.services.heldout_guard import HELD_OUT_BENCHMARK_NAMES, HeldOutEvalSet, refuse_held_out
from reliquary.services.mapping import mapping_artifact


def contract_for(source: bytes, dataset_id="train-slice"):
    value = json.loads((Path(__file__).parents[1] / "fixtures/service_contract_v1.json").read_text())
    value["dataset"] = {"id": dataset_id, "sha256": hashlib.sha256(source).hexdigest()}
    return ServiceContract.from_dict(value)


CATALOG_CARD = {"source_kind": "catalog", "source": "reliquary_dapo_math_v1", "split": "train",
                "index_range": [0, 3], "set_id": "dapo-train-slice"}


def obs(contract, row, group, rewards, generation="verified"):
    return {"schema": "prompt-observation/v1", "context_sha256": contract.context_sha256, "row_id": row,
            "group_id": group, "expected_samples": 2, "sample_ids": ["s0", "s1"], "rewards_bps": rewards,
            "tokens": [1, 1], "window": 0, "source_sha256": "e" * 64,
            "verification": {"generation": generation, "sampling": "unverified", "grading": "graded"}}


def test_the_clean_training_card_is_accepted():
    refuse_held_out(contract_for(b"x\n"), CATALOG_CARD)  # guard must not over-refuse


# Every card is a catalog training slice EXCEPT for the one field under test, so each case
# can only be refused by the rule it names.
@pytest.mark.parametrize("override", [
    {"set_id": "aime-2025"},
    {"source": "gpqa_diamond"},
    {"set_id": "bfcl-v3-slice"},
    {"env": "ifbench"},
    {"taskset": {"id": "mmlu-pro"}},
    {"set_id": "livecodebench-v6"},
    {"disjointness": {"external_benchmark": True, "held_out": []}},
    {"disjointness": {"held_out": [{"what": "code"}]}},
    {"split": "eval"},                                      # the math held-out split (sets.HELD_OUT)
    {"source": "reliquary_code_v1", "index_range": [2_400_000, 2_450_000]},   # code held-out tail
    {"source": "reliquary_logic_v2", "split": "eval"},
    {"source_kind": "verifiers"},
])
def test_held_out_and_benchmark_sets_are_refused(override):
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, **override})


def test_benchmark_name_in_dataset_id_is_refused():
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n", dataset_id="swe-bench-verified-sample"), CATALOG_CARD)


@pytest.mark.parametrize("name", ["claimed-rows", "paid-ament", "mainstream-slice", "xaime-slice", "ugpqa", "tau2x", "gpqas"])
def test_names_match_on_word_boundaries(name):
    refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, "set_id": name})


def test_guard_covers_every_taskset_the_environments_repo_declares():
    configs = Path.home() / "reliquadotai/reliquary-environments/benchmarks/heldout/configs"
    if not configs.is_dir():
        pytest.skip("reliquary-environments checkout not present")
    ids = {re.search(r'id = "([^"]+)"', p.read_text()).group(1) for p in configs.glob("*.toml")}
    assert ids
    for taskset in ids:
        with pytest.raises(HeldOutEvalSet):
            refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, "set_id": taskset})
    assert "aime" in HELD_OUT_BENCHMARK_NAMES


def test_curation_refuses_a_held_out_card_end_to_end():
    source = b'{"row_id":"a"}\n'
    contract = contract_for(source)
    body, manifest = mapping_artifact(contract, [obs(contract, "a", "g", [0, 10000])], expected_rows=1)
    with pytest.raises(HeldOutEvalSet):
        curate_rows(source, body, manifest, contract, set_card={**CATALOG_CARD, "set_id": "gpqa"})
    with pytest.raises(TypeError):
        curate_rows(source, body, manifest, contract)  # set_card is required


def test_row_is_selected_only_when_all_its_groups_match():
    source = b'{"row_id":"a"}\n{"row_id":"b"}\n'
    contract = contract_for(source)
    body, manifest = mapping_artifact(contract, [obs(contract, "a", "g1", [0, 10000]), obs(contract, "a", "g2", [0, 0]),
                                                 obs(contract, "b", "g1", [0, 10000])], expected_rows=2)
    curated, out = curate_rows(source, body, manifest, contract, set_card=CATALOG_CARD)
    assert curated == b'{"row_id":"b"}\n' and out["mixed_rows"] == 1
    assert out["generation_audit"] == "full"


def test_blank_lines_do_not_crash_and_are_not_emitted():
    source = b'{"row_id":"a"}\n\n   \n{"row_id":"b"}\n\n'
    contract = contract_for(source)
    body, manifest = mapping_artifact(contract, [obs(contract, "a", "g", [0, 10000]), obs(contract, "b", "g", [0, 10000])],
                                      expected_rows=2)
    padded = body + b"\n  \n"
    manifest["files"]["mapping.jsonl"]["sha256"] = hashlib.sha256(padded).hexdigest()
    curated, _ = curate_rows(source, padded, manifest, contract, set_card=CATALOG_CARD)
    assert curated == b'{"row_id":"a"}\n{"row_id":"b"}\n'


def _ledger_with(tmp_path, audit):
    ledger = ExplorationLedger(sqlite3.connect(tmp_path / "l.sqlite3"), order_sha256="a" * 64)
    ledger.db.execute("INSERT INTO exploration_entitlements(observation_id, order_id, window, environment, hotkey,"
                      " prompt_idx, amount, draw_round, forced, audit, status) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      ("run-obs", "a" * 64, 0, "env", "hk", 1, 1.0, 5, 0, audit, "reserved"))
    ledger.db.commit()
    return ledger


def _exploration_mapping():
    source = b'{"row_id":"a"}\n'
    contract = contract_for(source)
    row = obs(contract, "a", "g", [0, 10000], generation="unverified")
    body, manifest = mapping_artifact(contract, [row], expected_rows=1)
    body = json.dumps({**json.loads(body), "run_observation_id": "run-obs"}, sort_keys=True).encode() + b"\n"
    manifest["files"]["mapping.jsonl"]["sha256"] = hashlib.sha256(body).hexdigest()
    return source, contract, body, manifest


def test_exploration_group_counts_only_when_its_audit_passed(tmp_path):
    source, contract, body, manifest = _exploration_mapping()
    with pytest.raises(ValueError, match="generation"):
        curate_rows(source, body, manifest, contract, set_card=CATALOG_CARD)  # no ledger: not proven
    for audit in ("queued", "pending_draw", "unaudited", "failed"):   # sampled-but-not-drawn etc.
        with pytest.raises(ValueError, match="generation"):
            curate_rows(source, body, manifest, contract, set_card=CATALOG_CARD,
                        ledger=_ledger_with(tmp_path / audit if (tmp_path / audit).mkdir() is None else tmp_path, audit))
    passed = tmp_path / "ok"
    passed.mkdir()
    curated, _ = curate_rows(source, body, manifest, contract, set_card=CATALOG_CARD, ledger=_ledger_with(passed, "passed"))
    assert curated == source


def _collect(verdicts):
    records = SimpleNamespace(
        list_verdict_ids=lambda job: asyncio.sleep(0, result=list(verdicts)),
        read_verdict=lambda job, sid: asyncio.sleep(0, result=verdicts[sid]),
        read_submission=lambda job, sid: asyncio.sleep(0, result={"prompt_index": 0, "completions": ["c"]}))
    from reliquary.eval.grading import collect_job_records
    return asyncio.run(collect_job_records(SimpleNamespace(job_id="j"), records, include_generation_status=True))


def test_generation_status_under_partial_audit():
    verdicts = {"s1": {"passed": True, "audited": True, "hotkey": "h1"},
                "s2": {"passed": True, "audited": False, "hotkey": "h2"},
                "s3": {"passed": True, "hotkey": "h3"}}   # no key: predates sampling, counts as audited
    result = _collect(verdicts)
    assert result["generation_status"] == "sampled" and result["generation_verified"] is False  # R11
    verdicts["s2"]["audited"] = True
    result = _collect(verdicts)
    assert result["generation_status"] == "verified" and result["generation_verified"] is True
    verdicts["s4"] = {"passed": False, "audited": True, "hotkey": "h2"}
    verdicts["s2"]["audited"] = False
    result = _collect(verdicts)
    assert result["generation_status"] == "unverified" and result["generation_verified"] is False
