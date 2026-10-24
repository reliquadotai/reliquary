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
                "index_range": [0, 3], "set_id": "dapo-train-slice",
                "disjointness": {"external_benchmark": False, "held_out": []}}


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
    {"index_range": [0]},                                   # F2: incomplete card fails closed
    {"index_range": None},
    {"split": None},
    {"disjointness": None},
])
def test_held_out_and_benchmark_sets_are_refused(override):
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, **override})


def test_card_without_a_required_key_is_refused():
    for key in ("split", "index_range", "disjointness"):
        card = {k: v for k, v in CATALOG_CARD.items() if k != key}
        with pytest.raises(HeldOutEvalSet):
            refuse_held_out(contract_for(b"x\n"), card)


def test_card_disagreeing_with_a_contract_that_carries_provenance_is_refused():
    contract = contract_for(b"x\n")
    class Carrying:
        def __getattr__(self, name):
            return getattr(contract, name)
        def to_dict(self):
            value = contract.to_dict()
            value["dataset"] = {**value["dataset"], "source": "other_source"}
            return value
    with pytest.raises(HeldOutEvalSet, match="disagrees"):
        refuse_held_out(Carrying(), CATALOG_CARD)


SLIPPING_FORMS = [
    "princeton-nlp/SWE-bench_Verified", "SWE-bench Verified", "swe-bench", "swe-bench-lite", "swebench",
    "MMLU Pro", "terminalbench", "aimev2", "livecodebenchv6", "gpqadiamond", "Idavidrein/gpqa",
    "AIME_2025", "aime25", "LiveCodeBench/code_generation_lite", "TIGER-Lab/MMLU-Pro", "gorilla-llm/BFCL_v3",
    "allenai/IFBench_test", "sierra-research/tau2-bench", "tau-bench", "SWEBenchVerified", "lcb-v5",
    "AIMEI", "AIMEII", "AIME_I", "aime-ii", "aimei",
]
LEGIT_TRAIN_NAMES = [
    "nvidia/OpenMathInstruct-2", "agentica-org/DeepScaleR-Preview-Dataset", "open-r1/codeforces",
    "PrimeIntellect/verifiable-math-problems", "allenai/tulu-3-sft-mixture", "claimed-rows", "paid-ament",
    "mainstream-slice", "xaime-slice", "ugpqa", "tau2x", "gpqas", "HuggingFaceH4/ultrafeedback_binarized",
    "airbnb/listings", "aimed-dataset", "aimeiser", "aimeinc", "AI-MO/NuminaMath-CoT", "terminal-sessions-train",
]


@pytest.mark.parametrize("name", SLIPPING_FORMS)
def test_every_spelling_of_a_held_out_name_is_refused_in_every_name_field(name):
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n", dataset_id="train-slice"), {**CATALOG_CARD, "set_id": name})
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n", dataset_id="train-slice"), {**CATALOG_CARD, "source": name})
    for key in ("dataset", "repo", "hf_id", "id"):   # every identifier a card can carry its name in
        with pytest.raises(HeldOutEvalSet):
            refuse_held_out(contract_for(b"x\n", dataset_id="train-slice"), {**CATALOG_CARD, key: name})
    if re.fullmatch(r"[A-Za-z0-9_.-]+", name):   # dataset ids are canonical identifiers
        with pytest.raises(HeldOutEvalSet):
            refuse_held_out(contract_for(b"x\n", dataset_id=name), CATALOG_CARD)


@pytest.mark.parametrize("name", LEGIT_TRAIN_NAMES)
def test_legitimate_training_dataset_names_pass(name):
    refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, "set_id": name})
    refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, "source": name} | {"source": CATALOG_CARD["source"], "name": name})


@pytest.mark.parametrize("key", ["dataset", "repo", "hf_id", "id"])
def test_legitimate_names_pass_in_the_identifier_card_keys_too(key):
    refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, key: "nvidia/OpenMathInstruct-2"})


def test_benchmark_name_in_dataset_id_is_refused():
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n", dataset_id="swe-bench-verified-sample"), CATALOG_CARD)


@pytest.mark.parametrize("name", ["claimed-rows", "paid-ament", "mainstream-slice", "xaime-slice", "ugpqa", "tau2x", "gpqas"])
def test_names_match_on_word_boundaries(name):
    refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, "set_id": name})


# Checked-in copy of the 7 tasksets of reliquary-environments benchmarks/heldout/configs (CI has no checkout).
TASKSET_IDS = ["aime25", "aime26", "bfcl-v3", "gpqa", "ifbench", "livecodebench", "mmlu-pro"]


@pytest.mark.parametrize("taskset", TASKSET_IDS)
def test_guard_refuses_every_checked_in_taskset_id(taskset):
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, "set_id": taskset})
    with pytest.raises(HeldOutEvalSet):
        refuse_held_out(contract_for(b"x\n"), {**CATALOG_CARD, "source": f"org/{taskset}"})


def test_guard_covers_every_taskset_the_environments_repo_declares():
    configs = Path.home() / "reliquadotai/reliquary-environments/benchmarks/heldout/configs"
    if not configs.is_dir():
        pytest.skip("reliquary-environments checkout not present")
    ids = {re.search(r'id = "([^"]+)"', p.read_text()).group(1) for p in configs.glob("*.toml")}
    assert ids
    assert ids == set(TASKSET_IDS)   # the checked-in list must follow the repo
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


def _ledger_with(tmp_path, audit, status="reserved"):
    ledger = ExplorationLedger(sqlite3.connect(tmp_path / "l.sqlite3"), order_sha256="a" * 64)
    ledger.db.execute("INSERT INTO exploration_entitlements(observation_id, order_id, window, environment, hotkey,"
                      " prompt_idx, amount, draw_round, forced, audit, status) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                      ("run-obs", "a" * 64, 0, "env", "hk", 1, 1.0, 5, 0, audit, status))
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
        folder = tmp_path / audit
        folder.mkdir()
        with pytest.raises(ValueError, match="generation"):
            curate_rows(source, body, manifest, contract, set_card=CATALOG_CARD, ledger=_ledger_with(folder, audit))
    passed = tmp_path / "ok"
    passed.mkdir()
    curated, _ = curate_rows(source, body, manifest, contract, set_card=CATALOG_CARD, ledger=_ledger_with(passed, "passed"))
    assert curated == source


def test_a_passed_audit_with_a_forfeited_entitlement_does_not_count(tmp_path):
    from reliquary.services.exploration import STATUS_FORFEITED
    source, contract, body, manifest = _exploration_mapping()
    with pytest.raises(ValueError, match="generation"):
        curate_rows(source, body, manifest, contract, set_card=CATALOG_CARD,
                    ledger=_ledger_with(tmp_path, "passed", STATUS_FORFEITED))


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
