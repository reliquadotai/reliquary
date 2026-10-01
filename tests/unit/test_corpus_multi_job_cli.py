"""Starting one corpus validator on several task ids, and the contract it runs."""

from __future__ import annotations

from dataclasses import asdict, replace
import json

import pytest
from typer.testing import CliRunner

from reliquary.environment.abi import canonical_sha256
from reliquary.validator.task_config import (
    TaskConfigError,
    merge_corpus_contracts,
    resolve_corpus_task_configs,
)
from tests.unit.test_jobs_cli import _rl_entry, registry  # noqa: F401

PROFILE_ID = "corpus-multi-test"


def _process_contract():
    """Two environments, a toploc proof and an architecture: what a merged
    corpus contract looks like."""
    from reliquary.cli.main import _with_enforced_toploc
    from reliquary.constants import PROTOCOL_GENERATION_CONTRACT

    contract = _with_enforced_toploc(dict(PROTOCOL_GENERATION_CONTRACT))
    return {**contract, "profile_id": PROFILE_ID, "model_architecture": "Qwen3ForCausalLM"}


def _narrowed(contract, environment):
    return {**contract, "environments": {environment: contract["environments"][environment]}}


def _corpus_entry(task_id, job_id, environment, cap, contract=None):
    from reliquary.shared.task_registry import MECHANISM_CORPUS_GENERATION, TaskEntry
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    contract = contract if contract is not None else _narrowed(_process_contract(), environment)
    params = asdict(PRODUCTION_PRICE_PARAMS)
    params["cap"] = params["floor"] = cap
    return TaskEntry(
        task_id=task_id, profile_id=PROFILE_ID, profile_sha256=canonical_sha256(contract),
        mechanism=MECHANISM_CORPUS_GENERATION, params=params, status="active",
        retired_at=None, job_id=job_id, contract=contract,
    )


def _entries():
    return {
        "corpus-math": _corpus_entry("corpus-math", "math-v1", "openmathinstruct", 0.1),
        "corpus-code": _corpus_entry("corpus-code", "code-v1", "opencodeinstruct", 0.2),
    }


def _resolve(entries, ids=("corpus-math", "corpus-code"), process=None):
    return resolve_corpus_task_configs(
        entries, ids, profile_id=PROFILE_ID,
        generation_contract=process if process is not None else _process_contract(),
    )


# --------------------------------------------------------------------------
# merge_corpus_contracts
# --------------------------------------------------------------------------


def test_merging_two_narrowed_contracts_gives_back_the_whole():
    entries = _entries()
    merged = merge_corpus_contracts({t: e.contract for t, e in entries.items()})
    assert merged == _process_contract()


def test_merging_contracts_with_different_proofs_names_the_field_and_tasks():
    entries = _entries()
    other = json.loads(json.dumps(entries["corpus-code"].contract))
    other["proofs"][0]["mant_mean_threshold"] += 10.0
    with pytest.raises(ValueError) as caught:
        merge_corpus_contracts({"corpus-math": entries["corpus-math"].contract, "corpus-code": other})
    assert "proofs" in str(caught.value)
    assert "corpus-math" in str(caught.value) and "corpus-code" in str(caught.value)


def test_merging_one_environment_declared_two_ways_refuses():
    contract = _process_contract()
    a = _narrowed(contract, "openmathinstruct")
    b = json.loads(json.dumps(a))
    b["environments"]["openmathinstruct"]["max_new_tokens"] += 1
    with pytest.raises(ValueError, match="openmathinstruct"):
        merge_corpus_contracts({"corpus-a": a, "corpus-b": b})


# --------------------------------------------------------------------------
# resolve_corpus_task_configs
# --------------------------------------------------------------------------


def test_two_corpus_tasks_resolve_in_order_with_their_own_caps():
    configs = _resolve(_entries())
    assert [c.task_id for c in configs] == ["corpus-math", "corpus-code"]
    assert [c.entry.job_id for c in configs] == ["math-v1", "code-v1"]
    assert [c.emission_cap for c in configs] == [pytest.approx(0.1), pytest.approx(0.2)]


def test_an_rl_task_among_the_ids_refuses():
    entries = {**_entries(), "default": _rl_entry("default", 0.5)}
    with pytest.raises(TaskConfigError, match="corpus"):
        _resolve(entries, ids=("corpus-math", "default"))


def test_an_undeclared_id_refuses():
    with pytest.raises(TaskConfigError, match="corpus-nope"):
        _resolve(_entries(), ids=("corpus-math", "corpus-nope"))


def test_a_process_running_one_tasks_contract_refuses_with_the_remedy():
    process = _entries()["corpus-math"].contract
    with pytest.raises(TaskConfigError) as caught:
        _resolve(_entries(), process=process)
    message = str(caught.value)
    assert "corpus-code" in message and "environments" in message
    assert "tasks contract" in message


def test_a_task_whose_toploc_proof_differs_refuses_naming_proofs():
    entries = _entries()
    other = json.loads(json.dumps(entries["corpus-code"].contract))
    other["proofs"][0]["mant_mean_threshold"] += 10.0
    entries["corpus-code"] = _corpus_entry("corpus-code", "code-v1", "opencodeinstruct", 0.2,
                                           contract=other)
    with pytest.raises(TaskConfigError, match="proofs"):
        _resolve(entries)


def test_a_legacy_entry_without_a_contract_refuses():
    entries = _entries()
    entries["corpus-code"] = replace(entries["corpus-code"], contract=None)
    with pytest.raises(TaskConfigError, match="contract"):
        _resolve(entries)


# --------------------------------------------------------------------------
# The CLI
# --------------------------------------------------------------------------


def _boot(monkeypatch, registry, ids, side_effect=None):  # noqa: F811
    from types import SimpleNamespace

    import bittensor
    import reliquary.cli.main as cli_module
    import reliquary.constants as constants
    import reliquary.infrastructure.chain as chain
    import reliquary.validator.corpus_validator as corpus_validator

    monkeypatch.setattr(constants, "TASK_IDS", tuple(ids))
    monkeypatch.setattr(constants, "PROTOCOL_PROFILE_ID", PROFILE_ID)
    monkeypatch.setattr(constants, "PROTOCOL_GENERATION_CONTRACT", _process_contract())
    monkeypatch.setattr(bittensor, "Wallet", lambda **kw: SimpleNamespace())

    async def subtensor():
        return SimpleNamespace()

    monkeypatch.setattr(chain, "get_subtensor", subtensor)
    calls = []

    async def fake_run(**kwargs):
        calls.append(kwargs)
        if side_effect is not None:
            raise side_effect

    monkeypatch.setattr(corpus_validator, "run_corpus_validator", fake_run)
    return CliRunner().invoke(cli_module.app, ["validate"]), calls


def test_validate_on_two_corpus_ids_starts_one_validator_for_both(monkeypatch, registry):  # noqa: F811
    registry["entries"] = _entries()

    result, calls = _boot(monkeypatch, registry, ["corpus-math", "corpus-code"])

    assert result.exit_code == 0, (result.output, result.exception)
    assert len(calls) == 1
    jobs = calls[0]["jobs"]
    assert [(e.task_id, e.job_id) for e, _ in jobs] == [("corpus-math", "math-v1"),
                                                        ("corpus-code", "code-v1")]
    assert [cap for _, cap in jobs] == [pytest.approx(0.1), pytest.approx(0.2)]
    assert calls[0]["set_weights"] is False


def test_validate_on_ids_mixing_rl_and_corpus_exits_four(monkeypatch, registry):  # noqa: F811
    registry["entries"] = {"corpus-math": _entries()["corpus-math"],
                           "logic": _rl_entry("logic", 0.5)}

    result, calls = _boot(monkeypatch, registry, ["corpus-math", "logic"])

    assert result.exit_code == 4, (result.output, result.exception)
    assert calls == []


def test_a_multi_job_startup_refusal_exits_four(monkeypatch, registry):  # noqa: F811
    registry["entries"] = _entries()

    result, calls = _boot(monkeypatch, registry, ["corpus-math", "corpus-code"],
                          side_effect=RuntimeError("job 'code-v1' declares checkpoint_sha256 ..."))

    assert result.exit_code == 4, (result.output, result.exception)
    assert len(calls) == 1


def test_tasks_contract_with_two_ids_prints_the_merged_contract(registry):  # noqa: F811
    from reliquary.cli.main import app

    registry["entries"] = _entries()
    result = CliRunner().invoke(app, ["tasks", "contract", "--task-id", "corpus-math",
                                      "--task-id", "corpus-code"])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output) == _process_contract()


def test_tasks_contract_with_one_id_is_unchanged(registry):  # noqa: F811
    from reliquary.cli.main import app

    registry["entries"] = _entries()
    result = CliRunner().invoke(app, ["tasks", "contract", "--task-id", "corpus-math"])
    assert result.exit_code == 0, result.output
    assert result.output == json.dumps(_entries()["corpus-math"].contract, sort_keys=True,
                                       separators=(",", ":")) + "\n"


def test_tasks_contract_refuses_contracts_that_cannot_share_a_process(registry):  # noqa: F811
    from reliquary.cli.main import app

    entries = _entries()
    other = json.loads(json.dumps(entries["corpus-code"].contract))
    other["proofs"][0]["mant_mean_threshold"] += 10.0
    entries["corpus-code"] = _corpus_entry("corpus-code", "code-v1", "opencodeinstruct", 0.2,
                                           contract=other)
    registry["entries"] = entries
    result = CliRunner().invoke(app, ["tasks", "contract", "--task-id", "corpus-math",
                                      "--task-id", "corpus-code"])
    assert result.exit_code == 1
    assert "proofs" in result.output


def test_the_job_set_is_hot_only_when_the_operator_opts_in(monkeypatch, registry):  # noqa: F811
    registry["entries"] = _entries()
    monkeypatch.delenv("RELIQUARY_CORPUS_HOT_JOBS", raising=False)
    result, calls = _boot(monkeypatch, registry, ["corpus-math", "corpus-code"])
    assert result.exit_code == 0, (result.output, result.exception)
    assert calls[0]["read_registry"] is None

    monkeypatch.setenv("RELIQUARY_CORPUS_HOT_JOBS", "1")
    result, calls = _boot(monkeypatch, registry, ["corpus-math", "corpus-code"])
    assert result.exit_code == 0, (result.output, result.exception)
    import asyncio

    assert set(asyncio.run(calls[0]["read_registry"]())) == set(registry["entries"])


def test_a_bad_recheck_fraction_exits_four_with_the_critical_line(monkeypatch, registry):  # noqa: F811
    registry["entries"] = _entries()
    monkeypatch.setenv("RELIQUARY_CORPUS_REMOTE_AUDIT", "1")
    monkeypatch.setenv("RELIQUARY_CORPUS_RECHECK_FRACTION", "0")
    result, calls = _boot(monkeypatch, registry, ["corpus-math", "corpus-code"])
    assert result.exit_code == 4, (result.output, result.exception)
    assert calls == []
