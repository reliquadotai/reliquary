"""Service declaration uses the real CLI and requires explicit fleet acknowledgement."""

from dataclasses import replace
import json
from unittest.mock import Mock

import pytest
from typer.testing import CliRunner

from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, MECHANISM_SERVICE_RL, validate_entry
from tests.unit.test_service_task_registry import service_entry


@pytest.mark.parametrize("case", ["missing-ack", "ack-valid", "ack-without-file"])
def test_tasks_create_service_cli_requires_ack_and_preserves_the_bound_contract(tmp_path, monkeypatch, case):
    import reliquary.cli.main as cli
    import reliquary.infrastructure.task_registry_store as store

    service = service_entry()
    generation = replace(service, mechanism=MECHANISM_RL_DISCOVERED_PRICE, service_contract=None,
                         params={**service.params, "min_incentive_share": 0.01, "min_incentive_ramp_start": 0.005})
    validate_entry(generation)
    builder = Mock(return_value=generation)
    monkeypatch.setattr(cli, "build_task_entry", builder)
    captured = []
    async def create(entry):
        validate_entry(entry)
        captured.append(entry)
    monkeypatch.setattr(store, "create_task", create)
    contract_file = tmp_path / "service-contract.json"
    contract_file.write_text(json.dumps(service.service_contract))
    args = ["tasks", "create", "--task-id", generation.task_id, "--profile-id", generation.profile_id,
            "--cap", str(generation.params["cap"])]
    if case != "ack-without-file":
        args += ["--service-contract", str(contract_file)]
    if case != "missing-ack":
        args += ["--ack-service-fleet"]
    result = CliRunner().invoke(cli.app, args)
    builder.assert_called_once()
    if case == "ack-valid":
        assert result.exit_code == 0, result.output
        assert len(captured) == 1
        declared = captured[0]
        assert declared.mechanism == MECHANISM_SERVICE_RL
        assert declared.service_contract == service.service_contract
        assert declared.contract == generation.contract
        assert declared.profile_sha256 == generation.profile_sha256
        assert declared.params["min_incentive_share"] == 0.0
        assert declared.params["min_incentive_ramp_start"] == 0.0
    else:
        assert result.exit_code == 1, result.output
        assert not captured
        expected = "requires --ack-service-fleet" if case == "missing-ack" else "--ack-service-fleet requires --service-contract"
        assert expected in result.output
    assert generation.mechanism == MECHANISM_RL_DISCOVERED_PRICE
    assert generation.service_contract is None
    assert generation.params["min_incentive_share"] == 0.01
