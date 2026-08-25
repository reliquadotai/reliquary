from __future__ import annotations

from reliquary.validator.service import (
    _prompt_mismatch_circuit_local_path,
    _prompt_mismatch_circuit_namespace,
)


class _SourceEnvironment:
    name = "openmathinstruct"

    def __init__(self, revision: str) -> None:
        self.revision = revision

    def source_health(self):
        return {
            "repo": "org/prompts",
            "revision": self.revision,
        }


def _namespace(*, revision="a" * 40, run_id="run-a", netuid=81, hotkey="hk-a"):
    return _prompt_mismatch_circuit_namespace(
        run_id=run_id,
        netuid=netuid,
        validator_hotkey=hotkey,
        environments={"openmathinstruct": _SourceEnvironment(revision)},
    )


def test_namespace_binds_run_network_validator_contract_and_prompt_source():
    baseline = _namespace()

    assert baseline.startswith("prompt-binding-v2:")
    assert _namespace() == baseline
    assert _namespace(revision="b" * 40) != baseline
    assert _namespace(run_id="run-b") != baseline
    assert _namespace(netuid=82) != baseline
    assert _namespace(hotkey="hk-b") != baseline


def test_local_state_path_is_partitioned_by_run_network_and_validator(
    monkeypatch,
    tmp_path,
):
    monkeypatch.setenv("RELIQUARY_STATE_DIR", str(tmp_path))
    first = _prompt_mismatch_circuit_local_path(
        "run/a",
        netuid=81,
        validator_hotkey="hk-a",
    )
    same = _prompt_mismatch_circuit_local_path(
        "run/a",
        netuid=81,
        validator_hotkey="hk-a",
    )
    other = _prompt_mismatch_circuit_local_path(
        "run/a",
        netuid=81,
        validator_hotkey="hk-b",
    )

    assert first == same
    assert first != other
    assert first.parent == tmp_path / "prompt_mismatch_circuit"
    assert first.name.startswith("run_a.netuid-81.")
