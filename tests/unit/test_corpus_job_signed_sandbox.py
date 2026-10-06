"""`episode.execution`: replay (default, never written) or signed_sandbox."""

import hashlib
import json

import pytest
from pathlib import Path

from reliquary.corpus.job import (
    EXECUTION_REPLAY, JobError, is_signed_sandbox, parse_job, sandbox_split,
)
from tests.unit.test_corpus_job_episode import _episode, _manifest

SANDBOX_COMMIT = "d" * 40
# What the registry writes into record 0: `name==version+g<sha16 of the code>`.
ENV_PACKAGE = "reliquary-swe==0.1.0a1+g0123456789abcdef"
GIB = 1024**3


def sandbox_spec(**overrides):
    spec = {"env": "reliquary-swe", "env_package": ENV_PACKAGE,
            "tools": ["bash", "edit"], "sandbox_commit": SANDBOX_COMMIT,
            "budgets": {"max_calls": 64, "per_call_timeout_s": 600, "cpu_s": 3600,
                        "wall_s": 3600, "memory_bytes": 4 * GIB, "pids": 1024,
                        "disk_bytes": 10 * GIB}}
    spec.update(overrides)
    return spec


def signed_episode(**overrides):
    fields = {"execution": "signed_sandbox", "sandbox": sandbox_spec(),
              "replay_fraction_failed": 0.0}
    fields.update(overrides)
    return _episode(**fields)


def _sha(contract):
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def test_a_signed_job_parses_and_round_trips():
    raw = _manifest(episode=signed_episode())
    job = parse_job(raw)
    assert is_signed_sandbox(job)
    assert job.episode.sandbox.tools == ("bash", "edit")
    assert job.episode.sandbox.budgets.max_calls == 64
    assert job.to_contract()["episode"] == signed_episode()
    assert parse_job(job.to_contract()) == job
    assert _sha(job.to_contract()) == _sha(raw)


def test_a_replay_job_hashes_as_before():
    raw = _manifest()
    job = parse_job(raw)
    assert job.episode.execution == EXECUTION_REPLAY and job.episode.sandbox is None
    assert not is_signed_sandbox(job)
    assert "execution" not in job.to_contract()["episode"]
    assert _sha(job.to_contract()) == _sha(raw)


def test_replay_is_never_written_explicitly():
    with pytest.raises(JobError, match="only when"):
        parse_job(_manifest(episode=_episode(execution="replay")))


@pytest.mark.parametrize("episode,match", [
    (_episode(execution="signed_sandbox"), "needs episode.sandbox"),
    (_episode(sandbox=sandbox_spec()), "only for execution"),
    (_episode(execution="docker"), "execution must be one of"),
    (signed_episode(replay_fraction_failed=0.1), "never replayed"),
    (signed_episode(sandbox=sandbox_spec(env="reliquary-terminal")), "episode's package"),
    (signed_episode(sandbox=sandbox_spec(env_package="reliquary-swe")), "env_package"),
    (signed_episode(sandbox=sandbox_spec(env_package="other==1+g0123456789abcdef")), "env_package"),
    (signed_episode(sandbox=sandbox_spec(env_package="reliquary-swe==0.1.0a1")), "env_package"),
    (signed_episode(sandbox=sandbox_spec(env_package="reliquary-swe==0.1.0a1+gABCDEF0123456789")), "env_package"),
    (signed_episode(sandbox=sandbox_spec(env_package="reliquary-swe==0.1.0a1+g0123")), "env_package"),
    (signed_episode(sandbox=sandbox_spec(tools=["edit"])), "tools"),
    (signed_episode(sandbox=sandbox_spec(tools=["edit", "bash"])), "tools"),
    (signed_episode(sandbox=sandbox_spec(tools=["bash", "bash"])), "tools"),
    (signed_episode(sandbox=sandbox_spec(tools=["bash", "python"])), "tools"),
    (signed_episode(sandbox=sandbox_spec(sandbox_commit="main")), "sandbox_commit"),
    (signed_episode(sandbox={**sandbox_spec(), "extra": 1}), "unknown fields"),
])
def test_a_bad_signed_episode_is_refused(episode, match):
    with pytest.raises(JobError, match=match):
        parse_job(_manifest(episode=episode))


@pytest.mark.parametrize("field,value", [
    ("max_calls", 0), ("max_calls", 513), ("per_call_timeout_s", 601), ("cpu_s", 3601),
    ("wall_s", 14401), ("memory_bytes", 8 * GIB + 1), ("pids", 1025),
    ("disk_bytes", 10 * GIB + 1), ("wall_s", True), ("pids", 1.5),
])
def test_budgets_stay_within_the_gateways_default_caps(field, value):
    spec = sandbox_spec()
    spec["budgets"] = {**spec["budgets"], field: value}
    with pytest.raises(JobError, match="budgets"):
        parse_job(_manifest(episode=signed_episode(sandbox=spec)))


def test_max_calls_covers_every_turn():
    spec = sandbox_spec()
    spec["budgets"] = {**spec["budgets"], "max_calls": 10}
    with pytest.raises(JobError, match="max_calls"):
        parse_job(_manifest(episode=signed_episode(sandbox=spec, max_turns=40)))


def test_the_bash_only_tool_set_is_allowed():
    job = parse_job(_manifest(episode=signed_episode(sandbox=sandbox_spec(tools=["bash"]))))
    assert job.episode.sandbox.tools == ("bash",)


def test_the_sandbox_split_names_the_image_count():
    assert sandbox_split(parse_job(_manifest(episode=signed_episode())).episode) == "train:20"

_REAL_MANIFEST = (Path(__file__).resolve().parent.parent / "fixtures"
                  / "corpus_job_manifest_code_v1.json")


def test_an_existing_job_contract_still_hashes_identically():
    raw = json.loads(_REAL_MANIFEST.read_text())
    job = parse_job(raw)
    assert job.episode is None and not is_signed_sandbox(job)
    assert _sha(job.to_contract()) == _sha(raw)


def test_an_existing_episode_job_has_no_execution_key():
    raw = _manifest()
    assert "execution" not in raw["episode"] and "sandbox" not in raw["episode"]
    assert _sha(parse_job(raw).to_contract()) == _sha(raw)
