"""reliquary-sandbox is an optional extra pinned by one commit."""

import re
import tomllib
from pathlib import Path

import pytest

from reliquary import sandbox

ROOT = Path(__file__).resolve().parents[2]


def test_every_sandbox_extra_pins_the_same_commit_as_the_code():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    pins = set()
    for name in ("sandbox", "sandbox-miner", "sandbox-test"):
        (spec,) = extras[name]
        assert spec.startswith("reliquary-sandbox")
        pins.add(re.search(r"@([0-9a-f]{40})$", spec).group(1))
    assert pins == {sandbox.SANDBOX_COMMIT}


def test_the_miner_extra_brings_the_verifiers_bridge():
    extras = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["optional-dependencies"]
    assert extras["sandbox-miner"][0].startswith("reliquary-sandbox[verifiers]")


def test_a_missing_sandbox_names_the_extra(monkeypatch):
    def missing():
        raise ImportError("No module named 'reliquary_sandbox'")

    monkeypatch.setattr(sandbox, "_import_sandbox", missing)
    with pytest.raises(sandbox.SandboxUnavailable, match=r"reliquary\[sandbox\]"):
        sandbox.require_sandbox()


def test_an_installed_sandbox_passes():
    pytest.importorskip("reliquary_sandbox.attest")
    sandbox.require_sandbox()


def test_another_sandbox_commit_is_refused(monkeypatch):
    monkeypatch.setattr(sandbox, "_installed_commit", lambda: "0" * 40)
    assert "0000000" in sandbox.sandbox_commit_refusal(sandbox.SANDBOX_COMMIT)
    monkeypatch.setattr(sandbox, "_installed_commit", lambda: sandbox.SANDBOX_COMMIT)
    assert sandbox.sandbox_commit_refusal(sandbox.SANDBOX_COMMIT) is None
    assert "job pins" in sandbox.sandbox_commit_refusal("1" * 40)


def test_a_machine_advertising_a_transcript_cap_over_the_wire_cap_is_refused():
    """A machine whose signed transcripts may exceed the submission body's transcript
    cap would sign honest episodes no miner can submit: it is not placed."""
    from reliquary.protocol.corpus_submission import MAX_TRANSCRIPT_BYTES

    assert sandbox.transcript_cap_refusal({"max_transcript_bytes": MAX_TRANSCRIPT_BYTES}) is None
    assert sandbox.transcript_cap_refusal({"max_transcript_bytes": 256 * 1024}) is None
    assert "over" in sandbox.transcript_cap_refusal(
        {"max_transcript_bytes": MAX_TRANSCRIPT_BYTES + 1})
    for caps in ({}, {"max_transcript_bytes": "8"}, {"max_transcript_bytes": True},
                 {"max_transcript_bytes": 0}, None):
        assert sandbox.transcript_cap_refusal(caps) is not None
