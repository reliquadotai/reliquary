"""This build's bash harness is the reliquary-harness spec it pins, and that spec is the
verifiers it reproduces. Never skipped: the package is a dependency, and the upstream
files are checked whenever verifiers is installed (a miner, the sandbox bridge).

reliquary-harness's own tests hold the spec equal to the vendored verifiers b2e4e81
files; the digest pinned here covers that spec and the code that interprets it."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from reliquary_harness.spec import BASH_VF_B2E4E81

from reliquary.environment import agentic_swe


def test_the_installed_harness_is_the_one_this_build_pins():
    assert agentic_swe.HARNESS is BASH_VF_B2E4E81
    assert BASH_VF_B2E4E81.digest == agentic_swe.HARNESS_DIGEST
    assert agentic_swe.harness_refusal() is None


def test_another_harness_digest_refuses_every_episode_job(monkeypatch):
    monkeypatch.setattr(agentic_swe, "HARNESS_DIGEST", "0" * 64)
    refusal = agentic_swe.harness_refusal()
    assert refusal and "this build needs" in refusal
    assert agentic_swe.episode_support_refusal(object(), need_verifiers=False) == refusal


def test_the_names_this_module_exports_are_the_specs():
    spec = BASH_VF_B2E4E81
    assert agentic_swe.BASH_HARNESS_TOOLS == spec.tools
    assert agentic_swe.BASH_SYSTEM_PROMPT == spec.system_prompt_for(("bash", "edit"))
    assert agentic_swe.PINNED_NETWORK_NOTICE == spec.network_notice["text"]
    assert agentic_swe.SUPPORTED_RENDERER == spec.renderer == "renderers:qwen38@0.1.11"
    assert agentic_swe.SUPPORTED_VERIFIERS == spec.upstream["verifiers"]


def test_an_installed_verifiers_is_the_one_the_spec_reproduces():
    found = importlib.util.find_spec("verifiers")
    if found is None:
        return                      # this process runs no verifiers code to drift from
    root = Path(found.origin).resolve().parent
    for relative, sha in BASH_VF_B2E4E81.upstream["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == sha, relative
