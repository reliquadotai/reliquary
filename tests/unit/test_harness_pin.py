"""This build's bash harness is the reliquary-harness spec it pins, and that spec is the
verifiers it reproduces. Never skipped: the package is a dependency, and the upstream
files are checked whenever verifiers is installed (a miner, the sandbox bridge).

reliquary-harness's own tests hold the spec equal to the vendored verifiers b2e4e81
files; the digest pinned here covers that spec and the code that interprets it."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

from reliquary_harness.spec import BASH_SURFACE_V1, BASH_VF_B2E4E81, SPECS

from reliquary.environment import agentic_swe


def test_the_installed_harnesses_are_the_ones_this_build_pins():
    assert agentic_swe.HARNESS is BASH_VF_B2E4E81
    assert BASH_VF_B2E4E81.digest == agentic_swe.HARNESS_DIGEST
    assert agentic_swe.HARNESS_DIGESTS == {spec.key: spec.digest for spec in SPECS.values()}
    assert agentic_swe.harness_refusal() is None


def test_another_harness_digest_refuses_every_episode_job(monkeypatch):
    monkeypatch.setitem(agentic_swe.HARNESS_DIGESTS, BASH_VF_B2E4E81.key, "0" * 64)
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


def _pin(spec, digest=None):
    return {"id": spec.id, "version": spec.version, "digest": digest or spec.digest}


def test_a_pinned_harness_is_served_only_at_its_digest_and_variants_only_per_lease():
    refusal = agentic_swe.harness_pin_refusal
    assert refusal(None, variants=False) is None
    assert refusal(_pin(BASH_VF_B2E4E81), variants=False) is None
    assert refusal(_pin(BASH_SURFACE_V1), variants=True) is None
    assert "an RL lease's, not a corpus job's" in refusal(_pin(BASH_SURFACE_V1), variants=False)
    assert "this build serves" in refusal(_pin(BASH_SURFACE_V1, "f" * 64), variants=True)
    assert "not one this build serves" in refusal({"id": "openhands", "version": "1", "digest": "0" * 64},
                                                  variants=True)
    assert agentic_swe.harness_of(None) is BASH_VF_B2E4E81
    assert agentic_swe.harness_of(_pin(BASH_SURFACE_V1)) is BASH_SURFACE_V1


def test_a_lease_variant_is_its_nonces_lease_and_prompt_alone():
    from types import SimpleNamespace

    lease = lambda **kw: SimpleNamespace(**{"nonce": "ab" * 32, "lease_id": "L1", "prompt_idx": 7, **kw})
    v = agentic_swe.lease_variant
    assert v(BASH_VF_B2E4E81, SimpleNamespace()) == 0          # no axes: no draw, nothing read
    assert v(BASH_SURFACE_V1, lease()) == v(BASH_SURFACE_V1, lease(checkpoint=3, hotkey="x"))
    draws = {v(BASH_SURFACE_V1, lease(nonce=f"{i:064x}")) for i in range(64)}
    assert draws == {0, 1, 2, 3}
