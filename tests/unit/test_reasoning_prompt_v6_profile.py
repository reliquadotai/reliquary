"""Protocol v6: the Code prompt stops pinning the answer to the *last* fence.

v6 exists because the graded span changes (see
tests/unit/test_code_entry_extraction.py). Two things travel with that change:

* the Code prompt drops "in the last fenced Python code block" — that clause
  only ever existed to compensate ``matches[-1]``, and it is the constraint the
  model obeyed roughly half the time at −1.2σ per miss;
* the Math prompt does NOT move. Math has no extraction defect (its length and
  reward both rise under v5), and re-rendering its prompt would hand it the same
  transient reward dip the v5 cutover cost — 0.622 → 0.275, recovered over ~70
  windows — for no benefit.
"""

import pytest

from reliquary.protocol import profiles


V5 = "qwen3-4b-base-dapo-reasoning-v5"
V6 = "qwen3-4b-base-dapo-reasoning-v6"


def _profile(profile_id):
    return profiles.PROFILES[profile_id]


def test_v6_profile_is_registered():
    assert V6 in profiles.PROFILES
    assert _profile(V6).protocol_version == 6


def test_v6_math_prompt_is_byte_identical_to_v5():
    """Math is healthy under v5. Moving its prompt would cost a reward dip for nothing."""
    v5_math = _profile(V5).environments["openmathinstruct"].prompt_template
    v6_math = _profile(V6).environments["openmathinstruct"].prompt_template

    assert v6_math.template == v5_math.template
    assert v6_math.template_id == v5_math.template_id


def test_v6_code_prompt_drops_the_last_fenced_block_clause():
    """The clause compensated matches[-1]; v6 grades by definition, not position."""
    v6_code = _profile(V6).environments["opencodeinstruct"].prompt_template

    assert "last fenced" not in v6_code.template
    assert "Python code block" in v6_code.template


def test_v6_code_prompt_asks_for_a_complete_implementation():
    """Targets the rollouts that call helpers they never wrote."""
    v6_code = _profile(V6).environments["opencodeinstruct"].prompt_template

    assert "complete implementation" in v6_code.template


def test_v6_code_prompt_keeps_the_reasoning_cue():
    v6_code = _profile(V6).environments["opencodeinstruct"].prompt_template

    assert "step by step" in v6_code.template
    assert "reasoning" in v6_code.template


def test_v6_code_template_id_is_distinct_from_v5():
    """Template IDs are immutable by convention; a changed body needs a new ID."""
    v5_code = _profile(V5).environments["opencodeinstruct"].prompt_template
    v6_code = _profile(V6).environments["opencodeinstruct"].prompt_template

    assert v6_code.template_id != v5_code.template_id
    assert v6_code.template != v5_code.template


def test_v6_carries_the_v5_substrate_unchanged():
    """Only the graded span and the Code prompt move. Model, sampling, budgets stay."""
    v5, v6 = _profile(V5), _profile(V6)

    assert v6.model_id == v5.model_id
    assert v6.model_revision == v5.model_revision
    assert v6.prompt_encoding == v5.prompt_encoding
    assert v6.sampling == v5.sampling
    assert v6.collection_seconds == v5.collection_seconds
    assert v6.upload_grace_seconds == v5.upload_grace_seconds
    assert v6.throughput_tiebreak == v5.throughput_tiebreak
    for name in ("openmathinstruct", "opencodeinstruct"):
        assert v6.environments[name].max_new_tokens == v5.environments[name].max_new_tokens
        assert v6.environments[name].bft == v5.environments[name].bft
        assert v6.environments[name].answer_format == v5.environments[name].answer_format


def test_v5_code_prompt_is_frozen_as_the_historical_control():
    """v4 is the no-cue control and v5 the last-fence control; neither may drift."""
    v5_code = _profile(V5).environments["opencodeinstruct"].prompt_template

    assert v5_code.template_id == "opencodeinstruct-step-by-step-v1"
    assert "last fenced Python code block" in v5_code.template


def test_v6_generation_contract_differs_from_v5():
    """Checkpoint lineage hashes the contract; a v5 checkpoint must not pass as v6."""
    assert _profile(V6).to_generation_contract() != _profile(V5).to_generation_contract()


def test_v6_resolves_by_id():
    assert profiles.resolve_protocol_profile(V6).profile_id == V6
