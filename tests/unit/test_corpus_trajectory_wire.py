"""A trajectory submission: bounded on the wire, bound by its own signature domain."""

import pytest
from pydantic import ValidationError

from reliquary.corpus.job import EPISODE_STOPS, MAX_EPISODE_TOTAL_TOKENS, MAX_EPISODE_TURNS
from reliquary.protocol.corpus_submission import (
    MAX_TRAJECTORY_TOKENS,
    MAX_TRAJECTORY_TURNS,
    CorpusRejectReason,
    CorpusSubmissionRequest,
    CorpusTrajectory,
)
from reliquary.protocol.signatures import (
    CORPUS_DOMAIN,
    CORPUS_TRAJECTORY_DOMAIN,
    build_corpus_binding,
)

PROOF = "A" * 344


def _trajectory(**overrides):
    t = {"tokens": list(range(1, 81)),
         "turns": [{"start": 0, "end": 40, "proofs": [PROOF, PROOF]},
                   {"start": 60, "end": 80, "proofs": [PROOF]}],
         "final_diff": "diff --git a/x b/x\n", "stop": "agent_completed"}
    t.update(overrides)
    return t


def _body(**overrides):
    body = {"job_id": "swe-agentic-v1", "miner_hotkey": "5Hot", "cursor": 3, "prompt_index": 17,
            "checkpoint_sha256": "a" * 64, "rendered_prompt": "<prompt>",
            "trajectory": _trajectory(), "signature": "00"}
    body.update(overrides)
    return body


def test_bounds_match_the_contract():
    assert MAX_TRAJECTORY_TOKENS == MAX_EPISODE_TOTAL_TOKENS
    assert MAX_TRAJECTORY_TURNS == MAX_EPISODE_TURNS
    assert set(CorpusTrajectory.model_fields["stop"].annotation.__args__) == set(EPISODE_STOPS)


def test_a_trajectory_submission_parses_with_no_completions():
    request = CorpusSubmissionRequest(**_body())
    assert request.completions == [] and request.trajectory.turns[1].end == 80


@pytest.mark.parametrize("body", [
    _body(trajectory=None),                                                  # neither
    _body(completions=[{"tokens": [1, 2], "text": "x"}]),                    # both
    _body(trajectory=_trajectory(stop="error")),
    _body(trajectory=_trajectory(tokens=[-1, 2])),
    _body(trajectory=_trajectory(turns=[{"start": 5, "end": 5, "proofs": []}])),
    _body(trajectory=_trajectory(turns=[{"start": 0, "end": 2, "proofs": [PROOF] * 3}])),
    _body(trajectory=_trajectory(tokens=[1] * (MAX_TRAJECTORY_TOKENS + 1))),
    _body(trajectory=_trajectory(final_diff="x" * (1_048_576 + 1))),
])
def test_malformed_trajectories_are_refused(body):
    with pytest.raises(ValidationError):
        CorpusSubmissionRequest(**body)


def test_new_reject_reasons_match_the_pure_layer():
    from reliquary.corpus import checks

    assert CorpusRejectReason.BAD_TURNS.value == checks.REASON_BAD_TURNS
    assert CorpusRejectReason.SHORT_TURNS.value == checks.REASON_SHORT_TURNS
    assert CorpusRejectReason.BAD_OBSERVATION.value == "bad_observation"
    assert CorpusRejectReason.UNANSWERED_TOOL_CALL.value == "unanswered_tool_call"
    assert CorpusRejectReason.BAD_STOP.value == "bad_stop"


def test_a_trajectory_is_signed_under_its_own_domain():
    assert CORPUS_TRAJECTORY_DOMAIN != CORPUS_DOMAIN
    model = CorpusSubmissionRequest(**_body())
    assert build_corpus_binding(model) == build_corpus_binding(_body())   # dict and model agree


@pytest.mark.parametrize("change", [
    {"tokens": list(range(2, 82))},
    {"turns": [{"start": 0, "end": 41, "proofs": [PROOF, PROOF]},
               {"start": 60, "end": 80, "proofs": [PROOF]}]},
    {"turns": [{"start": 0, "end": 40, "proofs": [PROOF, "B" * 344]},
               {"start": 60, "end": 80, "proofs": [PROOF]}]},
    {"final_diff": "diff --git a/y b/y\n"},
    {"stop": "max_turns"},
])
def test_every_trajectory_field_is_bound(change):
    assert build_corpus_binding(_body()) != build_corpus_binding(_body(trajectory=_trajectory(**change)))


def test_a_single_turn_binding_is_unchanged():
    body = {"job_id": "math-v1", "miner_hotkey": "5Hot", "cursor": 3, "prompt_index": 17,
            "checkpoint_sha256": "a" * 64, "rendered_prompt": "<prompt>",
            "completions": [{"tokens": [1, 2, 3], "text": "123", "proofs": []}], "signature": "00"}
    assert build_corpus_binding(body) == build_corpus_binding(CorpusSubmissionRequest(**body))
