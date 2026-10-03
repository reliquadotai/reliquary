"""A trajectory from the endpoint's session log: every prompt extends the last."""

import pytest

from reliquary.corpus.trajectory import GeneratedTurn, TrajectoryUnbuildable, build_trajectory
from reliquary.protocol.corpus_submission import CorpusTrajectory
from reliquary.protocol.toploc import span_chunk_count

PROMPT = (10, 11, 12)
CHUNK_TOKENS = 32


def _turns():
    first = GeneratedTurn(PROMPT, (20, 21, 1), ("p1",))
    second_prompt = PROMPT + (20, 21, 1) + (30, 31)              # observation segment
    second = GeneratedTurn(second_prompt, (22, 1), ("p2",))
    return [first, second]


def test_spans_are_in_trajectory_coordinates():
    built = build_trajectory(_turns(), final_diff="d", stop="agent_completed")
    assert built.prompt_ids == PROMPT
    assert built.tokens == (20, 21, 1, 30, 31, 22, 1)
    assert built.spans == ((0, 3), (5, 7)) and built.proofs == (("p1",), ("p2",))
    wire = built.wire()
    CorpusTrajectory.model_validate(wire)
    assert wire["turns"][1] == {"start": 5, "end": 7, "proofs": ["p2"]}


def test_a_re_rendered_history_cannot_be_built():
    first, second = _turns()
    rewritten = GeneratedTurn(PROMPT + (20, 99, 1, 30, 31), second.completion_ids, second.proofs)
    with pytest.raises(TrajectoryUnbuildable, match="extend"):
        build_trajectory([first, rewritten], final_diff="d", stop="agent_completed")


def test_two_turns_need_a_segment_between_them():
    first, _ = _turns()
    glued = GeneratedTurn(PROMPT + (20, 21, 1), (22, 1), ("p2",))
    with pytest.raises(TrajectoryUnbuildable):
        build_trajectory([first, glued], final_diff="d", stop="agent_completed")


@pytest.mark.parametrize("stop", ["error", None, "max_total_tokens"])
def test_only_contract_stops_are_built(stop):
    with pytest.raises(TrajectoryUnbuildable):
        build_trajectory(_turns(), final_diff="d", stop=stop)


def test_no_turns_or_an_empty_completion_cannot_be_built():
    with pytest.raises(TrajectoryUnbuildable):
        build_trajectory([], final_diff="", stop="agent_completed")
    with pytest.raises(TrajectoryUnbuildable):
        build_trajectory([GeneratedTurn(PROMPT, (), ())], final_diff="", stop="agent_completed")


def test_proof_counts_must_match_span_chunk_count():
    """Spans longer than 32 tokens with correct proof counts pass validation."""
    # First turn: 80 tokens → span_chunk_count(80, 32) = 3 proofs
    first_completion = tuple(range(100, 180))  # 80 tokens
    first_proof_count = span_chunk_count(len(first_completion), CHUNK_TOKENS)
    assert first_proof_count == 3, f"Expected 3 proofs for 80 tokens, got {first_proof_count}"
    # Use valid base64 strings for proofs
    first_proofs = tuple("AAAA" * (i + 1) for i in range(first_proof_count))

    first = GeneratedTurn(PROMPT, first_completion, first_proofs)

    # Second turn: 65 tokens, with 2-token observation → span_chunk_count(65, 32) = 2 proofs
    second_prompt = PROMPT + first_completion + (200, 201)  # 2-token observation
    second_completion = tuple(range(300, 365))  # 65 tokens
    second_proof_count = span_chunk_count(len(second_completion), CHUNK_TOKENS)
    assert second_proof_count == 2, f"Expected 2 proofs for 65 tokens, got {second_proof_count}"
    second_proofs = tuple("BBBB" * (i + 1) for i in range(second_proof_count))

    second = GeneratedTurn(second_prompt, second_completion, second_proofs)

    # Build should succeed with correct proof counts
    built = build_trajectory([first, second], final_diff="d", stop="agent_completed")
    assert built.proofs == (first_proofs, second_proofs)

    # wire() should validate against CorpusTrajectory schema with real proofs (no substitution)
    wire = built.wire()
    validated = CorpusTrajectory.model_validate(wire)
    assert len(validated.turns[0].proofs) == 3, f"Expected 3 proofs for first span, got {len(validated.turns[0].proofs)}"
    assert len(validated.turns[1].proofs) == 2, f"Expected 2 proofs for second span, got {len(validated.turns[1].proofs)}"


def test_proof_count_mismatch_is_refused():
    """A turn with proof count not matching span_chunk_count is refused."""
    # 80 tokens need 3 proofs, but we provide only 2
    completion = tuple(range(100, 180))  # 80 tokens
    expected_count = span_chunk_count(len(completion), CHUNK_TOKENS)
    assert expected_count == 3, f"Expected span_chunk_count(80, 32) = 3, got {expected_count}"
    wrong_proofs = tuple("CCCC" * (i + 1) for i in range(expected_count - 1))

    turn = GeneratedTurn(PROMPT, completion, wrong_proofs)

    with pytest.raises(TrajectoryUnbuildable, match="proof"):
        build_trajectory([turn], final_diff="d", stop="agent_completed")
