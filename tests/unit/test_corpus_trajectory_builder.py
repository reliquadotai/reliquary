"""A trajectory from the endpoint's session log: every prompt extends the last."""

import pytest

from reliquary.corpus.trajectory import GeneratedTurn, TrajectoryUnbuildable, build_trajectory
from reliquary.protocol.corpus_submission import CorpusTrajectory

PROMPT = (10, 11, 12)


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
    CorpusTrajectory.model_validate({**wire, "turns": [{**t, "proofs": ["A" * 8]} for t in wire["turns"]]})
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
