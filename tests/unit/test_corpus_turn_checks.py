"""N3: per-turn checks, pure, applied to every trajectory submission."""

import pytest

from reliquary.corpus.admission import admit
from reliquary.corpus.checks import (
    check_short_turns,
    check_turn_budget,
    check_turn_proof_shape,
    check_turn_spans,
    check_turn_termination,
)
from reliquary.corpus.job import parse_job
from reliquary.corpus.slots import SlotLedger
from reliquary.corpus.walk import CursorLedger
from tests.unit.test_corpus_job_episode import _manifest

TERM, EOT = 1, 2


@pytest.mark.parametrize("spans,ok", [
    ([(0, 30)], True),
    ([(0, 10), (15, 30)], True),
    ([], False),
    ([(1, 10)], False),                    # the first turn follows the prompt directly
    ([(0, 10), (10, 30)], False),          # no segment between two turns
    ([(0, 10), (8, 30)], False),
    ([(0, 10), (15, 29)], False),          # does not end the tokens
    ([(0, 10), (15, 31)], False),
])
def test_turn_spans(spans, ok):
    assert check_turn_spans(spans, 30, max_turns=40).ok is ok


def test_too_many_turns():
    spans = [(k * 10, k * 10 + 5) for k in range(3)]
    result = check_turn_spans(spans, 25, max_turns=2)
    assert (result.ok, result.reason) == (False, "bad_turns")


def test_short_turns_are_bounded():
    assert check_short_turns([(0, 3), (10, 50), (60, 64)]).ok
    result = check_short_turns([(0, 3), (10, 12), (20, 50), (60, 64)])
    assert (result.ok, result.reason) == (False, "short_turns")


def test_budget():
    assert check_turn_budget([(0, 100)], prompt_len=50, length=100,
                             max_tokens_per_turn=100, max_total_tokens=150).ok
    assert not check_turn_budget([(0, 101)], prompt_len=50, length=101,
                                 max_tokens_per_turn=100, max_total_tokens=1000).ok
    result = check_turn_budget([(0, 100)], prompt_len=51, length=100,
                               max_tokens_per_turn=100, max_total_tokens=150)
    assert (result.ok, result.reason) == (False, "token_budget_exceeded")


def _termination(tokens, spans, prompt_len=10, per_turn=8, total=1000):
    return check_turn_termination(tokens, spans, prompt_len=prompt_len, terminator_id=TERM,
                                  stop_ids=frozenset({TERM, EOT}), max_tokens_per_turn=per_turn,
                                  max_total_tokens=total)


def test_turns_end_on_the_terminator():
    tokens = [7, 7, TERM, 9, 9, 7, EOT]
    assert _termination(tokens, [(0, 3), (5, 7)]).ok
    assert not _termination([7, 7, EOT, 9, 9, 7, EOT], [(0, 3), (5, 7)]).ok   # eos mid-trajectory


def test_a_capped_non_final_turn_is_accepted():
    tokens = [7] * 8 + [TERM, 9] + [7, TERM]
    assert _termination(tokens, [(0, 8), (10, 12)]).ok


def test_the_cap_shrinks_near_the_total_budget():
    tokens = [7] * 5
    assert _termination(tokens, [(0, 5)], prompt_len=10, total=15).ok          # 15 - 10 = 5
    result = _termination(tokens, [(0, 5)], prompt_len=10, total=16)
    assert (result.ok, result.reason) == (False, "bad_termination")


def test_proof_shape_uses_span_chunking():
    assert check_turn_proof_shape([(0, 35), (40, 80)], [1, 2], 32).ok          # 35 merges, 40 -> 32+8
    result = check_turn_proof_shape([(0, 35)], [2], 32)
    assert (result.ok, result.reason) == (False, "bad_proof_shape")
    assert not check_turn_proof_shape([(0, 35)], [1, 1], 32).ok


def _admit(job, **kw):
    args = dict(hotkey="5Hot", cursor=0, prompt_index=3, checkpoint_sha256=job.checkpoint_sha256,
                token_counts=[500], last_token_ids=[TERM], digests=["d" * 64],
                slots=SlotLedger(job.prompt_count, job.slots_per_prompt), cursors=CursorLedger(),
                seen=set())
    args.update(kw)
    return admit(job, **args)


def test_an_episode_job_is_admitted_only_once_its_trajectory_was_checked():
    job = parse_job(_manifest())
    assert _admit(job).reason == "malformed_submission"
    verdict = _admit(job, episode_checked=True)
    assert verdict.accepted and verdict.slots_remaining == 1


def test_a_duplicate_trajectory_is_still_refused():
    job = parse_job(_manifest())
    assert _admit(job, episode_checked=True, seen={"d" * 64}).reason == "hash_duplicate"
