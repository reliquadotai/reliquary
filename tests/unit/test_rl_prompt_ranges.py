import pytest

from reliquary.eval.sets import CORPUS_RANGES, HELD_OUT, UsedRange, overlaps
from reliquary.protocol.profiles import PROFILES
from reliquary.protocol.rl_prompt_ranges import (
    RL_PROMPT_RANGES, RL_REUSES_CORPUS_ROWS, rl_prompt_range,
)

pytestmark = pytest.mark.usefixtures("real_rl_prompt_ranges")

_SUITE = PROFILES["teutonic-9b-reliquary-suite-v9-dev1"]


def test_reserved_rows_are_cut_and_other_envs_are_whole():
    assert rl_prompt_range("reliquary_hard_math_v1", 23_227) == (15_749, 23_227)
    assert rl_prompt_range("reliquary_instruction_following_v1", 29_435) == (20_000, 29_435)
    assert rl_prompt_range("reliquary_code_v1", 2_481_806) == (200_000, 2_381_806)
    assert rl_prompt_range("reliquary_dapo_math_v1", 13_931) == (0, 13_931)


def test_a_split_of_another_length_fails_closed():
    with pytest.raises(ValueError, match="cut against"):
        rl_prompt_range("reliquary_hard_math_v1", 23_228)


def _rl(source):
    entry = RL_PROMPT_RANGES.get(source)
    if entry is None:
        return UsedRange("rl", source, "train", 0, None, "rl")
    end = entry.source_length if entry.end is None else entry.end
    return UsedRange("rl", source, "train", entry.start, end - entry.start, "rl")


def test_the_suite_never_offers_rl_a_row_a_corpus_job_distilled():
    """Every environment of the run, against every corpus job on its source,
    except the ones recorded as knowingly reusing corpus rows."""
    reused = set()
    for source in _SUITE.environments:
        rl = _rl(source)
        for job in CORPUS_RANGES:
            if job.source == source and overlaps(rl, job):
                reused.add(source)
    assert reused == set(RL_REUSES_CORPUS_ROWS)


def test_rl_never_offers_the_code_evaluation_rows():
    assert not overlaps(_rl("reliquary_code_v1"), HELD_OUT["code"].region)
