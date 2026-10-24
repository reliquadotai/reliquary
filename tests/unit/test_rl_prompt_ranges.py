import pytest

from reliquary.eval.sets import CORPUS_RANGES, HELD_OUT, UsedRange, overlaps
from reliquary.protocol.profiles import PROFILES
from reliquary.protocol.rl_prompt_ranges import (
    RL_PROMPT_RANGES, RL_REUSES_CORPUS_ROWS, PromptRows, parse_prompt_rows, rl_prompt_range,
)

pytestmark = pytest.mark.usefixtures("real_rl_prompt_ranges")

_SUITE = PROFILES["teutonic-9b-reliquary-suite-v9-dev1"]


def test_reserved_rows_are_cut_and_other_envs_are_whole():
    assert rl_prompt_range("reliquary_hard_math_v1", 23_227) == (15_749, 23_227)
    assert rl_prompt_range("reliquary_instruction_following_v1", 29_435) == (20_000, 29_435)
    assert rl_prompt_range("reliquary_code_v1", 2_481_806) == (200_000, 2_381_806)
    assert rl_prompt_range("reliquary_dapo_math_v1", 13_931) == (0, 13_931)
    # v1.1 appends SFT rows after the RL share: RL keeps [4181, 6899).
    assert rl_prompt_range("reliquary_competitive_code_v1", 8_792) == (4_181, 6_899)


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


# --- an episode env's declared rows (PromptRows) ----------------------------------------------------------


def test_declared_rows_are_a_slice_minus_rows_and_intervals():
    rows = parse_prompt_rows({"start": 10, "end": 100, "exclude": [50, [20, 30], [90, 100], 31]}, "env.rows")
    assert rows == PromptRows(10, 100, ((20, 30), (31, 32), (50, 51), (90, 100)))
    assert len(rows) == 90 - 10 - 1 - 1 - 10
    inside = [r for r in range(120) if r in rows]
    assert inside == [*range(10, 20), 30, *range(32, 50), *range(51, 90)]
    assert True not in rows and "11" not in rows                       # only integers are rows
    assert parse_prompt_rows({"start": 0, "end": 5}, "w") == PromptRows(0, 5) == PromptRows.of((0, 5))


@pytest.mark.parametrize("value, match", [
    ({"start": 0}, "start, end"),
    ({"start": 0, "end": 5, "extra": 1}, "start, end"),
    ({"start": 5, "end": 5}, "non-empty"),
    ({"start": -1, "end": 5}, "non-negative"),
    ({"start": 0, "end": 5.0}, "non-negative"),
    ({"start": 0, "end": True}, "non-negative"),
    ({"start": 0, "end": 10, "exclude": [10]}, "outside"),
    ({"start": 2, "end": 10, "exclude": [[0, 3]]}, "outside"),
    ({"start": 0, "end": 10, "exclude": [[4, 4]]}, "outside"),
    ({"start": 0, "end": 10, "exclude": [[1, 2, 3]]}, "interval"),
    ({"start": 0, "end": 10, "exclude": [[1, 5], 3]}, "overlapping"),
    ({"start": 0, "end": 10, "exclude": [2, 2]}, "overlapping"),
    ({"start": 0, "end": 10, "exclude": [[0, 10]]}, "every row"),
    ({"start": 0, "end": 10, "exclude": "3"}, "list"),
])
def test_malformed_rows_are_refused(value, match):
    with pytest.raises(ValueError, match=match):
        parse_prompt_rows(value, "env.rows")
