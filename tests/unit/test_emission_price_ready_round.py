"""When did supply actually meet the window's target?

This is the numerator of ``r``, and getting it from the wrong set is how the
sensor dies. A fill-closed window closes on PROVEN groups, so reading the close
time would measure our own proof plane (11 proofs/min against the 25.6 the
window wants) and never the market: doubling the miners would not move it by a
second.

What answers the question is the ADMISSIBLE set -- the candidates that passed
every cheap check and would have been proven had there been capacity. The
batcher already keeps exactly that in ``_pending``, whose docstring reads "passed
every CHEAP check and has been graded + scored, but has NOT been proven".
"""

from __future__ import annotations

from reliquary.validator.emission_price import (
    distinct_prompt_arrival_rounds,
    ready_round,
    window_ready_round,
)


def test_the_target_th_arrival_is_what_counts():
    """Supply met the target at the moment the target-th candidate landed."""
    assert ready_round([100, 101, 102, 103], target=3) == 102


def test_arrivals_are_ordered_before_counting():
    """Submissions land out of order; the answer is chronological regardless."""
    assert ready_round([103, 100, 102, 101], target=3) == 102


def test_a_target_never_reached_has_no_ready_round():
    """Fewer admissible candidates than the window wanted: that is the shortage."""
    assert ready_round([100, 101], target=3) is None


def test_the_slowest_environment_decides_the_window():
    """The window is not ready until EVERY environment holds its target.

    Math and code fill at very different rates -- math sealed at 27.7 s against
    code's 43.4 s under the old timed window -- so averaging them would report a
    readiness neither environment had.
    """
    ready = window_ready_round(
        {"openmathinstruct": [100, 101, 102], "opencodeinstruct": [100, 104, 108]},
        {"openmathinstruct": 3, "opencodeinstruct": 3},
    )

    assert ready == 108


def test_one_starved_environment_starves_the_window():
    ready = window_ready_round(
        {"openmathinstruct": [100, 101, 102], "opencodeinstruct": [100]},
        {"openmathinstruct": 3, "opencodeinstruct": 3},
    )

    assert ready is None


def test_an_environment_with_no_target_is_not_counted():
    """A window carries only the environments it actually ran."""
    ready = window_ready_round(
        {"openmathinstruct": [100, 101, 102]},
        {"openmathinstruct": 3},
    )

    assert ready == 102


def test_a_window_with_no_environments_has_no_ready_round():
    """Degenerate, but it must read as silence rather than raise.

    The replay walks whatever the archive holds; a record that names no
    environment has to fall through to "no signal" like any other unusable one.
    """
    assert window_ready_round({}, {}) is None


def test_a_prompt_arrives_when_its_FIRST_admissible_candidate_does():
    """The target counts distinct groups, so re-submissions do not advance it.

    A window wants 256 proven groups per environment, and a group is one
    prompt. MAX_SUBMISSIONS_PER_PROMPT lets ten candidates chase the same
    prompt; counting them all would report supply that the window cannot use.
    """
    rounds = distinct_prompt_arrival_rounds({7: [105, 100, 102], 9: [101]})

    assert sorted(rounds) == [100, 101]


def test_a_prompt_with_no_admissible_candidate_is_absent():
    """An empty bucket is a prompt nothing usable ever landed on."""
    assert distinct_prompt_arrival_rounds({7: [], 9: [101]}) == [101]


def test_readiness_composes_over_distinct_prompts():
    """The two halves meet here: distinct prompts, then the target-th of them."""
    arrivals = {1: [100, 100], 2: [140], 3: [120]}

    assert ready_round(distinct_prompt_arrival_rounds(arrivals), target=2) == 120
