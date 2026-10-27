"""The fill-closed backstop has to outlive a healthy window, not cut it short.

1800 seconds did not stay a backstop. Production had to cut picks per window
from the profile's 16 to 7 to fit under it, which buys the deadline with more
than half the training signal a window could carry — a bound met by shrinking
the work is setting the cadence, which is what a backstop is not for.

It could not be raised alone. The coherence check asks for `backstop * 2 <
window timeout` whenever proofs are unbounded, which is how production runs, so
1800 against 7200 was exactly half the budget. The wall is also the binding one
operationally: the backstop reads an environment variable and can be tuned on a
running fleet, while the wall is a literal, so leaving it low is what made 1800
unraisable without a deploy.
"""

import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest


def _constants(**environment):
    # A refused reload can leave partially assigned constants in other tests.
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,reliquary.constants as c;print(json.dumps({"
            "'FILL_CLOSED_MAX_SECONDS':c.FILL_CLOSED_MAX_SECONDS,"
            "'WINDOW_TIMEOUT_SECONDS':c.WINDOW_TIMEOUT_SECONDS}))",
        ],
        env={**os.environ, **environment},
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return SimpleNamespace(**json.loads(result.stdout))


def test_the_backstop_clears_the_next_policy_by_a_wide_margin() -> None:
    """Six hours, because the next policy is larger on every axis at once:
    roughly twice the parameters, up to four times the token ceiling on maths,
    and a multi-turn environment whose rollouts are conversations. Guessing the
    product of those and cutting it fine would repeat the mistake."""
    assert _constants().FILL_CLOSED_MAX_SECONDS == 21600.0


def test_the_ratio_that_binds_them_is_preserved() -> None:
    """Unbounded proofs may need a second pass, which is what the factor of two
    pays for. Keeping room above it makes this a rescale, not a loosening."""
    constants = _constants()
    assert constants.FILL_CLOSED_MAX_SECONDS * 2 < constants.WINDOW_TIMEOUT_SECONDS


def test_the_wall_leaves_room_to_tune_without_a_deploy() -> None:
    """The reason the wall moves at all. The backstop is env-overridable and
    the wall is not, so the wall decides how far a running fleet can be tuned:
    anything up to half of it starts, anything above does not."""
    constants = _constants()
    tunable = constants.WINDOW_TIMEOUT_SECONDS // 2
    assert tunable >= 4 * 1800
    assert (
        _constants(
            RELIQUARY_FILL_CLOSED_MAX_SECONDS=str(tunable - 1),
        ).FILL_CLOSED_MAX_SECONDS
        == tunable - 1
    )


# The refusal above half the wall is covered by
# `test_strict_fill_closed_deadline_must_fit_the_service_timeout` in
# test_fill_closed_profile.py, which derives the same value and runs it in a
# fresh interpreter — the profile is resolved at import of
# `reliquary.protocol.profiles`, so reloading `constants` cannot change it.


def test_a_non_positive_value_is_still_refused() -> None:
    import reliquary.constants as constants

    before = constants.FILL_CLOSED_MAX_SECONDS
    with pytest.raises(subprocess.CalledProcessError) as refused:
        _constants(RELIQUARY_FILL_CLOSED_MAX_SECONDS="0")
    assert "must be positive" in refused.value.stderr
    assert constants.FILL_CLOSED_MAX_SECONDS == before
