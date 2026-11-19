"""bittensor sets every logger existing when it is imported to CRITICAL; Reliquary's are
given back to the hierarchy so operator ALERTs reach the logs."""

import logging

from reliquary.shared.log_levels import unmute_reliquary_loggers


def test_muted_reliquary_loggers_are_given_back_and_others_left_alone():
    muted = logging.getLogger("reliquary.validator.test_log_levels_probe")
    own_level = logging.getLogger("reliquary.validator.test_log_levels_warning")
    foreign = logging.getLogger("thirdparty.test_log_levels_probe")
    try:
        muted.setLevel(logging.CRITICAL)            # what bittensor's default logging does
        own_level.setLevel(logging.WARNING)
        foreign.setLevel(logging.CRITICAL)
        freed = unmute_reliquary_loggers()
        assert "reliquary.validator.test_log_levels_probe" in freed
        assert muted.level == logging.NOTSET
        assert own_level.level == logging.WARNING   # a level set on purpose is kept
        assert foreign.level == logging.CRITICAL
    finally:
        for logger in (muted, own_level, foreign):
            logger.setLevel(logging.NOTSET)


def test_bittensors_default_logging_mutes_what_was_imported_before_it():
    """The mechanism, as bittensor runs it (skipped without bittensor)."""
    import pytest

    btlogging = pytest.importorskip("bittensor.utils.btlogging")
    probe = logging.getLogger("reliquary.validator.test_log_levels_order")
    every = [l for l in logging.root.manager.loggerDict.values() if isinstance(l, logging.Logger)]
    saved = [(logger, logger.level) for logger in every]
    try:
        btlogging.logging.before_enable_default()   # the transition its import runs
        assert probe.level == logging.CRITICAL
        unmute_reliquary_loggers()
        assert probe.level == logging.NOTSET
    finally:
        for logger, level in saved:                 # the rest of the session is untouched
            logger.setLevel(level)


def test_the_async_trainer_keeps_its_info_lines_when_bittensor_comes_after_it():
    """T12's trainer log had no ``reliquary.trainer.async_trainer`` line: the trainer modules were imported before
    the checkpoint sync imported bittensor. ``settle_logging`` (the entry point's first call) imports bittensor
    before anything else, then gives the muted loggers back; a later sweep catches any other import order."""
    import subprocess
    import sys

    import pytest

    pytest.importorskip("bittensor")
    script = (
        "import logging\n"
        "logging.basicConfig(level=logging.INFO, format='%(name)s|%(message)s')\n"
        "from reliquary.trainer import async_trainer\n"
        "from reliquary.trainer.async_cli import settle_logging\n"
        "settle_logging()\n"
        "import bittensor\n"
        "async_trainer.logger.info('probe-after-settle')\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert "reliquary.trainer.async_trainer|probe-after-settle" in out.stderr
