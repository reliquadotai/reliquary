"""Keep Reliquary's own log lines visible under bittensor.

bittensor enables its "default" logging when it is imported: every logger that exists at
that moment and is not one of its own is set to CRITICAL (`LoggingMachine.
before_enable_default`). Whatever Reliquary module was imported before bittensor is
then silent below CRITICAL for the life of the process: measured on a corpus
validator, `reliquary.validator.corpus_validator` (its operator ALERTs among them),
`reliquary.infrastructure.storage`, `reliquary.sandbox.fleet` and `reliquary.cli.main`
logged nothing. Which modules are hit depends only on import order.

`unmute_reliquary_loggers()` gives those loggers back to the hierarchy (level NOTSET:
the root's level and handlers decide, as for every other Reliquary logger). Reliquary
sets no logger to CRITICAL itself, so only bittensor's are undone. Call it after
bittensor is imported and logging is configured (the CLI's `setup_logging`, a process
entry point).
"""

from __future__ import annotations

import logging


def unmute_reliquary_loggers() -> list[str]:
    """Reset every `reliquary` logger left at CRITICAL to NOTSET; returns their names."""
    freed = []
    for name, logger in list(logging.root.manager.loggerDict.items()):
        if not isinstance(logger, logging.Logger):
            continue
        if (name == "reliquary" or name.startswith("reliquary.")) and logger.level == logging.CRITICAL:
            logger.setLevel(logging.NOTSET)
            freed.append(name)
    return sorted(freed)


__all__ = ["unmute_reliquary_loggers"]
