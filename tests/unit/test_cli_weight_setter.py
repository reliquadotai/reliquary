"""The trainer owns its setter's failure and shutdown lifecycle."""

import asyncio
import threading
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("failure", [True, False])
async def test_stopped_weight_setter_cancels_the_trainer(monkeypatch, failure):
    from reliquary.cli.main import _run_validator_with_weight_setter
    from reliquary.validator import weight_only

    cleaned = asyncio.Event()

    class Worker:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            if failure:
                raise ValueError("setter initialization failed")

    async def train(subtensor):
        try:
            await asyncio.Event().wait()
        finally:
            cleaned.set()

    monkeypatch.setattr(weight_only, "WeightOnlyValidator", Worker)
    with pytest.raises(RuntimeError, match="weight-setter stopped") as raised:
        await asyncio.wait_for(_run_validator_with_weight_setter(
            SimpleNamespace(run=train), None, wallet=None, netuid=0, signer_client=None,
        ), timeout=2)
    assert cleaned.is_set()
    assert isinstance(raised.value.__cause__, ValueError) is failure
    assert not any(thread.name == "weight-setter" for thread in threading.enumerate())


@pytest.mark.parametrize("fatal", [False, True])
async def test_trainer_shutdown_cancels_and_joins_its_setter(monkeypatch, fatal):
    from reliquary.cli.main import _run_validator_with_weight_setter
    from reliquary.validator.errors import FatalProofPlaneError
    from reliquary.validator import weight_only

    started = threading.Event()
    stopped = threading.Event()

    class Worker:
        def __init__(self, **kwargs):
            pass

        async def run(self):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                stopped.set()

    async def train(subtensor):
        assert await asyncio.to_thread(started.wait, 1)
        if fatal:
            raise FatalProofPlaneError("proof cleanup completed")

    monkeypatch.setattr(weight_only, "WeightOnlyValidator", Worker)
    operation = _run_validator_with_weight_setter(
        SimpleNamespace(run=train), None, wallet=None, netuid=0, signer_client=None,
    )
    if fatal:
        with pytest.raises(FatalProofPlaneError):
            await asyncio.wait_for(operation, timeout=2)
    else:
        await asyncio.wait_for(operation, timeout=2)
    assert stopped.is_set()
    assert not any(thread.name == "weight-setter" for thread in threading.enumerate())
