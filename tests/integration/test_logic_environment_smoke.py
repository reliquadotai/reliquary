"""``reliquarylogic_v1`` through mix resolution, admission, seal and trainer.

The unit tests prove the generators and checkers are correct in isolation.
This proves the environment actually fits the chain: that its manifest and
contract resolve, that admission overwrites the miner's claimed reward, that
a sealed batch carries the right provenance into the archive, and that the
training payload round-trips.

It deliberately drives one equality-checked family and one
constraint-checked family through the same path, because those take
different branches inside ``check_answer`` and only the first was ever
exercised end to end.
"""

from __future__ import annotations

import hashlib
import json
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from reliquary import constants as C
from reliquary.environment.logic_tasks import generate_logic_task
from reliquary.environment.registry import (
    get_environment_spec, resolve_environment_mix,
)
from reliquary.protocol.profiles import resolve_protocol_profile
from reliquary.validator.admission import AdmissionContext


ENVIRONMENT = "reliquarylogic_v1"
PROFILE = "qwen3-4b-reliquary-logic-v8-dev1"


class _Wallet:
    class hotkey:
        ss58_address = "validator-test"

        @staticmethod
        def sign(data: bytes) -> bytes:
            return hashlib.sha256(data).digest()


class _Tokenizer:
    eos_token_id = None

    def decode(self, _tokens):
        return ""


class _Model:
    class config:
        vocab_size = 100_000
        max_position_embeddings = 4096

    def eval(self):
        return self

    def parameters(self):
        return []

    def gradient_checkpointing_enable(self):
        return None


def _index_for(check: str) -> int:
    """First index whose family uses the given reward check."""
    for index in range(20_000):
        if generate_logic_task(index).check == check:
            return index
    raise AssertionError(f"no active family uses {check!r}")


def test_profile_declares_the_environment_on_the_cpu_lane():
    profile = resolve_protocol_profile(PROFILE)
    mix = resolve_environment_mix(
        [ENVIRONMENT],
        profile_environments=profile.environments,
        default_batch_target=C.B_BATCH,
    )
    assert mix == [(ENVIRONMENT, 16)]
    spec = get_environment_spec(ENVIRONMENT)
    # A logic puzzle must never start the code grader.
    assert spec.admission_resource_class == "cpu"
    assert spec.validator_authoritative_reward is True


