from copy import deepcopy

import pytest

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.observations import observation_signal
from tests.unit.service_v1_fixtures import contract, observation


def test_grader_error_does_not_become_zero_signal():
    c=contract()
    row=observation(c,rewards=(0,0));row["verification"]["grading"]="error"
    assert observation_signal(row,c).category == "unknown"
