import pytest
from reliquary.services.scoring import classify_signal, weighted_reward


def test_fractional_signal_and_unknown_are_distinct():
    assert classify_signal([0.4,0.6],expected=2,sigma_min_bps=2400).category == "diverse-below-threshold"
    signal = classify_signal([0,0.5,1,0.5],expected=4,sigma_min_bps=2400)
    assert signal.in_zone and signal.k is None
    assert classify_signal([None,1],expected=2,sigma_min_bps=2400).category == "unknown"
    assert classify_signal([0],expected=2,sigma_min_bps=2400).k is None
    assert weighted_reward({"a":0.5,"b":1},{"a":5000,"b":5000}) == 0.75
    assert weighted_reward({"a":None},{"a":10000}) is None
    with pytest.raises(ValueError):
        weighted_reward({"a":float("nan")},{"a":10000})
