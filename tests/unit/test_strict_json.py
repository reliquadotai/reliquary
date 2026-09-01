from __future__ import annotations

import json

import pytest

from reliquary.shared.strict_json import strict_json_loads


def test_strict_json_rejects_nested_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate JSON key: value"):
        strict_json_loads(b'{"outer":{"value":1,"value":2}}')


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_strict_json_rejects_nonfinite_constants(constant: str) -> None:
    with pytest.raises(ValueError, match="non-finite JSON constant"):
        strict_json_loads(f'{{"value":{constant}}}')


@pytest.mark.parametrize("number", ["1e999", "-1e999"])
def test_strict_json_rejects_float_overflow(number: str) -> None:
    with pytest.raises(ValueError, match="non-finite JSON number"):
        strict_json_loads(f'{{"value":{number}}}')


def test_strict_json_retains_finite_floats() -> None:
    assert strict_json_loads(b'{"value":1.25e2}') == {"value": 125.0}


def test_strict_json_preserves_decode_errors() -> None:
    with pytest.raises(UnicodeDecodeError):
        strict_json_loads(b'"\xff"')
    with pytest.raises(json.JSONDecodeError):
        strict_json_loads(b'{"value":}')
