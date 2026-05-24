"""Tests for the OpenCodeInstruct subset filter pipeline.

The filter functions are pure (no HF / no network), tested directly.
The push-to-Hub side is exercised manually when running the script.
"""

import pytest


def test_keep_row_filters_low_test_score():
    from scripts.build_opencodeinstruct_subset import keep_row
    row = {"average_test_score": 0.9, "unit_tests": "[]", "input": "p", "output": "x"}
    assert keep_row(row) is False


def test_keep_row_accepts_perfect_score():
    from scripts.build_opencodeinstruct_subset import keep_row
    row = {
        "average_test_score": 1.0,
        "unit_tests": '["assert f(1) == 1"]',
        "input": "p", "output": "x",
    }
    assert keep_row(row) is True


def test_parse_unit_tests_handles_string_list():
    from scripts.build_opencodeinstruct_subset import parse_unit_tests
    raw = '["assert f(1) == 1", "assert f(2) == 2"]'
    assert parse_unit_tests(raw) == ["assert f(1) == 1", "assert f(2) == 2"]


def test_parse_unit_tests_returns_none_on_garbage():
    from scripts.build_opencodeinstruct_subset import parse_unit_tests
    assert parse_unit_tests("not json") is None
    assert parse_unit_tests("[unterminated") is None


def test_has_nondeterministic_pattern_detects_random():
    from scripts.build_opencodeinstruct_subset import has_nondeterministic_pattern
    assert has_nondeterministic_pattern("import random\nassert random.random() > 0") is True
    assert has_nondeterministic_pattern("import time; assert time.time() > 0") is True
    assert has_nondeterministic_pattern("import socket") is True
    assert has_nondeterministic_pattern("import urllib.request") is True
    assert has_nondeterministic_pattern("import requests") is True
    assert has_nondeterministic_pattern("import subprocess") is True
    assert has_nondeterministic_pattern("import threading") is True


def test_has_nondeterministic_pattern_clean_code():
    from scripts.build_opencodeinstruct_subset import has_nondeterministic_pattern
    assert has_nondeterministic_pattern("assert sum([1,2,3]) == 6") is False
    assert has_nondeterministic_pattern("assert sorted([3,1,2]) == [1,2,3]") is False


def test_filter_tests_drops_nondeterministic():
    from scripts.build_opencodeinstruct_subset import filter_tests
    tests = ["assert f(1) == 1", "import random; assert random.random() > 0"]
    kept = filter_tests(tests)
    assert kept == ["assert f(1) == 1"]
