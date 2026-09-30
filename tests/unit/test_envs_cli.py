"""`reliquary envs list | show` prints what a composed task would carry."""

from __future__ import annotations

import json

from typer.testing import CliRunner

from reliquary.cli.main import app
from reliquary.protocol.environment_catalog import (
    CATALOG_PROVENANCE,
    ENVIRONMENT_CATALOG,
    TUNABLE_FIELDS,
    environment_body_contract,
)
from reliquary.protocol.profiles import PROFILES
from reliquary.protocol.release_contract import canonical_sha256
from tests.unit.test_environment_catalog import GOLDEN_BODY_SHA256


def test_body_contract_is_what_the_source_profile_writes():
    for name, source in CATALOG_PROVENANCE.items():
        written = PROFILES[source].to_generation_contract()["environments"][name]
        assert environment_body_contract(name) == written


def test_list_names_every_environment_with_provenance_and_digest():
    result = CliRunner().invoke(app, ["envs", "list"])
    assert result.exit_code == 0, result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == len(ENVIRONMENT_CATALOG) + 1  # header
    for name in ENVIRONMENT_CATALOG:
        line = next(line for line in lines if line.split()[0] == name)
        assert CATALOG_PROVENANCE[name] in line
        assert GOLDEN_BODY_SHA256[name][:12] in line
        assert str(ENVIRONMENT_CATALOG[name].max_new_tokens) in line


def test_show_prints_the_body_its_digest_provenance_and_tunable_fields():
    result = CliRunner().invoke(app, ["envs", "show", "reliquary_code_v1"])
    assert result.exit_code == 0, result.output
    shown = json.loads(result.output)
    assert shown == {
        "name": "reliquary_code_v1",
        "body": environment_body_contract("reliquary_code_v1"),
        "canonical_sha256": GOLDEN_BODY_SHA256["reliquary_code_v1"],
        "provenance": "teutonic-9b-reliquary-suite-v9-dev1",
        "tunable_fields": sorted(TUNABLE_FIELDS),
    }
    assert canonical_sha256(shown["body"]) == shown["canonical_sha256"]


def test_show_refuses_an_unknown_name_listing_the_known_ones():
    result = CliRunner().invoke(app, ["envs", "show", "no_such_env"])
    assert result.exit_code != 0
    assert "reliquary_code_v1" in result.output


def test_show_says_an_installed_but_uncatalogued_environment_has_no_entry():
    result = CliRunner().invoke(app, ["envs", "show", "envscaler_tools_v1"])
    assert result.exit_code != 0
    assert "no catalog entry" in result.output
