"""`jobs create` writes two objects in two places. Either both land or
neither does, or an operator is left with a job nobody pays for."""

from __future__ import annotations

import pytest
from typer.testing import CliRunner

from reliquary.cli.main import app


# --- An in-memory bucket, so the REAL store code runs: its key layout, its
# conditional puts and its manifest validation are all part of what these
# tests are checking. ---


class _FakeBucket:
    def __init__(self):
        self.objects: dict[str, tuple[bytes, str]] = {}
        self.deleted: list[str] = []
        self._version = 0

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    async def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _client_error("NoSuchKey")
        body, etag = self.objects[Key]

        class _Body:
            async def read(self):
                return body

        return {"Body": _Body(), "ETag": etag}

    async def put_object(self, Bucket, Key, Body, **condition):
        current = self.objects.get(Key)
        if "IfMatch" in condition and condition["IfMatch"] != (
            current[1] if current else None
        ):
            raise _client_error("PreconditionFailed")
        if "IfNoneMatch" in condition and current is not None:
            raise _client_error("PreconditionFailed")
        self._version += 1
        etag = f'"v{self._version}"'
        self.objects[Key] = (Body, etag)
        return {"ETag": etag}

    async def delete_object(self, Bucket, Key):
        self.deleted.append(Key)
        self.objects.pop(Key, None)
        return {}

    def get_paginator(self, name):
        objects = self.objects

        class _Paginator:
            def paginate(self, Bucket, Prefix="", **kwargs):
                keys = sorted(k for k in objects if k.startswith(Prefix))

                async def _pages():
                    yield {"Contents": [{"Key": key} for key in keys]}

                return _pages()

        return _Paginator()


def _client_error(code: str):
    from botocore.exceptions import ClientError

    return ClientError({"Error": {"Code": code}}, "PutObject")


def _episode_sources(profile_id: str) -> list[str]:
    """The profile's environments a corpus job may actually draw from.

    `jobs create` refuses a prompt source prompt fidelity cannot render, and
    only an episode environment renders through `initial_text`.
    """
    from reliquary.environment.registry import ENVIRONMENT_SPECS
    from reliquary.protocol.profiles import PROFILES

    return sorted(
        name
        for name in PROFILES[profile_id].environments
        if getattr(ENVIRONMENT_SPECS.get(name), "interaction_mode", None) == "episode"
    )


def _template() -> str:
    from reliquary.protocol.profiles import PROFILES

    for profile_id in sorted(PROFILES):
        if len(PROFILES[profile_id].environments) >= 2 and _episode_sources(profile_id):
            return profile_id
    pytest.skip("no compiled profile declares two environments with a renderable one")


def _prompt_source(template: str) -> str:
    return _episode_sources(template)[0]


def _rl_entry(task_id: str, cap: float):
    from dataclasses import asdict

    from reliquary.shared.task_registry import (
        MECHANISM_RL_DISCOVERED_PRICE,
        TaskEntry,
    )
    from reliquary.validator.emission_price import PRODUCTION_PRICE_PARAMS

    params = asdict(PRODUCTION_PRICE_PARAMS)
    params["cap"] = cap
    return TaskEntry(
        task_id=task_id,
        profile_id="qwen3-4b-base-dapo-reliquary-v1",
        profile_sha256="b" * 64,
        mechanism=MECHANISM_RL_DISCOVERED_PRICE,
        params=params,
        status="active",
        retired_at=None,
    )


@pytest.fixture
def bucket(monkeypatch):
    """The job store's bucket, shared by every call inside one test."""
    from reliquary.infrastructure import corpus_job_store as job_store

    fake = _FakeBucket()
    monkeypatch.setattr(job_store, "get_s3_client", lambda **kwargs: fake)
    return fake


@pytest.fixture
def registry(monkeypatch):
    """The task registry, as a dict the test can seed and read back."""
    from reliquary.infrastructure import task_registry_store as store

    state = {"entries": {}, "etag": None}

    async def _read(*, strict=True, **kwargs):
        return dict(state["entries"]), state["etag"]

    async def _write(entries, etag, **kwargs):
        from reliquary.shared.task_registry import validate_registry

        validate_registry(entries)
        state["entries"] = dict(entries)
        state["etag"] = '"next"'
        return state["etag"]

    monkeypatch.setattr(store, "read_registry", _read)
    monkeypatch.setattr(store, "write_registry", _write)
    return state


ACK = "--fleet-knows-corpus-generation"


def _create_args(**overrides) -> list[str]:
    template = _template()
    options = {
        "--job-id": "swe-v1",
        "--task-id": "corpus-run",
        "--model": "org/Frozen",
        "--model-revision": "abc123",
        "--model-architecture": "Qwen3ForCausalLM",
        "--checkpoint-sha256": "a" * 64,
        "--from-profile": template,
        "--prompt-source": _prompt_source(template),
        "--prompt-count": "1000",
        "--renderer-id": "reliquary-external-prompt-v1",
        "--eos-token-id": "151645",
        "--max-new-tokens": "4096",
        "--slots-per-prompt": "8",
        "--cap": "0.30",
    }
    options.update(overrides)
    argv = ["jobs", "create", ACK]
    for flag, value in options.items():
        argv += [flag, value]
    return argv


def _manifest_keys(bucket) -> list[str]:
    return [k for k in bucket.objects if k.endswith(".json")]


def test_jobs_create_writes_the_manifest_and_the_entry(bucket, registry):
    """One command produces both halves, or neither. A manifest with no entry
    is an orphan nobody pays for; an entry with no manifest refuses at the
    first submission."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}

    result = CliRunner().invoke(app, _create_args())

    assert result.exit_code == 0, result.output
    assert _manifest_keys(bucket) == ["reliquary/corpus/jobs/swe-v1.json"]

    entry = registry["entries"]["corpus-run"]
    assert entry.job_id == "swe-v1"
    assert entry.mechanism == "corpus-generation"
    assert entry.params["floor"] == entry.params["cap"] == 0.30
    assert list(entry.contract["environments"]) == [_prompt_source(_template())]


def test_the_manifest_carries_what_the_operator_declared(bucket, registry):
    """The manifest is what the miner generates against; a field dropped here
    is a fleet generating to the wrong sampling."""
    import json

    registry["entries"] = {"default": _rl_entry("default", 0.5)}

    result = CliRunner().invoke(
        app, _create_args(**{"--temperature": "0.7", "--n": "4"})
    )

    assert result.exit_code == 0, result.output
    body, _ = bucket.objects["reliquary/corpus/jobs/swe-v1.json"]
    manifest = json.loads(body)
    assert manifest["checkpoint_repo"] == "org/Frozen"
    assert manifest["checkpoint_revision"] == "abc123"
    assert manifest["prompt_count"] == 1000
    assert manifest["slots_per_prompt"] == 8
    assert manifest["sampling"]["temperature"] == 0.7
    assert manifest["sampling"]["n"] == 4
    assert manifest["filter"] is None


def test_the_manifest_and_the_contract_name_one_checkpoint(bucket, registry):
    """A validator verifying one model while admitting against another job
    would pay for work nobody can reproduce. One flag feeds both, so they
    cannot disagree -- name that here, or nothing does."""
    import json

    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0

    body, _ = bucket.objects["reliquary/corpus/jobs/swe-v1.json"]
    manifest = json.loads(body)
    contract = registry["entries"]["corpus-run"].contract

    assert manifest["checkpoint_repo"] == contract["model_id"]
    assert manifest["checkpoint_revision"] == contract["model_revision"]
    assert manifest["prompt_source"] in contract["environments"]


def test_jobs_create_rolls_back_the_manifest_when_the_entry_write_fails(
    bucket, registry
):
    """The registry write is the one that can conflict, so it goes last and
    its failure removes the manifest."""
    # The pool is already fully declared, so adding 0.30 breaks the sum rule
    # inside `add_task` -- a real refusal, not a stubbed one.
    registry["entries"] = {"default": _rl_entry("default", 1.0)}

    result = CliRunner().invoke(app, _create_args())

    assert result.exit_code != 0
    # Name the refusal, so an unrecognised command could never satisfy this.
    assert "pool" in result.output
    # The manifest was written, then taken back: an absent key alone would
    # also be true of a command that never wrote one.
    assert bucket.deleted == ["reliquary/corpus/jobs/swe-v1.json"]
    assert _manifest_keys(bucket) == []
    assert set(registry["entries"]) == {"default"}


def test_jobs_create_refuses_without_the_fleet_acknowledgement(bucket, registry):
    """One corpus entry makes the WHOLE registry unreadable to a validator
    whose binary predates the mechanism, so the hazard is a deliberate act."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    argv = [token for token in _create_args() if token != ACK]

    result = CliRunner().invoke(app, argv)

    assert result.exit_code != 0
    assert ACK in result.output
    # Nothing at all: the refusal happens before either write.
    assert _manifest_keys(bucket) == []
    assert bucket.deleted == []
    assert set(registry["entries"]) == {"default"}


def test_an_ambiguous_registry_failure_leaves_the_manifest(
    bucket, registry, monkeypatch
):
    """A transport error on a put that actually landed would otherwise leave a
    declared task holding a cap share with its manifest deleted, refusing
    every submission it is paid for."""
    from reliquary.infrastructure import task_registry_store as store

    registry["entries"] = {"default": _rl_entry("default", 0.5)}

    async def _write(entries, etag, **kwargs):
        raise OSError("connection reset by peer")

    monkeypatch.setattr(store, "write_registry", _write)

    result = CliRunner().invoke(app, _create_args())

    assert result.exit_code != 0
    assert bucket.deleted == []
    assert _manifest_keys(bucket) == ["reliquary/corpus/jobs/swe-v1.json"]
    assert "swe-v1" in result.output
    assert "jobs list" in result.output


def test_jobs_create_refuses_a_job_that_already_has_a_manifest(bucket, registry):
    """Overwriting a live job's manifest would change the work under miners
    that already hold slots against it."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0

    second = CliRunner().invoke(
        app, _create_args(**{"--task-id": "corpus-run-2", "--cap": "0.1"})
    )

    assert second.exit_code != 0
    assert "swe-v1" in second.output
    assert set(registry["entries"]) == {"default", "corpus-run"}


class _RowsOnlySpec:
    """The real spec with only its build replaced, so a test can choose the
    source's row count. `jobs create` BUILDS the source to count its rows, and
    a unit test must not read a dataset to declare a job."""

    def __init__(self, spec, rows: int):
        self._spec = spec
        self._rows = rows

    def __getattr__(self, name):
        return getattr(self._spec, name)

    def create(self):
        rows = self._rows

        class _Environment:
            def __len__(self):
                return rows

        return _Environment()


def stub_source_rows(monkeypatch, source: str, rows: int) -> None:
    """Give one installed source a row count without reading its dataset."""
    from reliquary.validator import corpus_service

    specs = corpus_service.ENVIRONMENT_SPECS
    monkeypatch.setattr(
        corpus_service,
        "ENVIRONMENT_SPECS",
        {**specs, source: _RowsOnlySpec(specs[source], rows)},
    )


def test_jobs_create_refuses_a_prompt_count_the_source_cannot_fill(
    bucket, registry, monkeypatch
):
    """A job claiming more rows than its source has does not fail at
    declaration unless this check runs: `prompt_job_for_spec` first raises on
    the FIRST submission, as a 500, for every miner, forever."""
    stub_source_rows(monkeypatch, _prompt_source(_template()), 10)
    registry["entries"] = {"default": _rl_entry("default", 0.5)}

    result = CliRunner().invoke(app, _create_args(**{"--prompt-count": "1000"}))

    assert result.exit_code != 0
    # Both numbers, or the operator cannot tell which way the gap runs.
    assert "1000" in result.output and "10" in result.output
    # Nothing landed: the refusal precedes both writes.
    assert _manifest_keys(bucket) == []
    assert set(registry["entries"]) == {"default"}

    # The other side of the same bound, so the test above cannot be satisfied
    # by refusing every declaration.
    assert CliRunner().invoke(
        app, _create_args(**{"--prompt-count": "10"})
    ).exit_code == 0


def test_jobs_cancel_retires_the_entry_and_leaves_the_manifest(bucket, registry):
    """Submissions stop; the manifest stays readable for settlement."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0

    result = CliRunner().invoke(
        app, ["jobs", "cancel", "--job-id", "swe-v1", "--retired-at", "5000000"]
    )

    assert result.exit_code == 0, result.output
    assert registry["entries"]["corpus-run"].status == "retired"
    assert registry["entries"]["corpus-run"].retired_at == 5000000
    assert _manifest_keys(bucket) == ["reliquary/corpus/jobs/swe-v1.json"]


def test_jobs_cancel_on_a_job_nothing_declares_is_refused(bucket, registry):
    registry["entries"] = {"default": _rl_entry("default", 0.5)}

    result = CliRunner().invoke(
        app, ["jobs", "cancel", "--job-id", "ghost", "--retired-at", "5000000"]
    )

    assert result.exit_code != 0
    assert "ghost" in result.output


def test_jobs_list_shows_a_job_with_no_entry(bucket, registry):
    """The orphan case above must be visible, not hidden."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0
    # A manifest whose registry entry never landed: exactly what a rollback
    # that itself failed would leave behind.
    del registry["entries"]["corpus-run"]

    result = CliRunner().invoke(app, ["jobs", "list"])

    assert result.exit_code == 0, result.output
    assert "swe-v1" in result.output
    assert "no task entry" in result.output


def test_jobs_list_shows_a_task_whose_manifest_is_gone(bucket, registry):
    """The mirror of the orphan: this task holds a cap share and would refuse
    every submission, which is exactly what an ambiguous rollback can leave."""
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0
    bucket.objects.pop("reliquary/corpus/jobs/swe-v1.json")

    result = CliRunner().invoke(app, ["jobs", "list"])

    assert result.exit_code == 0, result.output
    assert "swe-v1" in result.output
    assert "NO MANIFEST" in result.output
    assert "corpus-run" in result.output


def test_jobs_list_names_the_task_that_declares_a_job(bucket, registry):
    registry["entries"] = {"default": _rl_entry("default", 0.5)}
    assert CliRunner().invoke(app, _create_args()).exit_code == 0

    result = CliRunner().invoke(app, ["jobs", "list"])

    assert result.exit_code == 0, result.output
    assert "corpus-run" in result.output
    assert "no task entry" not in result.output
