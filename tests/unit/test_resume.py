"""Parser + resolver for the --resume-from source string."""

import pytest


def test_parse_sha_source():
    from reliquary.validator.resume import parse_resume_source, ShaSource
    r = parse_resume_source("sha:fa53996ed1533fadfc86be0e6158ddd8465acf34")
    assert isinstance(r, ShaSource)
    assert r.sha == "fa53996ed1533fadfc86be0e6158ddd8465acf34"


def test_parse_path_source(tmp_path):
    from reliquary.validator.resume import parse_resume_source, PathSource
    r = parse_resume_source(f"path:{tmp_path}")
    assert isinstance(r, PathSource)
    assert r.path == str(tmp_path)


def test_parse_path_source_relative():
    from reliquary.validator.resume import parse_resume_source, PathSource
    r = parse_resume_source("path:./state/checkpoints/ckpt_5")
    assert isinstance(r, PathSource)
    assert r.path == "./state/checkpoints/ckpt_5"


def test_parse_rejects_unknown_scheme():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="unknown scheme"):
        parse_resume_source("weird:xyz")


def test_parse_rejects_bare_string():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="expected scheme"):
        parse_resume_source("fa53996ed153")


def test_parse_rejects_malformed_sha():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="not a 40-char hex"):
        parse_resume_source("sha:notahex")


def test_parse_rejects_empty_path():
    from reliquary.validator.resume import parse_resume_source
    with pytest.raises(ValueError, match="path is empty"):
        parse_resume_source("path:")


from unittest.mock import MagicMock


def test_resolve_sha_downloads_and_extracts_n():
    """SHA mode: snapshot_download is called with the right args, and the
    checkpoint_n is parsed from the commit title."""
    from reliquary.validator.resume import (
        ShaSource, resolve_resume_source,
    )

    calls = {}

    def fake_download(repo_id, revision, **kwargs):
        calls["repo_id"] = repo_id
        calls["revision"] = revision
        return "/tmp/fake-hf-cache/snapshots/" + revision

    def fake_commit_title(repo_id, revision):
        calls["title_query"] = (repo_id, revision)
        return "checkpoint 7"

    local_path, checkpoint_n = resolve_resume_source(
        source=ShaSource(sha="a" * 40),
        hf_repo_id="myorg/repo",
        download_fn=fake_download,
        commit_title_fn=fake_commit_title,
    )
    assert checkpoint_n == 7
    assert local_path == "/tmp/fake-hf-cache/snapshots/" + "a" * 40
    assert calls["repo_id"] == "myorg/repo"
    assert calls["revision"] == "a" * 40


@pytest.mark.parametrize(
    "title, expected",
    [
        ("checkpoint 623 (cadence)", 623),
        ("checkpoint 624 (adaptive_policy_ratio_drift)", 624),
    ],
)
def test_resolve_sha_accepts_detached_trainer_titles(title, expected):
    """A revision discovered from a detached publication must also load.

    Production previously discovered checkpoint 623 correctly, then rejected
    that same SHA here because the resolver still required ``checkpoint N``.
    """
    from reliquary.validator.resume import ShaSource, resolve_resume_source

    local_path, checkpoint_n = resolve_resume_source(
        source=ShaSource(sha="c" * 40),
        hf_repo_id="myorg/repo",
        download_fn=lambda **kwargs: "/tmp/checkpoint-623",
        commit_title_fn=lambda **kwargs: title,
    )

    assert local_path == "/tmp/checkpoint-623"
    assert checkpoint_n == expected


def test_resolve_sha_rejects_unparseable_title():
    """If the commit title isn't 'checkpoint N', refuse (don't silently
    pick a wrong N)."""
    from reliquary.validator.resume import (
        ShaSource, resolve_resume_source,
    )

    with pytest.raises(ValueError, match="could not parse"):
        resolve_resume_source(
            source=ShaSource(sha="b" * 40),
            hf_repo_id="myorg/repo",
            download_fn=lambda **kw: "/x",
            commit_title_fn=lambda **kw: "some random commit",
        )


def test_resolve_path_uses_dir_as_is(tmp_path):
    """path mode: the provided directory is returned verbatim; checkpoint_n
    is extracted from ``ckpt_<N>`` in the trailing component."""
    from reliquary.validator.resume import (
        PathSource, resolve_resume_source,
    )

    target = tmp_path / "ckpt_12"
    target.mkdir()
    local_path, checkpoint_n = resolve_resume_source(
        source=PathSource(path=str(target)),
        hf_repo_id="unused",
    )
    assert local_path == str(target)
    assert checkpoint_n == 12


def test_resolve_path_rejects_unparseable_dir(tmp_path):
    """path mode without ``ckpt_N`` → ValueError. Operators must name the
    directory so the checkpoint number is unambiguous."""
    from reliquary.validator.resume import (
        PathSource, resolve_resume_source,
    )
    target = tmp_path / "random_name"
    target.mkdir()
    with pytest.raises(ValueError, match="could not derive checkpoint_n"):
        resolve_resume_source(
            source=PathSource(path=str(target)),
            hf_repo_id="unused",
        )
