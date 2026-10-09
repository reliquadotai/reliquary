"""Dataset publication is the completion marker for an episode export."""

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reliquary.corpus.export_bundle import publish_episode_export


@pytest.fixture
def staged(tmp_path):
    path = tmp_path / "staged.jsonl"
    path.write_text('{"completed": true}\n')
    return path


def test_episode_dataset_is_published_after_its_counts(monkeypatch, tmp_path, staged):
    out = tmp_path / "result.jsonl"
    counts = {"exported": 1, "drained": True}
    real_link = os.link
    publications = []

    def link(source, destination):
        publications.append(destination)
        if destination == out:
            assert json.loads((tmp_path / "result.jsonl.counts.json").read_text()) == counts
        real_link(source, destination)

    monkeypatch.setattr(os, "link", link)
    publish_episode_export(staged, out, counts)
    assert out.read_text() == '{"completed": true}\n'
    assert publications == [tmp_path / "result.jsonl.counts.json", out]
    assert not staged.exists()
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize("existing", ["result.jsonl", "result.jsonl.counts.json"])
def test_episode_publication_preserves_existing_results(tmp_path, staged, existing):
    path = tmp_path / existing
    path.write_text("previous result")
    with pytest.raises(FileExistsError, match="new output"):
        publish_episode_export(staged, tmp_path / "result.jsonl", {"exported": 1})
    assert path.read_text() == "previous result"
    assert staged.exists()


@pytest.mark.parametrize("failed", ["result.jsonl", "result.jsonl.counts.json"])
def test_failed_publication_rolls_back_new_bundle(monkeypatch, tmp_path, staged, failed):
    real_link = os.link

    def link(source, destination):
        if destination.name == failed:
            raise OSError("publication unavailable")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(OSError, match="publication unavailable"):
        publish_episode_export(staged, tmp_path / "result.jsonl", {"exported": 1})
    assert list(tmp_path.iterdir()) == [staged]


def test_failed_counts_serialization_never_publishes_dataset(tmp_path, staged):
    with pytest.raises(TypeError):
        publish_episode_export(staged, tmp_path / "result.jsonl", {"exported": object()})
    assert list(tmp_path.iterdir()) == [staged]


def test_concurrent_dataset_creation_is_preserved(monkeypatch, tmp_path, staged):
    out = tmp_path / "result.jsonl"
    real_link = os.link

    def link(source, destination):
        if destination == out:
            out.write_text("concurrent result")
        real_link(source, destination)

    monkeypatch.setattr(os, "link", link)
    with pytest.raises(FileExistsError):
        publish_episode_export(staged, out, {"exported": 1})
    assert out.read_text() == "concurrent result"
    assert not (tmp_path / "result.jsonl.counts.json").exists()


def test_directory_sync_failure_rolls_back_published_bundle(monkeypatch, tmp_path, staged):
    real_sync = os.fsync
    calls = 0

    def sync(fd):
        nonlocal calls
        calls += 1
        if calls == 4:
            raise OSError("directory sync unavailable")
        real_sync(fd)

    monkeypatch.setattr(os, "fsync", sync)
    with pytest.raises(OSError, match="directory sync unavailable"):
        publish_episode_export(staged, tmp_path / "result.jsonl", {"exported": 1})
    assert list(tmp_path.iterdir()) == [staged]


def test_interrupted_publication_leaves_no_completed_dataset(tmp_path, staged):
    out = tmp_path / "result.jsonl"
    script = """
import os
from pathlib import Path
import sys
from reliquary.corpus import export_bundle

temporary, out = map(Path, sys.argv[1:])
real_link = os.link
def link(source, destination):
    if destination == out:
        os._exit(17)
    real_link(source, destination)
export_bundle.os.link = link
export_bundle.publish_episode_export(temporary, out, {"exported": 1})
"""
    env = {**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])}
    result = subprocess.run([sys.executable, "-c", script, str(staged), str(out)],
                            env=env, capture_output=True, timeout=5)
    assert result.returncode == 17, result.stderr.decode()
    assert not out.exists()
    assert json.loads((tmp_path / "result.jsonl.counts.json").read_text()) == {"exported": 1}
