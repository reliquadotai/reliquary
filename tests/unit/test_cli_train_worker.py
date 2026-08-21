"""Resume-point resolution for the train-worker CLI."""

import json

import pytest

from reliquary.trainer.resume import resolve_resume_point


def _manifest_fetch(doc):
    payload = json.dumps(doc).encode("utf-8")

    def fetch(key):
        if key == "reliquary/training/candidate-manifest.json":
            return payload
        return None

    return fetch


def test_manifest_present_wins_over_env():
    revision, cursor = resolve_resume_point(
        _manifest_fetch({
            "checkpoint_n": 5, "repo_id": "org/repo", "revision": "rev-7",
            "trained_window_cursor": 30110, "reason": "cadence",
        }),
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "999"},
    )
    assert revision == "rev-7"
    assert cursor == 30110


def test_no_manifest_falls_back_to_env_cursor():
    revision, cursor = resolve_resume_point(
        lambda key: None,
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30050"},
    )
    assert revision is None
    assert cursor == 30050


def test_no_manifest_no_env_refuses_to_guess():
    with pytest.raises(SystemExit):
        resolve_resume_point(lambda key: None, env={})
