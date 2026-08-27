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
    revision, cursor, checkpoint_n, publication_seq = resolve_resume_point(
        _manifest_fetch({
            "checkpoint_n": 530, "repo_id": "org/repo", "revision": "rev-7",
            "trained_window_cursor": 30110, "reason": "cadence",
        }),
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "999"},
    )
    assert revision == "rev-7"
    assert cursor == 30110
    # Numbering must never regress across restarts (two FATALs already
    # came from trusting a derived/inherited counter).
    assert checkpoint_n == 530
    assert publication_seq is None


def test_no_manifest_falls_back_to_env_cursor():
    revision, cursor, checkpoint_n, publication_seq = resolve_resume_point(
        lambda key: None,
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30050"},
    )
    assert revision is None
    assert cursor == 30050
    assert checkpoint_n == 0
    assert publication_seq == 0


def test_no_manifest_no_env_refuses_to_guess():
    with pytest.raises(SystemExit):
        resolve_resume_point(lambda key: None, env={})


def test_bootstrap_revision_for_shadow_and_cutover():
    # Mid-run start (shadow / cutover): begin from the validator's last
    # published checkpoint instead of the base model.
    revision, cursor, checkpoint_n, publication_seq = resolve_resume_point(
        lambda key: None,
        env={
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30110",
            "RELIQUARY_TRAINER_BOOTSTRAP_REVISION": "2463086760b7",
            "RELIQUARY_TRAINER_CHECKPOINT_N": "527",
            "RELIQUARY_TRAINER_PUBLICATION_SEQ": "12",
        },
    )
    assert revision == "2463086760b7"
    assert cursor == 30110
    assert checkpoint_n == 527
    assert publication_seq == 12


def test_new_run_base_reset_starts_publication_sequence_at_zero():
    identity = {
        "protocol_profile_id": "reasoning-v5",
        "protocol_version": 5,
        "training_run_id": "new-run-v5",
        "generation_contract_sha256": "a" * 64,
    }
    revision, cursor, checkpoint_n, publication_seq = resolve_resume_point(
        _manifest_fetch({
            "checkpoint_n": 659,
            "revision": "old-run-rev",
            "trained_window_cursor": 32790,
            "training_run_id": "old-run-v4",
        }),
        env={
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "32790",
            "RELIQUARY_TRAINER_BOOTSTRAP_REVISION": "base-reset-660",
            "RELIQUARY_TRAINER_CHECKPOINT_N": "660",
        },
        expected_identity=identity,
    )
    assert (revision, cursor, checkpoint_n, publication_seq) == (
        "base-reset-660", 32790, 660, 0,
    )


def test_matching_v5_manifest_wins_over_bootstrap():
    identity = {
        "protocol_profile_id": "reasoning-v5",
        "protocol_version": 5,
        "training_run_id": "run-v5",
        "generation_contract_sha256": "a" * 64,
    }
    revision, cursor, checkpoint_n, publication_seq = resolve_resume_point(
        _manifest_fetch({
            **identity,
            "checkpoint_n": 531,
            "repo_id": "org/repo",
            "revision": "v5-rev",
            "trained_window_cursor": 30200,
            "reason": "cadence",
            "publication_seq": 19,
        }),
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "999"},
        expected_identity=identity,
    )
    assert (revision, cursor, checkpoint_n, publication_seq) == (
        "v5-rev", 30200, 531, 19,
    )


def test_stale_v4_manifest_uses_explicit_v5_base_reset():
    identity = {
        "protocol_profile_id": "reasoning-v5",
        "protocol_version": 5,
        "training_run_id": "run-v5",
        "generation_contract_sha256": "a" * 64,
    }
    revision, cursor, checkpoint_n, publication_seq = resolve_resume_point(
        _manifest_fetch({
            "checkpoint_n": 530,
            "repo_id": "org/repo",
            "revision": "v4-rev",
            "trained_window_cursor": 30110,
            "reason": "cadence",
        }),
        env={
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30250",
            "RELIQUARY_TRAINER_BOOTSTRAP_REVISION": "v5-base-reset",
            "RELIQUARY_TRAINER_CHECKPOINT_N": "531",
            "RELIQUARY_TRAINER_PUBLICATION_SEQ": "0",
        },
        expected_identity=identity,
    )
    assert (revision, cursor, checkpoint_n, publication_seq) == (
        "v5-base-reset",
        30250,
        531,
        0,
    )


def test_legacy_manifest_accepts_audited_publication_sequence_from_env():
    revision, cursor, checkpoint_n, publication_seq = resolve_resume_point(
        _manifest_fetch({
            "checkpoint_n": 1085,
            "repo_id": "org/repo",
            "revision": "legacy-rev",
            "trained_window_cursor": 40000,
            "reason": "cadence",
        }),
        env={"RELIQUARY_TRAINER_PUBLICATION_SEQ": "229"},
    )

    assert (revision, cursor, checkpoint_n, publication_seq) == (
        "legacy-rev", 40000, 1085, 229,
    )


def test_stale_manifest_without_v5_bootstrap_refuses_to_guess():
    with pytest.raises(SystemExit):
        resolve_resume_point(
            _manifest_fetch({
                "checkpoint_n": 530,
                "revision": "v4-rev",
                "trained_window_cursor": 30110,
            }),
            env={},
            expected_identity={"protocol_version": 5},
        )
