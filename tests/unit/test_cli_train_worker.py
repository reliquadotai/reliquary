"""Resume-point resolution for the train-worker CLI."""

import json

import pytest

from reliquary.trainer.resume import resolve_resume_point


REV_5 = "5" * 40
REV_7 = "7" * 40


def _manifest_fetch(doc):
    payload = json.dumps(doc).encode("utf-8")

    def fetch(key):
        if key == "reliquary/training/candidate-manifest.json":
            return payload
        return None

    return fetch


def test_manifest_present_wins_over_env():
    revision, cursor, checkpoint_n = resolve_resume_point(
        _manifest_fetch({
            "checkpoint_n": 530, "repo_id": "org/repo", "revision": REV_7,
            "trained_window_cursor": 30110, "reason": "cadence",
        }),
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "999"},
    )
    assert revision == REV_7
    assert cursor == 30110
    # Numbering must never regress across restarts (two FATALs already
    # came from trusting a derived/inherited counter).
    assert checkpoint_n == 530


def test_no_manifest_falls_back_to_env_cursor():
    revision, cursor, checkpoint_n = resolve_resume_point(
        lambda key: None,
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30050"},
    )
    assert revision is None
    assert cursor == 30050
    assert checkpoint_n == 0


def test_no_manifest_no_env_refuses_to_guess():
    with pytest.raises(SystemExit):
        resolve_resume_point(lambda key: None, env={})


def test_bootstrap_revision_for_shadow_and_cutover():
    # Mid-run start (shadow / cutover): begin from the validator's last
    # published checkpoint instead of the base model.
    revision, cursor, checkpoint_n = resolve_resume_point(
        lambda key: None,
        env={
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30110",
            "RELIQUARY_TRAINER_BOOTSTRAP_REVISION": REV_5,
            "RELIQUARY_TRAINER_CHECKPOINT_N": "527",
        },
    )
    assert revision == REV_5
    assert cursor == 30110
    assert checkpoint_n == 527


def test_matching_v5_manifest_wins_over_bootstrap():
    identity = {
        "protocol_profile_id": "reasoning-v5",
        "protocol_version": 5,
        "training_run_id": "run-v5",
        "generation_contract_sha256": "a" * 64,
    }
    revision, cursor, checkpoint_n = resolve_resume_point(
        _manifest_fetch({
            **identity,
            "checkpoint_n": 531,
            "repo_id": "org/repo",
            "revision": REV_5,
            "trained_window_cursor": 30200,
            "reason": "cadence",
        }),
        env={"RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "999"},
        expected_identity=identity,
    )
    assert (revision, cursor, checkpoint_n) == (REV_5, 30200, 531)


def test_stale_v4_manifest_uses_explicit_v5_base_reset():
    identity = {
        "protocol_profile_id": "reasoning-v5",
        "protocol_version": 5,
        "training_run_id": "run-v5",
        "generation_contract_sha256": "a" * 64,
    }
    revision, cursor, checkpoint_n = resolve_resume_point(
        _manifest_fetch({
            "checkpoint_n": 530,
            "repo_id": "org/repo",
            "revision": "v4-rev",
            "trained_window_cursor": 30110,
            "reason": "cadence",
        }),
        env={
            "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30250",
            "RELIQUARY_TRAINER_BOOTSTRAP_REVISION": REV_5,
            "RELIQUARY_TRAINER_CHECKPOINT_N": "531",
        },
        expected_identity=identity,
    )
    assert (revision, cursor, checkpoint_n) == (
        REV_5,
        30250,
        531,
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


def test_matching_manifest_rejects_mutable_revision():
    with pytest.raises(ValueError, match="40-character commit OID"):
        resolve_resume_point(
            _manifest_fetch(
                {
                    "checkpoint_n": 530,
                    "repo_id": "org/repo",
                    "revision": "main",
                    "trained_window_cursor": 30110,
                }
            ),
            env={},
        )


@pytest.mark.parametrize("checkpoint_n", [True, 2.0, "2", -1])
def test_matching_manifest_rejects_noncanonical_checkpoint_number(
    checkpoint_n,
):
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_resume_point(
            _manifest_fetch(
                {
                    "checkpoint_n": checkpoint_n,
                    "repo_id": "org/repo",
                    "revision": REV_5,
                    "trained_window_cursor": 30110,
                }
            ),
            env={},
        )


def test_matching_manifest_rejects_duplicate_checkpoint_number():
    raw = (
        b'{"checkpoint_n":529,"checkpoint_n":530,"repo_id":"org/repo",'
        + f'"revision":"{REV_5}","trained_window_cursor":30110}}'.encode()
    )

    with pytest.raises(ValueError, match="duplicate JSON key: checkpoint_n"):
        resolve_resume_point(lambda key: raw, env={})


def test_bootstrap_rejects_negative_checkpoint_number():
    with pytest.raises(ValueError, match="non-negative integer"):
        resolve_resume_point(
            lambda key: None,
            env={
                "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30110",
                "RELIQUARY_TRAINER_CHECKPOINT_N": "-1",
            },
        )


def test_bootstrap_rejects_mutable_revision():
    with pytest.raises(ValueError, match="40-character commit OID"):
        resolve_resume_point(
            lambda key: None,
            env={
                "RELIQUARY_TRAINER_BOOTSTRAP_CURSOR": "30110",
                "RELIQUARY_TRAINER_BOOTSTRAP_REVISION": "main",
            },
        )
