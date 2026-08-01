"""Qualification carry-over scoped to the proof path.

The capacity manifest pins the full image ``software_revision``, so ANY
deploy — docs, HTTP, tests — used to fail closed with "software revision
mismatch" until the benchmark was re-run, even when nothing that determines
proof latency had changed (2026-08-01: a 35-minute outage for an HTTP-only
change). The guard's intent is "the thing you measured is the thing that
runs"; its correct scope is the proof path, not the repo.

``proof_path_hash`` captures that scope directly: the bytes of every file on
the proof execution path plus the values of the constants that parametrize a
proof. At startup, a software-revision mismatch is accepted iff the manifest
carries a proof-path hash equal to the one recomputed from the running code —
otherwise it fails closed exactly as before. Legacy manifests (no hash) keep
the strict behavior.
"""

import pytest

from reliquary.validator.proof_capacity import (
    PROOF_PATH_FILES,
    ProofCapacityQualification,
    ProofCapacityQualificationError,
    compute_proof_path_hash,
)

from tests.unit.test_proof_capacity import _manifest


def _validate(manifest, *, software_revision, proof_path_hash=None, devices=9):
    qualification = ProofCapacityQualification.from_mapping(manifest)
    return qualification.validate(
        profile_id="qwen35-4b-auction-v3",
        model_revision="a" * 40,
        software_revision=software_revision,
        checkpoint_revision="c" * 40,
        runtime_fingerprint_hash="e" * 64,
        proof_path_hash=proof_path_hash,
        configured_devices=tuple(f"cuda:{i}" for i in range(devices)),
        configured_hardware=tuple(
            "NVIDIA H100 80GB HBM3" for _ in range(devices)
        ),
        configured_device_uuids=tuple(
            f"gpu-{index}" for index in range(devices)
        ),
        proof_wall_seconds=240.0,
        minimum_proofs_per_environment=18,
        minimum_completion_tokens_per_environment={
            "openmathinstruct": 14_746,
            "opencodeinstruct": 29_492,
        },
    )


# --------------------------------------------------------------------------
# the hash itself
# --------------------------------------------------------------------------

def _fake_tree(tmp_path, contents=b"x = 1\n"):
    for rel in PROOF_PATH_FILES:
        target = tmp_path / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(contents)
    return tmp_path


def test_hash_is_deterministic_and_well_formed(tmp_path):
    root = _fake_tree(tmp_path)
    first = compute_proof_path_hash(repo_root=root, parameters={"K": 32})
    second = compute_proof_path_hash(repo_root=root, parameters={"K": 32})
    assert first == second
    assert len(first) == 64 and set(first) <= set("0123456789abcdef")


def test_hash_changes_when_any_proof_file_changes(tmp_path):
    root = _fake_tree(tmp_path)
    baseline = compute_proof_path_hash(repo_root=root, parameters={"K": 32})
    for rel in PROOF_PATH_FILES:
        target = root / rel
        original = target.read_bytes()
        target.write_bytes(original + b"# touched\n")
        changed = compute_proof_path_hash(repo_root=root, parameters={"K": 32})
        assert changed != baseline, f"{rel} must be covered by the hash"
        target.write_bytes(original)


def test_hash_changes_with_proof_parameters(tmp_path):
    root = _fake_tree(tmp_path)
    a = compute_proof_path_hash(repo_root=root, parameters={"K": 32})
    b = compute_proof_path_hash(repo_root=root, parameters={"K": 64})
    assert a != b


def test_hash_of_the_real_tree_uses_live_constants():
    """Default invocation hashes the installed tree + live constants."""
    value = compute_proof_path_hash()
    assert value == compute_proof_path_hash()
    assert len(value) == 64


def test_missing_proof_file_raises(tmp_path):
    root = _fake_tree(tmp_path)
    (root / PROOF_PATH_FILES[0]).unlink()
    with pytest.raises(ProofCapacityQualificationError):
        compute_proof_path_hash(repo_root=root, parameters={"K": 32})


# --------------------------------------------------------------------------
# manifest round-trip
# --------------------------------------------------------------------------

def test_manifest_reads_optional_proof_path_hash():
    with_hash = ProofCapacityQualification.from_mapping(
        _manifest(proof_path_hash="f" * 64)
    )
    assert with_hash.proof_path_hash == "f" * 64
    legacy = ProofCapacityQualification.from_mapping(_manifest())
    assert legacy.proof_path_hash is None


# --------------------------------------------------------------------------
# validate(): the carry-over rule
# --------------------------------------------------------------------------

def test_same_revision_still_validates_without_any_hash():
    result = _validate(_manifest(), software_revision="b" * 40)
    assert result["qualified"] is True
    assert result.get("qualification_carried_over_from") is None


def test_revision_mismatch_with_matching_hash_carries_over():
    result = _validate(
        _manifest(proof_path_hash="f" * 64),
        software_revision="9" * 40,          # a different, valid image
        proof_path_hash="f" * 64,            # but the same proof path
    )
    assert result["qualified"] is True
    assert result["qualification_carried_over_from"] == "b" * 40


def test_revision_mismatch_against_legacy_manifest_fails_closed():
    with pytest.raises(
        ProofCapacityQualificationError, match="software revision"
    ):
        _validate(
            _manifest(),                      # no proof_path_hash stored
            software_revision="9" * 40,
            proof_path_hash="f" * 64,
        )


def test_revision_mismatch_with_different_hash_fails_closed():
    with pytest.raises(
        ProofCapacityQualificationError, match="software revision"
    ):
        _validate(
            _manifest(proof_path_hash="f" * 64),
            software_revision="9" * 40,
            proof_path_hash="0" * 64,         # the proof path DID change
        )


def test_revision_mismatch_with_missing_runtime_hash_fails_closed():
    """The caller not supplying its own hash must never unlock carry-over."""
    with pytest.raises(
        ProofCapacityQualificationError, match="software revision"
    ):
        _validate(
            _manifest(proof_path_hash="f" * 64),
            software_revision="9" * 40,
            proof_path_hash=None,
        )


def test_malformed_stored_hash_fails_closed():
    with pytest.raises(ProofCapacityQualificationError):
        _validate(
            _manifest(proof_path_hash="not-a-hash"),
            software_revision="9" * 40,
            proof_path_hash="f" * 64,
        )
