"""Real HF cache deletion preserves shared weights and active revisions."""
import os

from reliquary.validator.remote_proof_server import prune_checkpoint_cache


def test_retention_preserves_active_recent_and_other_repositories(tmp_path, monkeypatch):
    monkeypatch.setattr("huggingface_hub.constants.HF_HUB_CACHE", str(tmp_path))
    hashes = [str(n) * 40 for n in range(1, 7)]
    for name, revisions in (("models--test--checkpoints", hashes),
                            ("models--other--base", ["a" * 40])):
        repo = tmp_path / name
        (repo / "blobs").mkdir(parents=True)
        shared = repo / "blobs" / "shared"
        shared.write_text("shared")
        for index, revision in enumerate(revisions):
            snapshot = repo / "snapshots" / revision
            snapshot.mkdir(parents=True)
            blob = repo / "blobs" / revision
            blob.write_text(revision)
            os.utime(blob, (100 + index, 100 + index))
            (snapshot / "model").symlink_to(blob)
            (snapshot / "config").symlink_to(shared)
            os.utime(shared, (1, 1))
    prune_checkpoint_cache("test/checkpoints", {hashes[0], hashes[2]})
    repo = tmp_path / "models--test--checkpoints"
    assert {p.name for p in (repo / "snapshots").iterdir()} == {
        hashes[0], hashes[2], hashes[4], hashes[5]}
    assert (repo / "blobs" / "shared").read_text() == "shared"
    assert not (repo / "blobs" / hashes[1]).exists()
    assert (tmp_path / "models--other--base" / "snapshots" / ("a" * 40) / "model").is_file()
