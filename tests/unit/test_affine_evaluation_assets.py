"""Local asset evidence is distinct from a qualified Affine evaluation."""

import base64
import hashlib
import json
from pathlib import Path

from nacl.signing import SigningKey
import pytest

from reliquary.integrations.affine_cli import main
from reliquary.integrations.affine import AffineRuntimeError
from reliquary.integrations.affine_epoch import canonical
from reliquary.integrations.affine_evaluation import prepare
from reliquary.integrations.affine_evaluation_assets import check_assets
from reliquary.integrations import affine_evaluation_assets as assets
from reliquary.protocol.release_contract import canonical_sha256
from tests.unit.test_affine_evaluation import intake


def private_json(path, value):
    path.write_text(json.dumps(value))
    path.chmod(0o600)


def sign(payload):
    signer = SigningKey(b"\x01" * 32)
    return {
        "payload": payload,
        "signer": signer.verify_key.encode().hex(),
        "signature": base64.b64encode(
            signer.sign(canonical(payload)).signature
        ).decode(),
    }


def fixture(tmp_path):
    tmp_path.mkdir(mode=0o700, parents=True, exist_ok=True)
    intake_path, original, _ = intake(tmp_path)
    directory = tmp_path / "checkpoint"
    directory.mkdir(mode=0o700)
    for name, body in {
        "config.json": Path(original["modelConfig"]).read_bytes(),
        "model.safetensors": b"synthetic bytes, never a loadable model",
    }.items():
        member = directory / name
        member.write_bytes(body)
        member.chmod(0o600)
    files = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()
    }
    manifest = {
        "checkpoint": {
            "id": hashlib.sha256(canonical(files)).hexdigest(),
            "files": files,
        },
        "evaluation_sets": [original["nativeSuite"]["suite"]],
    }
    original["nativeSuite"]["manifestSha256"] = hashlib.sha256(
        canonical(manifest)
    ).hexdigest()
    private_json(intake_path, original)
    offer = tmp_path / "offer.private.json"
    prepare(intake_path, offer)
    opening = tmp_path / "opening.private.json"
    envelope = sign(manifest)
    private_json(opening, envelope)
    config = {
        "schema": "affine-evaluation-assets/v1",
        "preparedOffer": str(offer),
        "manifestEnvelope": str(opening),
        "authority": envelope["signer"],
        "modelSnapshot": original["modelSnapshot"],
        "modelDirectory": str(directory),
    }
    path = tmp_path / "assets.private.json"
    private_json(path, config)
    return path, config


def edit_offer(config, change):
    path = Path(config["preparedOffer"])
    offer = json.loads(path.read_text())
    change(offer)
    offer["offerDigest"] = canonical_sha256(
        {k: v for k, v in offer.items() if k != "offerDigest"}
    )
    private_json(path, offer)


def test_local_bytes_and_opening_are_bound_without_remote_or_native_claims(tmp_path):
    path, config = fixture(tmp_path)
    output = tmp_path / "checked.private.json"
    result = check_assets(path, output)
    receipt = json.loads(output.read_text())
    assert receipt["openingSignatureVerified"] is True
    assert receipt["localCheckpointBytesVerified"] is True
    assert receipt["fileCount"] == 2
    for key in (
        "publicModelLiveVerified",
        "nativeTaskMappingVerified",
        "nativeParityVerified",
        "gpuQualificationVerified",
        "activated",
    ):
        assert receipt[key] is False
    assert (
        receipt["offerDigest"]
        == json.loads(Path(config["preparedOffer"]).read_text())["offerDigest"]
    )
    assert result["stage"] == "local_assets_checked"
    assert str(tmp_path) not in output.read_text()
    assert "fixture/model" not in json.dumps(result)
    assert output.stat().st_mode & 0o777 == 0o600
    with pytest.raises(AffineRuntimeError):
        check_assets(path, output)


@pytest.mark.parametrize("change", ["changed", "missing", "extra", "symlink"])
def test_different_or_unsafe_local_inventory_is_refused(tmp_path, change):
    path, config = fixture(tmp_path)
    directory = Path(config["modelDirectory"])
    member = directory / "model.safetensors"
    if change == "changed":
        member.write_bytes(b"different")
    elif change == "extra":
        (directory / "README.md").write_text(
            "dedicated exports have exactly descriptor files"
        )
    else:
        member.unlink()
        if change == "symlink":
            member.symlink_to(directory / "config.json")
    output = tmp_path / "checked.private.json"
    with pytest.raises((ValueError, AffineRuntimeError)):
        check_assets(path, output)
    assert not output.exists()


def test_opening_signature_and_offer_digest_cannot_drift(tmp_path):
    path, config = fixture(tmp_path)
    opening = Path(config["manifestEnvelope"])
    envelope = json.loads(opening.read_text())
    envelope["payload"]["evaluation_sets"][0]["seed"] += 1
    private_json(opening, envelope)
    with pytest.raises(ValueError, match="signature"):
        check_assets(path, tmp_path / "checked.private.json")
    offer_path = Path(config["preparedOffer"])
    offer = json.loads(offer_path.read_text())
    offer["quoteRequest"]["revision"] = "0" * 40
    private_json(offer_path, offer)
    with pytest.raises(ValueError, match="frozen preparation"):
        check_assets(path, tmp_path / "checked.private.json")


def test_signed_suite_and_model_revision_must_match(tmp_path):
    path, config = fixture(tmp_path)
    edit_offer(config, lambda offer: offer["nativeSuite"]["suite"].update(seed=123))
    with pytest.raises(ValueError, match="suite differs"):
        check_assets(path, tmp_path / "checked.private.json")
    path, config = fixture(tmp_path / "second")
    snapshot = Path(config["modelSnapshot"])
    value = json.loads(snapshot.read_text())
    value["sha"] = "0" * 40
    private_json(snapshot, value)
    edit_offer(
        config,
        lambda offer: offer.update(
            modelSnapshotSha256=hashlib.sha256(snapshot.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="model snapshot differs"):
        check_assets(path, tmp_path / "checked.private.json")


def test_signed_checkpoint_needs_its_content_identifier(tmp_path):
    path, config = fixture(tmp_path)
    opening = Path(config["manifestEnvelope"])
    manifest = json.loads(opening.read_text())["payload"]
    manifest["checkpoint"]["id"] = "0" * 64
    private_json(opening, sign(manifest))
    edit_offer(
        config,
        lambda offer: offer["nativeSuite"].update(
            manifestSha256=hashlib.sha256(canonical(manifest)).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="content identifier"):
        check_assets(path, tmp_path / "checked.private.json")


def test_prepared_architecture_config_must_be_the_checked_model_config(tmp_path):
    path, config = fixture(tmp_path)
    changed = tmp_path / "config.private.json"
    private_json(changed, {"architectures": ["LlamaForCausalLM"]})
    edit_offer(
        config,
        lambda offer: offer.update(
            modelConfigSha256=hashlib.sha256(changed.read_bytes()).hexdigest()
        ),
    )
    with pytest.raises(ValueError, match="prepared model config"):
        check_assets(path, tmp_path / "checked.private.json")


def test_mutation_after_a_member_was_hashed_is_detected(tmp_path, monkeypatch):
    path, config = fixture(tmp_path)
    original = assets._member

    def mutate(member):
        result = original(member)
        if member.name == "model.safetensors":
            (member.parent / "config.json").write_bytes(
                b"changed during full inventory check"
            )
        return result

    monkeypatch.setattr(assets, "_member", mutate)
    with pytest.raises(ValueError, match="changed during verification"):
        check_assets(path, tmp_path / "checked.private.json")


@pytest.mark.parametrize("extra", ["read_urls", "unexpected"])
def test_only_existing_transport_metadata_may_extend_the_native_descriptor(
    tmp_path, extra
):
    path, config = fixture(tmp_path)
    opening = Path(config["manifestEnvelope"])
    manifest = json.loads(opening.read_text())["payload"]
    manifest["checkpoint"][extra] = {}
    private_json(opening, sign(manifest))
    edit_offer(
        config,
        lambda offer: offer["nativeSuite"].update(
            manifestSha256=hashlib.sha256(canonical(manifest)).hexdigest()
        ),
    )
    output = tmp_path / "checked.private.json"
    if extra == "read_urls":
        assert check_assets(path, output)["localCheckpointBytesVerified"] is True
    else:
        with pytest.raises(ValueError, match="exact native checkpoint descriptor"):
            check_assets(path, output)


def test_cli_prints_only_a_local_verification_summary(tmp_path, capsys):
    path, config = fixture(tmp_path)
    assert (
        main(
            [
                "evaluation",
                "check-assets",
                "--config",
                str(path),
                "--out",
                str(tmp_path / "checked.private.json"),
            ]
        )
        == 0
    )
    raw = capsys.readouterr().out
    assert (
        "fixture/model" not in raw
        and str(tmp_path) not in raw
        and config["authority"] not in raw
    )
    assert json.loads(raw)["activated"] is False
