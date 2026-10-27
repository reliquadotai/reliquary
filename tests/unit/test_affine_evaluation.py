"""Private Affine provenance intake feeds the existing subnet evaluation API."""

from __future__ import annotations

import json

import pytest

from reliquary.eval.sets import build_set
from reliquary.integrations import affine_evaluation as evaluation
from reliquary.integrations.affine import AffineRuntimeError
from reliquary.integrations.affine_cli import main
from reliquary.protocol.release_contract import canonical_sha256
from tests.unit.test_eval_sets import opener


def intake(tmp_path):
    reviewed = {
        "data": {"source.json": "1" * 64},
        "licenses": [{"id": "fixture", "sha256": "2" * 64}],
        "disjointness": {"disjoint": True, "evidence_sha256": "3" * 64},
    }
    directory = tmp_path / "set"
    card = build_set(
        "logic",
        count=2,
        seed=1,
        out=directory,
        open_environment=opener(),
        protocol=reviewed,
    )
    for path in directory.iterdir():
        path.chmod(0o600)
    snapshot = tmp_path / "model.private.json"
    snapshot.write_text(
        json.dumps(
            {"id": "fixture/model", "sha": "a" * 40, "private": False, "gated": False}
        )
    )
    snapshot.chmod(0o600)
    model = tmp_path / "config.private.json"
    model.write_text(json.dumps({"architectures": ["Qwen3ForCausalLM"]}))
    model.chmod(0o600)
    rows = [
        json.loads(line)
        for line in (directory / "prompts.jsonl").read_text().splitlines()
    ]
    config = {
        "schema": "affine-subnet-evaluation-intake/v1",
        "setDirectory": str(directory),
        "modelSnapshot": str(snapshot),
        "modelConfig": str(model),
        "nativeSuite": {
            "manifestSha256": "b" * 64,
            "sourceRevision": "c" * 40,
            "suite": {
                "name": "fixture",
                "env_id": "fixture_math",
                "indices": [5, 8],
                "seed": 42,
                "repeats": 1,
            },
            "taskMapping": [
                {
                    "nativeIndex": index,
                    "taskSha256": "d" * 64,
                    "problemId": row["problem_id"],
                    "promptSha256": canonical_sha256(row["messages"]),
                }
                for index, row in zip([5, 8], rows, strict=True)
            ],
        },
        "request": {
            "problems": 2,
            "samples": 1,
            "max_new_tokens": 128,
            "thinking": False,
            "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "seed": 42},
        },
        "review": dict.fromkeys(evaluation.REVIEW_FIELDS, True),
    }
    path = tmp_path / "intake.private.json"
    path.write_text(json.dumps(config))
    path.chmod(0o600)
    return path, config, card


def rewrite(path, config):
    path.write_text(json.dumps(config))


def test_prepare_exports_existing_quote_body_without_claiming_native_parity(tmp_path):
    path, config, card = intake(tmp_path)
    output = tmp_path / "offer.private.json"
    summary = evaluation.prepare(path, output)
    offer = json.loads(output.read_text())
    assert offer["availability"] == "withdrawn"
    assert offer["compatibility"] == "independent-subnet-evaluation"
    assert not any(
        offer[name]
        for name in (
            "nativeParityVerified",
            "gpuQualificationVerified",
            "publicationVerified",
            "nativeTaskMappingVerified",
        )
    )
    assert offer["promptsVisibleToMiners"] is True
    assert offer["quoteRequest"] == {
        "model": "fixture/model",
        "revision": "a" * 40,
        "mode": "subnet",
        "envs": [
            {"env": card["env"], "set_id": card["set_id"], "problems": 2, "samples": 1}
        ],
        "max_new_tokens": 128,
        "thinking": False,
        "sampling": config["request"]["sampling"],
    }
    assert offer["offerDigest"] == canonical_sha256(
        {key: value for key, value in offer.items() if key != "offerDigest"}
    )
    assert output.stat().st_mode & 0o777 == 0o600
    assert str(tmp_path) not in output.read_text()
    assert "nativeSuite" not in summary and "quoteRequest" not in summary
    with pytest.raises(AffineRuntimeError, match="private runtime file"):
        evaluation.prepare(path, output)


@pytest.mark.parametrize("field", sorted(evaluation.REVIEW_FIELDS))
def test_each_explicit_review_is_required(tmp_path, field):
    path, config, _ = intake(tmp_path)
    config["review"][field] = False
    rewrite(path, config)
    with pytest.raises(ValueError, match="require review"):
        evaluation.prepare(path, tmp_path / "offer.private.json")


def test_source_bytes_and_native_prompt_mapping_cannot_drift(tmp_path):
    path, config, _ = intake(tmp_path)
    config["nativeSuite"]["taskMapping"][0]["promptSha256"] = "0" * 64
    rewrite(path, config)
    with pytest.raises(ValueError, match="mapping differs"):
        evaluation.prepare(path, tmp_path / "offer.private.json")
    path, config, _ = intake(tmp_path / "other")
    prompts = tmp_path / "other/set/prompts.jsonl"
    prompts.write_bytes(prompts.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="sha256"):
        evaluation.prepare(path, tmp_path / "offer.private.json")


@pytest.mark.parametrize(
    "change",
    [
        {"temperature": 0},
        {"temperature": float("inf")},
        {"top_k": 0},
        {"seed": -1},
    ],
)
def test_subnet_sampling_is_bounded_and_non_greedy(tmp_path, change):
    path, config, _ = intake(tmp_path)
    config["request"]["sampling"].update(change)
    rewrite(path, config)
    with pytest.raises(ValueError):
        evaluation.prepare(path, tmp_path / "offer.private.json")


def test_private_gated_and_unsupported_models_are_refused(tmp_path):
    path, config, _ = intake(tmp_path)
    snapshot = tmp_path / "model.private.json"
    body = json.loads(snapshot.read_text())
    for changes in ({"private": True}, {"gated": "auto"}, {"sha": "main"}):
        snapshot.write_text(json.dumps({**body, **changes}))
        with pytest.raises(ValueError, match="public ungated"):
            evaluation.prepare(path, tmp_path / "offer.private.json")
    snapshot.write_text(json.dumps(body))
    (tmp_path / "config.private.json").write_text(
        json.dumps({"architectures": ["UnsupportedForCausalLM"]})
    )
    with pytest.raises(ValueError, match="not supported"):
        evaluation.prepare(path, tmp_path / "offer.private.json")


def test_cli_only_prints_safe_summary_and_retains_private_offer(tmp_path, capsys):
    path, _, _ = intake(tmp_path)
    output = tmp_path / "offer.private.json"
    assert (
        main(["evaluation", "prepare", "--config", str(path), "--out", str(output)])
        == 0
    )
    printed = capsys.readouterr().out
    assert "fixture/model" not in printed and str(tmp_path) not in printed
    assert json.loads(printed)["availability"] == "withdrawn"
    path.chmod(0o644)
    assert (
        main(["evaluation", "prepare", "--config", str(path), "--out", str(output)])
        == 1
    )
    assert "fixture/model" not in capsys.readouterr().out
