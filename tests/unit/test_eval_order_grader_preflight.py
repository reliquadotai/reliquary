"""A new evaluation cannot engage work its frozen grader will refuse."""
from __future__ import annotations

import asyncio
import copy

import pytest

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.eval import protocol
from reliquary.eval import qualification as qual
from reliquary.eval.sets import build_set
from reliquary.eval.storage import SubnetEvalStore, publish_set, subnet_key
from reliquary.infrastructure import corpus_job_store as job_store
from tests.unit.test_admin_eval_jobs import _eval_job, _qualification, _qualify, admin  # noqa: F401
from tests.unit.test_eval_grading import GradingEnvironment
from tests.unit.test_jobs_cli import registry  # noqa: F401
from tests.unit.test_order_pinned import _order, pins  # noqa: F401

SET_ID = "logic-frozen-grader"
OLD_GRADER = "a" * 64


def _publish(admin, tmp_path, monkeypatch, *, grader=OLD_GRADER, sealed=True):
    directory = tmp_path / "frozen-grader-set"
    with monkeypatch.context() as patch:
        patch.setattr(protocol, "grader_sha256", lambda: grader)
        card = build_set("logic", count=8, seed=1, set_id=SET_ID, out=directory,
                         open_environment=lambda source, split: GradingEnvironment(source))
    assert card["protocol"]["source"]["grader_sha256"] == grader
    if not sealed:
        card = {key: value for key, value in card.items() if not key.startswith("protocol")}
        (directory / "set.json").write_bytes(protocol.canonical(card))
    asyncio.run(publish_set(directory, platform=LocalDirectorySink(admin.root / "p"),
                            subnet=SubnetEvalStore()))
    return card


def _body(admin, monkeypatch, kind, *, grader=OLD_GRADER):
    if kind == "pinned":
        return _order(eval_set_id=SET_ID)
    # The qualification was accepted while this grader was installed.
    with monkeypatch.context() as patch:
        patch.setattr(protocol, "grader_sha256", lambda: grader)
        assert admin("POST", "/admin/v1/qualifications", _qualification(set_id=SET_ID)).status_code == 201
        _qualify(admin)
    return _eval_job(eval_set_id=SET_ID)


@pytest.mark.parametrize("kind", ["pinned", "qualified"])
@pytest.mark.parametrize("path", ["/admin/v1/jobs/validate", "/admin/v1/jobs"])
def test_new_evaluation_refuses_frozen_grader_drift_before_any_write(
        admin, pins, tmp_path, monkeypatch, kind, path):
    card = _publish(admin, tmp_path, monkeypatch)
    assert protocol.validate_set(card) == card["protocol_sha256"]
    with pytest.raises(ValueError, match="set protocol grader drift"):
        protocol.validate_set(card, current_grader=True)
    body = _body(admin, monkeypatch, kind)
    before = copy.deepcopy(admin.bucket.objects), copy.deepcopy(admin.registry["entries"])

    response = admin("POST", path, body)

    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "set protocol grader drift"
    assert asyncio.run(job_store.read_job(body["job_id"]))[0] is None
    assert asyncio.run(qual.EvalJobStore().read(body["job_id"])) is None
    assert (admin.bucket.objects, admin.registry["entries"]) == before


@pytest.mark.parametrize("kind", ["pinned", "qualified"])
def test_a_current_frozen_grader_still_validates_and_creates(admin, pins, tmp_path, monkeypatch, kind):
    current = protocol.grader_sha256()
    _publish(admin, tmp_path, monkeypatch, grader=current)
    body = _body(admin, monkeypatch, kind, grader=current)
    assert admin("POST", "/admin/v1/jobs/validate", body).status_code == 200
    assert admin("POST", "/admin/v1/jobs", body).status_code == 201


@pytest.mark.parametrize("kind", ["pinned", "qualified"])
def test_an_existing_job_replay_survives_later_grader_drift(admin, pins, tmp_path, monkeypatch, kind):
    _publish(admin, tmp_path, monkeypatch)
    body = _body(admin, monkeypatch, kind)
    with monkeypatch.context() as patch:
        patch.setattr(protocol, "grader_sha256", lambda: OLD_GRADER)
        assert admin("POST", "/admin/v1/jobs", body).status_code == 201
    before = copy.deepcopy(admin.bucket.objects), copy.deepcopy(admin.registry["entries"])
    assert admin("POST", "/admin/v1/jobs/validate", body).status_code == 200
    assert admin("POST", "/admin/v1/jobs", body).status_code == 200
    assert (admin.bucket.objects, admin.registry["entries"]) == before


@pytest.mark.parametrize("kind", ["pinned", "qualified"])
def test_an_unsealed_legacy_card_keeps_its_existing_behavior(admin, pins, tmp_path, monkeypatch, kind):
    _publish(admin, tmp_path, monkeypatch, sealed=False)
    body = _body(admin, monkeypatch, kind)
    assert admin("POST", "/admin/v1/jobs/validate", body).status_code == 200
    assert admin("POST", "/admin/v1/jobs", body).status_code == 201


def test_a_new_qualification_refuses_grader_drift_before_it_is_queued(admin, tmp_path, monkeypatch):
    _publish(admin, tmp_path, monkeypatch)
    before = copy.deepcopy(admin.bucket.objects), copy.deepcopy(admin.registry["entries"])
    response = admin("POST", "/admin/v1/qualifications", _qualification(set_id=SET_ID))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "set protocol grader drift"
    assert asyncio.run(qual.QualificationStore().read("order-q1"))[0] is None
    assert (admin.bucket.objects, admin.registry["entries"]) == before


def test_an_existing_qualification_replay_survives_later_grader_drift(admin, tmp_path, monkeypatch):
    _publish(admin, tmp_path, monkeypatch)
    body = _qualification(set_id=SET_ID)
    with monkeypatch.context() as patch:
        patch.setattr(protocol, "grader_sha256", lambda: OLD_GRADER)
        assert admin("POST", "/admin/v1/qualifications", body).status_code == 201
    before = copy.deepcopy(admin.bucket.objects)
    assert admin("POST", "/admin/v1/qualifications", body).status_code == 200
    assert admin.bucket.objects == before


@pytest.mark.parametrize("path", ["/admin/v1/jobs/validate", "/admin/v1/jobs"])
def test_a_tampered_sealed_card_is_refused_before_any_write(admin, pins, tmp_path, monkeypatch, path):
    card = _publish(admin, tmp_path, monkeypatch, grader=protocol.grader_sha256())
    key = subnet_key(SET_ID, "set.json")
    _, etag = admin.bucket.objects[key]
    admin.bucket.objects[key] = (protocol.canonical({**card, "protocol_sha256": "0" * 64}), etag)
    before = copy.deepcopy(admin.bucket.objects), copy.deepcopy(admin.registry["entries"])
    response = admin("POST", path, _order(eval_set_id=SET_ID))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "protocol digest/body mismatch"
    assert (admin.bucket.objects, admin.registry["entries"]) == before


@pytest.mark.parametrize("path", ["/admin/v1/jobs/validate", "/admin/v1/jobs"])
def test_a_malformed_job_id_still_returns_422(admin, pins, tmp_path, monkeypatch, path):
    _publish(admin, tmp_path, monkeypatch, grader=protocol.grader_sha256())
    before = copy.deepcopy(admin.bucket.objects), copy.deepcopy(admin.registry["entries"])
    response = admin("POST", path, _order(eval_set_id=SET_ID, job_id="order-eval-bad_id"))
    assert response.status_code == 422, response.text
    assert "unusable job id" in response.json()["detail"]
    assert (admin.bucket.objects, admin.registry["entries"]) == before


@pytest.mark.parametrize("malformation", ["missing_file_digest", "source_is_a_list"])
def test_a_malformed_sealed_card_returns_422_before_any_write(
        admin, pins, tmp_path, monkeypatch, malformation):
    card = _publish(admin, tmp_path, monkeypatch, grader=protocol.grader_sha256())
    if malformation == "missing_file_digest":
        del card["grading_sha256"]
    else:
        body = {**card["protocol"], "source": []}
        card.update(protocol=body, protocol_json=protocol.canonical(body).decode(),
                    protocol_sha256=protocol.digest(body))
    key = subnet_key(SET_ID, "set.json")
    _, etag = admin.bucket.objects[key]
    admin.bucket.objects[key] = (protocol.canonical(card), etag)
    before = copy.deepcopy(admin.bucket.objects), copy.deepcopy(admin.registry["entries"])
    response = admin("POST", "/admin/v1/jobs", _order(eval_set_id=SET_ID))
    assert response.status_code == 422, response.text
    assert response.json()["detail"] == "set protocol malformed"
    assert (admin.bucket.objects, admin.registry["entries"]) == before
