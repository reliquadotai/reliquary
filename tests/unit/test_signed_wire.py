"""The transcript travels in the trajectory, is bound by the miner's signature under
its own domain, and is kept in the record; replay submissions are byte-identical."""

import pytest
from pydantic import ValidationError

from reliquary.corpus import signed_reasons
from reliquary.corpus.job import parse_job
from reliquary.corpus.trajectory import BuiltTrajectory
from reliquary.protocol.corpus_submission import (
    MAX_TRANSCRIPT_BYTES, CorpusRejectReason, CorpusSubmissionRequest,
)
from reliquary.protocol.signatures import build_corpus_binding
from reliquary.validator.corpus_service import MIN_SUBMIT_BODY_BYTES, worst_case_body
from tests.unit.test_corpus_job_episode import _manifest
from tests.unit.test_corpus_job_signed_sandbox import signed_episode
from tests.unit.test_corpus_route_episode import episode_store  # noqa: F401
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401

TRAJ = {"tokens": [200] * 10, "turns": [{"start": 0, "end": 10, "proofs": ["A" * 200]}],
        "final_diff": "d", "stop": "agent_completed"}
# build_corpus_binding of `request()` on c45f729b, before this plan.
FROZEN_REPLAY_BINDING = "678ea37806b6e2f00522d0a4dd4bf58bcadc8cbc1387a29ba3e0961e7e21cce0"


def request(**trajectory):
    return CorpusSubmissionRequest(
        job_id="j", miner_hotkey="5Hot", cursor=0, prompt_index=0, checkpoint_sha256="c" * 64,
        rendered_prompt="p", trajectory={**TRAJ, **trajectory}, signature="s")


def test_every_sandbox_reason_is_a_wire_reason():
    for reason in signed_reasons.SANDBOX_REASONS:
        assert CorpusRejectReason(reason).value == reason


def test_a_replay_binding_is_unchanged():
    assert build_corpus_binding(request()).hex() == FROZEN_REPLAY_BINDING
    plain = request().model_dump()
    plain["trajectory"].pop("transcript")
    assert build_corpus_binding(plain).hex() == FROZEN_REPLAY_BINDING


def test_the_transcript_is_bound_under_its_own_domain():
    one = build_corpus_binding(request(transcript={"token": {}, "records": [1]}))
    two = build_corpus_binding(request(transcript={"token": {}, "records": [2]}))
    assert one != two and one.hex() != FROZEN_REPLAY_BINDING


def test_an_oversized_transcript_is_refused_by_the_wire():
    with pytest.raises(ValidationError, match="transcript"):
        request(transcript={"token": {}, "records": ["x" * MAX_TRANSCRIPT_BYTES]})


def test_a_built_trajectory_carries_its_transcript_only_when_signed():
    built = BuiltTrajectory(prompt_ids=(1,), tokens=(2, 3), spans=((0, 2),), proofs=(("p",),),
                            final_diff="", stop="agent_completed")
    assert "transcript" not in built.wire()
    signed = BuiltTrajectory(**{**built.__dict__, "transcript": {"token": {}, "records": []}})
    assert signed.wire()["transcript"] == {"token": {}, "records": []}


def test_a_signed_job_allows_the_transcript_in_its_body_cap():
    replay = parse_job(_manifest())
    signed = parse_job(_manifest(episode=signed_episode()))
    assert worst_case_body(replay) == MIN_SUBMIT_BODY_BYTES
    assert worst_case_body(signed) == MIN_SUBMIT_BODY_BYTES + MAX_TRANSCRIPT_BYTES


def test_a_replay_record_has_no_transcript_key(episode_store):  # noqa: F811
    from tests.unit.test_corpus_route_episode import _client, _post, _request
    from tests.unit.test_corpus_route_records import _Records

    records, accepted = _Records(), []
    assert _post(_client(episode_store, records, accepted), _request())["reason"] == "accepted"
    (record,) = records.written.values()
    assert "transcript" not in record["completions"][0]
