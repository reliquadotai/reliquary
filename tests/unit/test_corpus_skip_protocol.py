"""The skip's wire and its signature: a skip signature is never a submission
signature, and the other way round."""

from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from reliquary.protocol.corpus_submission import (
    CorpusRejectReason,
    CorpusSkipRequest,
    CorpusSubmissionRequest,
)
from reliquary.protocol.signatures import (
    build_corpus_binding,
    build_corpus_skip_binding,
    sign_corpus_skip,
    verify_corpus_signature,
    verify_corpus_skip_signature,
)


def _skip(**overrides):
    body = {"job_id": "math-v1", "miner_hotkey": "5Hot", "cursor": 3,
            "prompt_index": 17, "to_cursor": 9, "signature": "00"}
    body.update(overrides)
    return body


def _submission(**overrides):
    body = {
        "job_id": "math-v1", "miner_hotkey": "5Hot", "cursor": 3, "prompt_index": 17,
        "checkpoint_sha256": "a" * 64, "rendered_prompt": "<prompt>",
        "completions": [{"tokens": [1, 2, 3], "text": "123", "proofs": []}],
        "signature": "00",
    }
    body.update(overrides)
    return body


def test_the_not_full_reason_is_one_name_everywhere():
    from reliquary.corpus.checks import REASON_PROMPT_NOT_FULL

    assert CorpusRejectReason.PROMPT_NOT_FULL.value == REASON_PROMPT_NOT_FULL == "prompt_not_full"


def test_a_skip_parses_and_refuses_unknown_fields():
    CorpusSkipRequest(**_skip())
    with pytest.raises(ValidationError):
        CorpusSkipRequest(**_skip(completions=[]))
    body = _skip()
    del body["to_cursor"]
    with pytest.raises(ValidationError):
        CorpusSkipRequest(**body)
    for bad in ({"cursor": -1}, {"prompt_index": -1}, {"signature": ""}, {"job_id": ""},
                {"to_cursor": 0}):
        with pytest.raises(ValidationError):
            CorpusSkipRequest(**_skip(**bad))


def test_the_skip_binding_ignores_the_signature():
    assert build_corpus_skip_binding(_skip(signature="aa")) == build_corpus_skip_binding(
        _skip(signature="bb"))


@pytest.mark.parametrize("field,value", [
    ("job_id", "math-v2"), ("miner_hotkey", "5Other"), ("cursor", 4), ("prompt_index", 18),
    ("to_cursor", 10),
])
def test_every_skip_field_moves_the_binding(field, value):
    assert build_corpus_skip_binding(_skip(**{field: value})) != build_corpus_skip_binding(_skip())


def test_the_mapping_and_pydantic_paths_bind_the_same_bytes():
    assert build_corpus_skip_binding(_skip()) == build_corpus_skip_binding(
        CorpusSkipRequest(**_skip()))


def test_a_skip_binding_is_never_a_submission_binding():
    assert build_corpus_skip_binding(_skip()) != build_corpus_binding(_submission())


def test_a_bad_hex_skip_signature_does_not_verify():
    assert verify_corpus_skip_signature(_skip(signature="zz")) is False


def test_a_real_keypair_round_trips_a_skip():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    body = _skip(miner_hotkey=keypair.ss58_address, signature="")
    body["signature"] = sign_corpus_skip(SimpleNamespace(hotkey=keypair), body)
    assert verify_corpus_skip_signature(body) is True
    assert verify_corpus_skip_signature(CorpusSkipRequest(**body)) is True
    assert verify_corpus_skip_signature(dict(body, cursor=4)) is False


def test_a_skip_signature_cannot_sign_a_submission_and_vice_versa():
    """Same job, hotkey, cursor and index on both: only the domain tells the
    two apart, and it must."""
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    wallet = SimpleNamespace(hotkey=keypair)
    hotkey = keypair.ss58_address

    skip = _skip(miner_hotkey=hotkey, signature="")
    skip_signature = sign_corpus_skip(wallet, skip)
    as_submission = _submission(miner_hotkey=hotkey, signature=skip_signature)
    assert verify_corpus_signature(as_submission) is False
    assert verify_corpus_signature(CorpusSubmissionRequest(**as_submission)) is False

    submission = _submission(miner_hotkey=hotkey)
    submission_signature = keypair.sign(build_corpus_binding(submission)).hex()
    assert verify_corpus_signature(dict(submission, signature=submission_signature)) is True
    assert verify_corpus_skip_signature(
        _skip(miner_hotkey=hotkey, signature=submission_signature)) is False


def test_sign_corpus_skip_requires_bittensor(monkeypatch):
    from reliquary.protocol import signatures

    monkeypatch.setattr(signatures, "bt", None)
    with pytest.raises(ImportError):
        signatures.sign_corpus_skip(object(), _skip())
    assert signatures.verify_corpus_skip_signature(_skip()) is False


def test_the_skip_domain_is_v2():
    from reliquary.protocol import signatures

    assert signatures.CORPUS_SKIP_DOMAIN == b"reliquary/corpus-skip/v2"
