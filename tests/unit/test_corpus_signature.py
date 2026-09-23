"""The corpus submission's signature binds everything the validator pays on."""

import pytest

from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from reliquary.protocol.signatures import (
    build_corpus_binding,
    corpus_submission_id,
    verify_corpus_signature,
)


def _request(**overrides):
    body = {
        "job_id": "math-v1",
        "miner_hotkey": "5Hot",
        "cursor": 3,
        "prompt_index": 17,
        "checkpoint_sha256": "a" * 64,
        "rendered_prompt": "<prompt>",
        "completions": [{"tokens": [1, 2, 3], "text": "123", "proofs": []}],
        "signature": "00",
    }
    body.update(overrides)
    return CorpusSubmissionRequest(**body)


def test_the_binding_ignores_the_signature_itself():
    assert build_corpus_binding(_request(signature="aa")) == build_corpus_binding(
        _request(signature="bb")
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("job_id", "math-v2"),
        ("miner_hotkey", "5Other"),
        ("cursor", 4),
        ("prompt_index", 18),
        ("checkpoint_sha256", "b" * 64),
        ("rendered_prompt", "<other>"),
        ("completions", [{"tokens": [1, 2, 4], "text": "123", "proofs": []}]),
        ("completions", [{"tokens": [1, 2, 3], "text": "124", "proofs": []}]),
        ("completions", [{"tokens": [1, 2, 3], "text": "123", "proofs": ["AAAA"]}]),
    ],
)
def test_every_paid_field_moves_the_binding(field, value):
    assert build_corpus_binding(_request(**{field: value})) != build_corpus_binding(
        _request()
    )


def test_completion_order_is_bound():
    a = [{"tokens": [1], "text": "1", "proofs": []}, {"tokens": [2], "text": "2", "proofs": []}]
    assert build_corpus_binding(_request(completions=a)) != build_corpus_binding(
        _request(completions=list(reversed(a)))
    )


def test_the_submission_id_is_the_binding_in_hex():
    request = _request()
    assert corpus_submission_id(request) == build_corpus_binding(request).hex()
    assert len(corpus_submission_id(request)) == 64


def test_a_signature_that_is_not_hex_does_not_verify():
    assert verify_corpus_signature(_request(signature="zz")) is False


def test_a_real_keypair_round_trips():
    bt = pytest.importorskip("bittensor")
    keypair = bt.Keypair.create_from_mnemonic(bt.Keypair.generate_mnemonic())
    request = _request(miner_hotkey=keypair.ss58_address)
    signature = keypair.sign(build_corpus_binding(request)).hex()
    signed = _request(miner_hotkey=keypair.ss58_address, signature=signature)
    assert verify_corpus_signature(signed) is True
    assert verify_corpus_signature(_request(miner_hotkey=keypair.ss58_address, signature=signature, cursor=4)) is False
