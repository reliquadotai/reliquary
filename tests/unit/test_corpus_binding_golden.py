"""The single-turn corpus binding, pinned byte for byte.

Every deployed miner's submission id and signature are this digest. These
bytes were computed from the code BEFORE the trajectory refactor (Ruling P5);
if this fails, the refactor changed what miners sign.
"""

from reliquary.protocol.signatures import build_corpus_binding, corpus_submission_id

GOLDEN_BINDING_HEX = "bd6fcd3cedb5744387ef22ffff6327b9953b269e372ca21e973ded94cf8f9678"

BODY = {
    "job_id": "math-v1", "miner_hotkey": "5Hot", "cursor": 3, "prompt_index": 17,
    "checkpoint_sha256": "a" * 64, "rendered_prompt": "<prompt>",
    "completions": [
        {"tokens": [1, 2, 3, 70000], "text": "123 é", "proofs": ["A" * 344, "B" * 344]},
        {"tokens": [9], "text": "x", "proofs": []},
    ],
    "signature": "00",
}


def test_the_single_turn_binding_bytes_are_pinned():
    assert build_corpus_binding(BODY).hex() == GOLDEN_BINDING_HEX


def test_the_submission_id_and_signature_input_are_the_pinned_digest():
    assert corpus_submission_id(BODY) == GOLDEN_BINDING_HEX
    assert len(build_corpus_binding(BODY)) == 32   # what wallet.hotkey.sign receives


def test_a_model_dump_with_an_unset_trajectory_binds_like_the_dict():
    from reliquary.protocol.corpus_submission import CorpusSubmissionRequest

    body = {**BODY, "completions": [
        {"tokens": [1, 2, 3, 70000], "text": "123 \u00e9", "proofs": ["A" * 344]},
        {"tokens": [9], "text": "x", "proofs": []}]}
    assert build_corpus_binding(CorpusSubmissionRequest(**body)) == build_corpus_binding(body)
