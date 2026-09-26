"""The three encodings a miner and a validator must agree on byte for byte."""

import hashlib

import pytest

from reliquary.corpus.encoding import checkpoint_fingerprint, completion_text, prompt_token_ids
from reliquary.validator.corpus_text import check_text_matches_tokens

EOS = 99


class _Tokenizer:
    def encode(self, text, add_special_tokens=True):
        ids = [ord(c) for c in text]
        return ([1] + ids) if add_special_tokens else ids

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return "".join(chr(i) for i in ids)


class _EncodingTokenizer(_Tokenizer):
    def encode(self, text, add_special_tokens=True):
        class _Encoding:
            ids = [ord(c) for c in text]
        return _Encoding()


def test_the_prompt_is_encoded_without_special_tokens():
    assert prompt_token_ids(_Tokenizer(), "ab") == [97, 98]


def test_a_tokenizers_encoding_is_unwrapped():
    assert prompt_token_ids(_EncodingTokenizer(), "ab") == [97, 98]


@pytest.mark.parametrize("tokens", [[104, 105], [104, 105, EOS], [EOS, 104]])
def test_the_completion_text_is_what_the_route_accepts(tokens):
    text = completion_text(_Tokenizer(), tokens, EOS)
    assert check_text_matches_tokens(tokens, text, tokenizer=_Tokenizer(), eos_token_id=EOS).ok


def test_the_fingerprint_covers_every_shard_by_name_and_content(tmp_path):
    (tmp_path / "b.safetensors").write_bytes(b"two")
    (tmp_path / "a.safetensors").write_bytes(b"one")
    (tmp_path / "config.json").write_bytes(b"ignored")
    expected = hashlib.sha256(
        b"a.safetensors\0" + hashlib.sha256(b"one").hexdigest().encode() + b"\n"
        + b"b.safetensors\0" + hashlib.sha256(b"two").hexdigest().encode() + b"\n"
    ).hexdigest()
    assert checkpoint_fingerprint(tmp_path) == expected
    (tmp_path / "b.safetensors").write_bytes(b"TWO")
    assert checkpoint_fingerprint(tmp_path) != expected


def test_a_directory_without_shards_has_no_fingerprint(tmp_path):
    with pytest.raises(ValueError, match="safetensors"):
        checkpoint_fingerprint(tmp_path)


def test_the_cli_prints_the_fingerprint_of_a_local_directory(tmp_path):
    from typer.testing import CliRunner

    from reliquary.cli.main import app

    (tmp_path / "a.safetensors").write_bytes(b"one")
    result = CliRunner().invoke(app, ["jobs", "fingerprint", str(tmp_path)])
    assert result.exit_code == 0
    assert result.output.strip() == checkpoint_fingerprint(tmp_path)
