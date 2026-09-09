"""Exercise the pinned Transformers writer with malicious template names."""
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from reliquary.shared.modeling import save_tokenizer


def test_real_tokenizer_cannot_write_outside_snapshot(tmp_path):
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")))
    output = tmp_path / "snapshot"
    for name in ("../../PWNED", str(tmp_path / "PWNED"), "..\\PWNED", "C:PWNED", "bad\0name"):
        tokenizer.chat_template = {"default": "{{'a'}}", name: "attacker content"}
        with pytest.raises(ValueError, match="template filename"):
            save_tokenizer(tokenizer, output)
        assert not (tmp_path / "PWNED.jinja").exists()
        assert not output.exists()
    tokenizer.chat_template = {"default": "{{'a'}}", "tools": "{{'b'}}"}
    save_tokenizer(tokenizer, output)
    assert (output / "chat_template.jinja").read_text() == "{{'a'}}"
    assert (output / "additional_chat_templates/tools.jinja").read_text() == "{{'b'}}"
    assert PreTrainedTokenizerFast.from_pretrained(output).chat_template == tokenizer.chat_template
