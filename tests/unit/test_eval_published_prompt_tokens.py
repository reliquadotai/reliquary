"""The public set card keeps the ordered context lengths used to size evaluation jobs."""
from __future__ import annotations

import asyncio
import json

import pytest

from reliquary.corpus.delivery import LocalDirectorySink
from reliquary.eval import protocol, verifiers_source
from reliquary.eval.storage import SetConflict, publish_set
from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle, opener


class _Tokenizer:
    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize):
        assert add_generation_prompt and tokenize
        return range(int(messages[-1]["content"]))


def _built(tmp_path, *, tokenizer="fixture/tokenizer"):
    handle = fake_handle([FakeTask(i, prompt=str(n)) for i, n in enumerate((700, 12000, 9000, 16000))])
    directory = tmp_path / "set"
    card = verifiers_source.build_set("verifiers:fake", out=directory, start=0, count=None,
                            sample=None, seed=None, set_id=None, taskset_args={}, open_taskset=opener(handle),
                            count_tokens=tokenizer, load_tokenizer=lambda _: _Tokenizer(),
                            protocol={"data": {"fixture.json": "1" * 64},
                                "licenses": [{"id": "MIT", "sha256": "2" * 64}],
                                "disjointness": {"disjoint": False, "evidence_sha256": "3" * 64}})
    return directory, card


def _snapshot(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def test_reviewed_publication_preserves_ordered_prompt_lengths_and_is_create_only(tmp_path):
    directory, card = _built(tmp_path)
    frozen = _snapshot(directory)
    platform, subnet = LocalDirectorySink(tmp_path / "public"), LocalDirectorySink(tmp_path / "private")
    answer = asyncio.run(publish_set(directory, platform=platform, subnet=subnet, publish_reviewed=True))
    public_path = tmp_path / "public" / "eval-sets" / card["set_id"] / "set.json"
    public = json.loads(public_path.read_bytes())

    assert public["prompt_tokens"] == {
        "chat_template": True, "per_problem": [700, 12000, 9000, 16000],
        "mean": 9425, "max": 16000,
    }
    # The first two ordered problems need 12,000 prompt tokens; the full set needs 16,000.
    assert max(public["prompt_tokens"]["per_problem"][:2]) == 12000
    assert public["prompt_tokens"]["max"] == 16000
    assert protocol.validate_set(public, current_grader=True) == card["protocol_sha256"]
    assert answer["registration"]["availability"] == "withdrawn"
    published = _snapshot(tmp_path / "public"), _snapshot(tmp_path / "private")
    again = asyncio.run(publish_set(directory, platform=platform, subnet=subnet, publish_reviewed=True))
    assert again["written"] == []
    assert (_snapshot(tmp_path / "public"), _snapshot(tmp_path / "private")) == published
    assert _snapshot(directory) == frozen


def test_reviewed_publication_keeps_tokenizer_provenance_private(tmp_path):
    directory, card = _built(tmp_path, tokenizer="private/cached-tokenizer")
    card["prompt_tokens"].update(revision="a" * 40, renderer_id="chat-template-v1",
                                 extra={"path": "/private/fixture"})
    card_path = directory / "set.json"
    card_path.write_bytes(protocol.canonical(card))
    frozen = card_path.read_bytes()
    platform, subnet = LocalDirectorySink(tmp_path / "public"), LocalDirectorySink(tmp_path / "private")

    asyncio.run(publish_set(directory, platform=platform, subnet=subnet, publish_reviewed=True))

    public_path = tmp_path / "public" / "eval-sets" / card["set_id"] / "set.json"
    private_path = tmp_path / "private" / "reliquary" / "eval-sets" / card["set_id"] / "set.json"
    assert json.loads(public_path.read_bytes())["prompt_tokens"] == {
        "chat_template": True, "per_problem": [700, 12000, 9000, 16000],
        "mean": 9425, "max": 16000,
    }
    assert private_path.read_bytes() == card_path.read_bytes() == frozen


def test_republication_refuses_a_historical_public_card_without_prompt_lengths(tmp_path):
    directory, card = _built(tmp_path)
    platform, subnet = LocalDirectorySink(tmp_path / "public"), LocalDirectorySink(tmp_path / "private")
    asyncio.run(publish_set(directory, platform=platform, subnet=subnet, publish_reviewed=True))
    public_path = tmp_path / "public" / "eval-sets" / card["set_id"] / "set.json"
    historical = json.loads(public_path.read_bytes())
    historical.pop("prompt_tokens", None)
    assert protocol.validate_set(historical) == card["protocol_sha256"]
    # Seed the exact older public serialization: publication must never upgrade this object in place.
    public_path.write_bytes(protocol.canonical(historical))
    frozen = _snapshot(directory), _snapshot(tmp_path / "public"), _snapshot(tmp_path / "private")

    with pytest.raises(SetConflict):
        asyncio.run(publish_set(directory, platform=platform, subnet=subnet, publish_reviewed=True))
    assert (_snapshot(directory), _snapshot(tmp_path / "public"), _snapshot(tmp_path / "private")) == frozen
