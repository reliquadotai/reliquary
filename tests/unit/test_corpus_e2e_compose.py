"""The rehearsal (scripts/corpus_e2e.py) declares a composed task with
--compose, as `jobs create` without --from-profile does; --base-profile stays."""

from __future__ import annotations

import json

import pytest

from reliquary.cli.main import _corpus_base_profile, build_corpus_task_entry
from scripts import corpus_e2e


def _args(*argv):
    return corpus_e2e.build_parser().parse_args(["run", *argv])


def _declare(tmp_path, args, **kwargs):
    kwargs.setdefault("renderer", None)
    return corpus_e2e.declare_task(
        tmp_path, args, task_id="corpus-e2e", job_id="e2e", revision="r" * 40, **kwargs
    )


def test_compose_declares_what_jobs_create_would(tmp_path):
    args = _args("--compose", "--prompt-source", "reliquary_stateful_tools_v1")
    contract = _declare(tmp_path, args)
    base = _corpus_base_profile(
        task_id="corpus-e2e", from_profile=None, model=args.honest_model,
        model_revision="r" * 40, model_architecture=args.model_architecture,
        prompt_encoding=None, renderer_id=None, prompt_source="reliquary_stateful_tools_v1",
    )
    expected = build_corpus_task_entry(
        task_id="corpus-e2e", job_id="e2e", base=base, model_id=args.honest_model,
        model_revision="r" * 40, model_architecture=args.model_architecture,
        prompt_source="reliquary_stateful_tools_v1", cap=args.cap, overrides={},
    ).contract
    assert contract == expected
    assert contract["protocol_version"] == 9
    assert json.loads((tmp_path / "contract.json").read_text()) == contract


def test_compose_takes_the_encoding_from_the_renderer_unless_named(tmp_path):
    args = _args("--compose", "--prompt-source", "reliquary_stateful_tools_v1")
    assert _declare(tmp_path, args, renderer="chat-template-v1")["prompt_encoding"] == "chat_template"
    named = _args("--compose", "--prompt-source", "reliquary_stateful_tools_v1",
                  "--prompt-encoding", "raw")
    assert _declare(tmp_path, named, renderer="chat-template-v1")["prompt_encoding"] == "raw"


def test_base_profile_still_seeds_from_the_template(tmp_path):
    args = _args("--base-profile", "qwen3-4b-reliquary-episode-v7-dev1",
                 "--prompt-source", "reliquary_stateful_tools_v1")
    contract = _declare(tmp_path, args)
    assert contract["protocol_version"] == 7


@pytest.mark.parametrize("extra", [
    ["--base-profile", "qwen3-4b-reliquary-episode-v7-dev1"],
    ["--second-base-profile", "teutonic-9b-reliquary-suite-v9-dev1"],
])
def test_compose_refuses_a_template_flag(extra):
    with pytest.raises(SystemExit, match="--compose"):
        corpus_e2e.check_compose_flags(_args("--compose", *extra))


def test_prompt_encoding_without_compose_is_refused():
    with pytest.raises(SystemExit, match="--prompt-encoding"):
        corpus_e2e.check_compose_flags(_args("--prompt-encoding", "raw"))
