"""reliquary_general_v1 as a corpus prompt source, stage A.

The package (reliquary-environments, `reliquary-general`) serves general and
tool-use prompts in (block, thinking mode) segments, each opening with its
single-turn rows. Stage A serves only single-turn rows of chat, safety, and the
single-turn parts of ifeval and structured: a corpus job renders `prompt` as one
user turn, so a row with a history, tools or a system message would be served
as a different task. Twice refused: `jobs create` refuses any range outside the
stage-A ranges, and the prompt source refuses any row whose metadata says it is
not one plain user turn, whatever the job declares.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import pytest

from reliquary.environment.agentic.types import canonical_json
from reliquary.environment.registry import ENVIRONMENT_SPECS

NAME = "reliquary_general_v1"
FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "general"

# The package's train segments (README of reliquary_general, dataset
# ReliquaryForge/general-prompts-curated@02b3ef7a): (start, single-turn stop, stop).
STAGE_A = {
    "chat/direct": (0, 30141),
    "chat/thinking": (30141, 51965),
    "ifeval/direct": (66291, 70971),
    "ifeval/thinking": (71720, 82808),
    "structured/direct": (84587, 86967),
    "structured/thinking": (87540, 93067),
    "safety/direct": (94476, 101778),
    "safety/thinking": (101778, 107165),
}
STAGE_B = {
    "multiturn": (51965, 66291),
    "ifeval/direct multi-turn": (70971, 71720),
    "ifeval/thinking multi-turn": (82808, 84587),
    "structured/direct with a system": (86967, 87540),
    "structured/thinking with a system": (93067, 94476),
    "clarification and identity": (107165, 110643),
    "tools": (110643, 122638),
}


def test_the_source_is_registered_as_a_binary_checked_answer() -> None:
    spec = ENVIRONMENT_SPECS[NAME]
    assert spec.factory_path == "reliquary_general:GeneralPromptsEnvironment"
    assert spec.scorer_path == (
        "reliquary.environment.agentic.external:score_external_answers"
    )
    assert spec.contract_version == "reliquary/checked-answer/v1"
    # The graders read text: IFEvalG verifiers, a parse and a JSON Schema. No
    # model-written code runs, so the CPU worker is the right place.
    assert spec.admission_resource_class == "cpu"
    assert spec.final_answer_policy == "text"
    assert (spec.reward_lattice_policy, spec.attainable_rewards) == (
        "binary-v1", (0.0, 1.0))
    assert spec.reward_materializer_method is None
    assert spec.external_distribution == "reliquary-general"
    assert spec.interaction_mode == "single_turn"


def test_the_spec_pins_the_packaged_artifact() -> None:
    artifact = json.loads((FIXTURES / "artifact.json").read_text())
    spec = ENVIRONMENT_SPECS[NAME]
    assert artifact["environment"] == NAME
    assert artifact["contract"] == spec.contract_version
    assert artifact["entrypoints"]["replay"] == spec.factory_path
    digest = hashlib.sha256(canonical_json(artifact).encode()).hexdigest()
    assert digest == spec.environment_manifest_sha256
    # The corpus is the pinned Hub dataset, not a file of the wheel.
    assert not any(name.endswith(".jsonl.gz") for name in artifact["files"])


def test_catalog_and_profile_declare_it() -> None:
    from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG
    from reliquary.protocol.profiles import resolve_protocol_profile

    spec = ENVIRONMENT_SPECS[NAME]
    profile = resolve_protocol_profile("teutonic-9b-reliquary-suite-v9-dev1")
    for body in (ENVIRONMENT_CATALOG[NAME], profile.environments[NAME]):
        assert body.environment_manifest_sha256 == spec.environment_manifest_sha256
        assert body.environment_contract_id == spec.contract_version
        assert body.answer_format == "text"
        # The longest segment budget the package suggests (chat, thinking).
        assert body.max_new_tokens == 16384
        assert body.prompt_template.template_id == "reliquary-external-prompt-v1"


def test_lineage_names_the_pinned_dataset() -> None:
    from reliquary.eval.sets import lineage

    assert lineage(NAME) == (
        "ReliquaryForge/general-prompts-curated@02b3ef7a", NAME)


@pytest.mark.parametrize("segment", sorted(STAGE_A))
def test_a_stage_a_range_can_be_declared(segment) -> None:
    from reliquary.eval.sets import refuse_held_out_overlap

    start, stop = STAGE_A[segment]
    refuse_held_out_overlap(NAME, start, stop - start)
    refuse_held_out_overlap(NAME, start + 10, 100)  # a slice of one


@pytest.mark.parametrize("part", sorted(STAGE_B))
def test_a_stage_b_range_is_refused(part) -> None:
    from reliquary.eval.sets import refuse_held_out_overlap

    start, stop = STAGE_B[part]
    for begin, count in ((start, stop - start), (start, 1), (stop - 1, 1)):
        with pytest.raises(ValueError, match="stage A"):
            refuse_held_out_overlap(NAME, begin, count)


def test_a_range_spanning_two_stage_a_segments_is_refused() -> None:
    """chat/direct and chat/thinking touch, but one job has one renderer."""
    from reliquary.eval.sets import refuse_held_out_overlap

    with pytest.raises(ValueError, match="stage A"):
        refuse_held_out_overlap(NAME, 30000, 200)
    with pytest.raises(ValueError, match="stage A"):
        refuse_held_out_overlap(NAME, 70900, 100)  # into ifeval's multi-turn part


class _Environment:
    """Rows shaped as the package's task metadata shapes them."""

    def __init__(self, rows) -> None:
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def get_problem(self, index: int) -> dict:
        return {"prompt": f"question {index}", "environment": NAME, **self._rows[index]}


def _plain(index: int, renderer: str = "chat-template-v1") -> dict:
    return {
        "single_turn": True,
        "tools": None,
        "messages": [{"role": "user", "content": f"question {index}"}],
        "renderer": renderer,
        "system": "You are Teutonic.",
        "system_scope": "generation-only",
    }


def _job(count: int, renderer: str = "chat-template-v1"):
    from tests.unit.test_corpus_export import _job_spec

    return dataclasses.replace(
        _job_spec(prompt_source=NAME, prompt_count=count), renderer_id=renderer)


def test_a_single_turn_row_is_served_as_its_prompt_alone() -> None:
    from reliquary.validator.corpus_service import SingleTurnPromptJob

    prompts = SingleTurnPromptJob(_job(1), _Environment([_plain(0)]))
    task = prompts.task_for(0)
    assert task.prompt == "question 0"
    # The generation system is not rendered in stage A.
    assert task.metadata == {}


@pytest.mark.parametrize("why, row", [
    ("history", {**_plain(0), "single_turn": False, "messages": [
        {"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"},
        {"role": "user", "content": "question 0"}]}),
    ("task system", {**_plain(0), "single_turn": False, "messages": [
        {"role": "system", "content": "Answer in YAML."},
        {"role": "user", "content": "question 0"}]}),
    ("tools", {**_plain(0), "single_turn": False, "tools": [{"name": "f"}]}),
    # Each lock holds on its own, even against a row that claims single_turn.
    ("tools alone", {**_plain(0), "tools": [{"name": "f"}]}),
    ("messages alone", {**_plain(0), "messages": [
        {"role": "system", "content": "s"}, {"role": "user", "content": "question 0"}]}),
    ("a different user turn", {**_plain(0), "messages": [
        {"role": "user", "content": "another question"}]}),
])
def test_a_row_that_is_not_one_plain_user_turn_is_never_served(why, row) -> None:
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError,
        SingleTurnPromptJob,
    )

    prompts = SingleTurnPromptJob(_job(1), _Environment([row]))
    with pytest.raises(CorpusPromptSourceError, match="one user turn"):
        prompts.task_for(0)


def test_a_row_is_served_only_through_the_renderer_it_was_drawn_for() -> None:
    from reliquary.validator.corpus_service import (
        CorpusPromptSourceError,
        SingleTurnPromptJob,
    )

    rows = [_plain(0, "chat-template-thinking-v1")]
    assert SingleTurnPromptJob(
        _job(1, "chat-template-thinking-v1"), _Environment(rows)).task_for(0)
    with pytest.raises(CorpusPromptSourceError, match="renderer"):
        SingleTurnPromptJob(_job(1, "chat-template-v1"), _Environment(rows)).task_for(0)


def test_a_source_without_these_fields_is_unaffected() -> None:
    from reliquary.validator.corpus_service import SingleTurnPromptJob

    class Plain:
        def __len__(self) -> int:
            return 1

        def get_problem(self, index: int) -> dict:
            return {"prompt": "p", "environment": "other"}

    assert SingleTurnPromptJob(_job(1), Plain()).task_for(0).prompt == "p"
