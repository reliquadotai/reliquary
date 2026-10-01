"""Code completions are graded by the sandboxed grading service, never by the
package's own runner on the grading host (review C1)."""

from __future__ import annotations

import pytest

from reliquary.corpus import export
from reliquary.corpus.export import job_grader, reward_scorer


class _CodeEnvironment:
    name = "fake-code"

    def get_problem(self, index):
        return {"id": index, "generator_index": index}

    def compute_reward(self, problem, completion):
        raise AssertionError("ran model-written code outside the sandbox")

    def admission_reward_cases(self, problem):
        return [{"input": [problem["id"]], "output": problem["id"]}]


class _CodeSpec:
    name = "fake-code"
    interaction_mode = "single_turn"
    reward_materializer_method = "admission_reward_cases"
    scorer_path = "tests.unit.test_corpus_code_grading_sandbox:_sandbox_scorer"

    def create(self):
        return _CodeEnvironment()


CALLS = []


def _sandbox_scorer(problem, completion_texts, reward_materials=None):
    CALLS.append((problem["id"], list(completion_texts), reward_materials))
    return [0.5 for _ in completion_texts]


def test_a_materials_source_is_scored_through_its_sandboxed_scorer():
    CALLS.clear()
    score = reward_scorer(_CodeSpec(), _CodeEnvironment())
    assert score({"id": 3, "generator_index": 3}, "```python\nx\n```") == 0.5
    assert CALLS == [(3, ["```python\nx\n```"], [{"input": [3], "output": 3}])]


def test_a_plain_source_keeps_its_own_reward():
    class Plain:
        reward_materializer_method = None

    class Env:
        def compute_reward(self, problem, text):
            return 1.0

    assert reward_scorer(Plain(), Env())({}, "x") == 1.0


def test_the_real_code_sources_use_the_admission_scorer():
    from reliquary.environment.registry import ENVIRONMENT_SPECS

    for name in ("reliquary_code_v1", "opencodeinstruct"):
        spec = ENVIRONMENT_SPECS[name]
        assert spec.reward_materializer_method == "admission_reward_cases"
        assert spec.scorer_path == "reliquary.validator.admission:_score_opencode_adapter"


def _job(source="fake-code"):
    from tests.unit.test_corpus_export import _job_spec
    from reliquary.corpus.job import Filter

    return _job_spec(prompt_source=source, filter_=Filter(grader_id="g", threshold=1.0))


def test_job_grader_refuses_code_without_the_grading_service(monkeypatch, tmp_path):
    from reliquary.environment import registry

    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", {"fake-code": _CodeSpec()})
    monkeypatch.setattr(export, "GRADER_SOCKET_PATH", str(tmp_path / "absent.sock"))
    with pytest.raises(ValueError, match="sandboxed grading service"):
        job_grader(_job())


def test_job_grader_scores_code_in_the_sandbox(monkeypatch, tmp_path):
    from reliquary.environment import registry

    socket_path = tmp_path / "grader.sock"
    socket_path.touch()
    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", {"fake-code": _CodeSpec()})
    monkeypatch.setattr(export, "GRADER_SOCKET_PATH", str(socket_path))
    CALLS.clear()
    grade = job_grader(_job())
    assert grade(2, "text") == (False, 0.5)
    assert CALLS[0][0] == 2
