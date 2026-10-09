"""Only finished completions are delivered: EOS last, and in a thinking render
exactly one ``</think>``. The filter grades the answer after it, and a
delivery may keep the finished-but-wrong traces, annotated, instead of
omitting them."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from reliquary.corpus.delivery import LocalDirectorySink, delivery_rows, export_delivery
from reliquary.corpus.export import (
    UNFINISHED_REASONS,
    answer_text,
    export_rows,
    job_grader,
    unfinished_reason,
)

EOS = 248046
THINK = "chat-template-thinking-v1"
DIRECT = "chat-template-v1"


def _sid(i: int) -> str:
    return f"{i:064x}"


def _done(text, *, eos=True):
    return {"text": text, "tokens": [5, 6, EOS] if eos else [5, 6, 7]}


# One passing submission per prompt; the completions say what each case is.
THINKING_COMPLETIONS = {
    0: _done("reason \\boxed{1}\n</think>\n\nAnswer \\boxed{42}"),   # finished, right
    1: _done("reason \\boxed{42}\n</think>\n\nAnswer \\boxed{7}"),   # finished, wrong answer
    2: _done("reason \\boxed{42} and then cut", eos=False),          # truncated at the cap
    3: _done("reason \\boxed{42} and EOS but never closed"),          # EOS, no </think>
    4: _done("a</think>b</think>\\boxed{42}"),                        # two closes
}


class _Records:
    def __init__(self, completions):
        self.subs = {_sid(i): {"hotkey": f"HK{i}", "prompt_index": i,
                               "rendered_prompt": f"q{i}<think>\n", "completions": [c]}
                     for i, c in completions.items()}
        self.verdicts = {sid: {"passed": True} for sid in self.subs}

    async def list_verdict_ids(self, job_id):
        return sorted(self.verdicts)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def read_submission(self, job_id, sid):
        return self.subs.get(sid)


def _job(renderer=THINK, **kw):
    fields = {"job_id": "hard-math-v1", "filter": None, "renderer_id": renderer,
              "eos_token_id": EOS, "episode": None, "prompt_start": 0, "prompt_count": 100,
              "prompt_end": 100, "prompt_source": "fixture",
              "owns": lambda index: 0 <= index < 100,
              "to_contract": lambda: {"job_id": "hard-math-v1"}}
    return SimpleNamespace(**{**fields, **kw})


def _right(prompt_index, text):
    """The answer region holds 42: what a boxed grader given the sliced text sees."""
    ok = "\\boxed{42}" in answer_text(text, thinking=True)
    return ok, 1.0 if ok else 0.0


# --------------------------------------------------------------------------
# The rule
# --------------------------------------------------------------------------


@pytest.mark.parametrize("completion,thinking,reason", [
    (_done("x</think>y"), True, None),
    (_done("x</think>y", eos=False), True, "unfinished_no_eos"),
    ({"text": "x</think>y", "tokens": []}, True, "unfinished_no_eos"),
    ({"text": "x</think>y"}, True, "unfinished_no_eos"),
    (_done("no close"), True, "unclosed_think"),
    (_done("a</think>b</think>c"), True, "multiple_think_close"),
    (_done("a plain answer"), False, None),
    (_done("a plain answer", eos=False), False, "unfinished_no_eos"),
])
def test_unfinished_reason(completion, thinking, reason):
    assert unfinished_reason(completion, eos_token_id=EOS, thinking=thinking) == reason
    assert reason is None or reason in UNFINISHED_REASONS


def test_the_answer_of_a_thinking_render_is_what_follows_think_close():
    assert answer_text("r \\boxed{1}</think>\n\\boxed{2}", thinking=True) == "\n\\boxed{2}"
    assert answer_text("never closed \\boxed{1}", thinking=True) == ""
    assert answer_text("direct \\boxed{1}", thinking=False) == "direct \\boxed{1}"


# --------------------------------------------------------------------------
# delivery_rows (export v2)
# --------------------------------------------------------------------------


def _rows(job, records, **kw):
    async def go():
        counts = {}
        rows = [r async for r in delivery_rows(job=job, records=records, counts=counts, **kw)]
        return rows, counts

    return asyncio.run(go())


def test_unfinished_completions_are_dropped_before_grading_and_counted():
    graded = []

    def grade(prompt_index, text):
        graded.append(prompt_index)
        return _right(prompt_index, text)

    rows, counts = _rows(_job(), _Records(THINKING_COMPLETIONS), grade=grade)
    assert [(r["prompt_index"], r["accepted"]) for r in rows] == [(0, True), (1, False)]
    assert sorted(graded) == [0, 1]                 # the unfinished were never graded
    assert counts["completions"] == 5 and counts["rows"] == 2
    assert counts["rows_accepted"] == 1
    assert counts["unfinished"] == {"unfinished_no_eos": 1, "unclosed_think": 1,
                                    "multiple_think_close": 1}


def test_a_direct_render_needs_eos_but_no_think_close():
    completions = {0: _done("Answer in three words."), 1: _done("cut", eos=False)}
    rows, counts = _rows(_job(DIRECT), _Records(completions))
    assert [r["completion"] for r in rows] == ["Answer in three words."]
    assert counts["unfinished"] == {"unfinished_no_eos": 1, "unclosed_think": 0,
                                    "multiple_think_close": 0}


# --------------------------------------------------------------------------
# export_delivery: report counts and the keep-rejected option
# --------------------------------------------------------------------------


class _RawSource:
    def task_for(self, index):
        return SimpleNamespace(prompt=f"Raw question {index}?", metadata={})


def _deliver(tmp_path, *, job, **kw):
    sink = LocalDirectorySink(tmp_path / "bucket")
    manifest = asyncio.run(export_delivery(
        job=job, records=_Records(THINKING_COMPLETIONS), sink=sink, delivery_id="d1",
        work_dir=tmp_path / "work", clock=lambda: 1.0, instruction_source=_RawSource(), **kw))
    root = tmp_path / "bucket" / "deliveries" / "d1"
    report = json.loads((root / "report.json").read_text())
    instructions = [json.loads(line) for shard in manifest["instruction_shards"]
                    for line in (tmp_path / "bucket" / shard["key"]).read_text().splitlines()]
    return manifest, report, instructions


FILTERED = {"filter": SimpleNamespace(grader_id="hard-math", threshold=1.0)}


def test_a_filtered_delivery_omits_the_wrong_and_reports_the_unfinished(tmp_path):
    manifest, report, instructions = _deliver(tmp_path, job=_job(**FILTERED), grade=_right)
    assert manifest["rows"] == 2
    assert report["counts"]["unfinished"] == {"unfinished_no_eos": 1, "unclosed_think": 1,
                                              "multiple_think_close": 1}
    assert report["filter"]["keep_rejected"] is False
    assert report["instruction"]["policy"] == "accepted_completions"
    assert report["instruction"]["omitted"] == {"filter_rejected": 1}
    assert [r["prompt"] for r in instructions] == ["Raw question 0?"]


def test_keep_rejected_delivers_finished_wrong_traces_annotated(tmp_path):
    manifest, report, instructions = _deliver(tmp_path, job=_job(**FILTERED), grade=_right,
                                              keep_rejected=True)
    assert manifest["rows"] == 2
    assert report["filter"]["applied"] is True and report["filter"]["keep_rejected"] is True
    assert report["instruction"]["policy"] == "finished_completions_annotated"
    assert report["instruction"]["omitted"] == {}
    assert report["instruction"]["rejected_kept"] == 1
    assert [r["prompt"] for r in instructions] == ["Raw question 0?", "Raw question 1?"]
    # Unfinished traces are never kept, whatever the option.
    assert report["counts"]["unfinished"]["unfinished_no_eos"] == 1


def test_keep_rejected_without_a_grader_is_refused(tmp_path):
    with pytest.raises(ValueError, match="keep_rejected"):
        _deliver(tmp_path, job=_job(), keep_rejected=True)


# --------------------------------------------------------------------------
# export_rows (the `jobs export` CLI) and job_grader
# --------------------------------------------------------------------------


def test_export_rows_skips_unfinished_and_counts_them():
    async def go():
        counts = {}
        rows = [r async for r in export_rows(job=_job(), records=_Records(THINKING_COMPLETIONS),
                                             grade=_right, counts=counts)]
        return rows, counts

    rows, counts = asyncio.run(go())
    assert [(r["prompt_index"], r["accepted"]) for r in rows] == [(0, True), (1, False)]
    assert counts == {"completions": 5, "rows": 2, "rows_accepted": 1,
                      "unfinished": {"unfinished_no_eos": 1, "unclosed_think": 1,
                                     "multiple_think_close": 1}}


class _SeenEnvironment:
    def __init__(self):
        self.seen = []

    def get_problem(self, index):
        return {"id": index}

    def compute_reward(self, problem, completion):
        self.seen.append(completion)
        return 1.0 if "\\boxed{42}" in completion else 0.0


def _job_spec(renderer):
    from reliquary.corpus.job import Filter, JobSpec, Sampling

    return JobSpec(
        job_id="hard-math-v1", checkpoint_repo="org/Frozen", checkpoint_revision="abc123",
        checkpoint_sha256="a" * 64, prompt_source="fake-env", prompt_count=10,
        renderer_id=renderer, eos_token_id=EOS,
        sampling=Sampling(temperature=1.0, top_p=1.0, top_k=0, min_new_tokens=2,
                          max_new_tokens=64, n=1),
        slots_per_prompt=1, filter=Filter(grader_id="fake", threshold=0.5),
        prompt_order="free", deadline_round=None, prompt_start=0)


@pytest.mark.parametrize("renderer,graded", [
    (THINK, "\n\nAnswer \\boxed{7}"),
    (DIRECT, "reason \\boxed{42}\n</think>\n\nAnswer \\boxed{7}"),
])
def test_job_grader_scores_the_answer_after_think_close(monkeypatch, renderer, graded):
    import reliquary.validator.corpus_service  # noqa: F401  (bind the real catalog first)
    from reliquary.environment import registry

    environment = _SeenEnvironment()
    spec = SimpleNamespace(interaction_mode="single_turn", create=lambda: environment)
    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS", {"fake-env": spec})
    grade = job_grader(_job_spec(renderer))
    accepted, score = grade(0, "reason \\boxed{42}\n</think>\n\nAnswer \\boxed{7}")
    assert environment.seen == [graded]
    # The reasoning's box no longer passes a wrong final answer in a thinking render.
    assert (accepted, score) == ((False, 0.0) if renderer == THINK else (True, 1.0))
