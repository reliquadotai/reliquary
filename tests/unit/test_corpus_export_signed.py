"""An exported signed episode: messages rebuilt from the proven tokens, observations
from the signed records (never decoded), the system prompt for the job's tools."""

import json

import pytest

pytest.importorskip("reliquary_sandbox.attest")

from reliquary.corpus.delivery import episode_row  # noqa: E402
from reliquary.corpus.job import parse_job  # noqa: E402
from reliquary.corpus.trajectory_parse import TrajectoryRefused  # noqa: E402
from reliquary.environment.agentic_swe import BASH_SENTENCE, BASH_SYSTEM_PROMPT  # noqa: E402
from tests.unit.sandbox_fixtures import claims, signer, transcript  # noqa: E402
from tests.unit.test_corpus_job_episode import _manifest  # noqa: E402
from tests.unit.test_corpus_job_signed_sandbox import sandbox_spec, signed_episode  # noqa: E402
from tests.unit.test_signed_parse import CALL, S, TERM, TEXT  # noqa: E402


class ExportRenderer(type(S)):
    def render_messages(self, messages):
        return list(self._rendered)

    def whitespace_free(self, ids):
        return tuple(ids)


def build(tmp_path, tools=("bash", "edit"), observation="a.py\n"):
    prompt = S.initial_ids("Fix task 0.")
    first = [TEXT] * 8 + [CALL, TERM]
    full = S.next_prompt(prompt, first, [observation])
    tokens = full[len(prompt):] + [TEXT] * 9 + [TERM]
    spans = [(0, len(first)), (len(full) - len(prompt), len(tokens))]
    signed = transcript(signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1"), claims(),
                        calls=[{"turn": 0, "k": 0, "arguments": {"command": "c0"},
                                "output": "a.py\n"}], tools=tools)
    trajectory = {"prompt_tokens": prompt, "tokens": tokens, "stop": "agent_completed",
                  "turns": [{"start": s, "end": e} for s, e in spans], "final_diff": "d\n",
                  "transcript": signed}
    renderer = ExportRenderer()
    renderer._rendered = prompt + tokens
    renderer.initial_ids = lambda user_prompt: prompt
    return trajectory, renderer


def row(trajectory, renderer, job):
    return episode_row(job=job, submission_id="sub-1", record={"completions": [trajectory],
                                                               "prompt_index": 0},
                       grade={"instance_id": "repo__0", "graded_success": True,
                              "replay_certified": True},
                       renderer=renderer, user_prompt="Fix task 0.")


def test_observations_come_from_the_signed_records(tmp_path):
    job = parse_job(_manifest(episode=signed_episode()))
    trajectory, renderer = build(tmp_path)
    messages = json.loads(row(trajectory, renderer, job)["messages"])
    assert messages[0] == {"role": "system", "content": BASH_SYSTEM_PROMPT}
    assert [m["content"] for m in messages if m["role"] == "tool"] == ["a.py\n"]


def test_a_bash_only_job_exports_its_own_system_prompt(tmp_path):
    job = parse_job(_manifest(episode=signed_episode(sandbox=sandbox_spec(tools=["bash"]))))
    trajectory, renderer = build(tmp_path, tools=("bash",))
    messages = json.loads(row(trajectory, renderer, job)["messages"])
    assert messages[0]["content"] == BASH_SENTENCE


def test_tokens_that_are_not_the_records_rendering_do_not_export(tmp_path):
    job = parse_job(_manifest(episode=signed_episode()))
    trajectory, renderer = build(tmp_path, observation="forged\n")
    with pytest.raises(TrajectoryRefused):
        row(trajectory, renderer, job)


def test_a_signed_jobs_record_without_its_transcript_is_refused(tmp_path):
    """C1: the job decides the parser; a signed job's record with no transcript is a
    typed refusal, not a replay parse."""
    from reliquary.corpus.delivery import EpisodeKindMismatch

    job = parse_job(_manifest(episode=signed_episode()))
    trajectory, renderer = build(tmp_path)
    del trajectory["transcript"]
    with pytest.raises(EpisodeKindMismatch):
        row(trajectory, renderer, job)


def test_a_signed_row_without_the_sandbox_package_is_a_typed_refusal(tmp_path, monkeypatch):
    import sys

    from reliquary.corpus.delivery import EpisodeSandboxUnavailable

    job = parse_job(_manifest(episode=signed_episode()))
    trajectory, renderer = build(tmp_path)
    monkeypatch.setitem(sys.modules, "reliquary.corpus.signed_parse", None)
    with pytest.raises(EpisodeSandboxUnavailable):
        row(trajectory, renderer, job)
