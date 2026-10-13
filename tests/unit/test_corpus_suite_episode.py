"""Corpus jobs whose unit is a suite episode (design 2026-10-13): the manifest,
the intake's replay, the local grader, the export and its chat-template
transform.

Most tests run on the builtin `reliquary_stateful_tools_v1` environment,
declared here under the ChatML renderer (the registry declares JSONL for it,
and a corpus job takes ChatML only), with a byte-level tokenizer whose ChatML
markers are single tokens. The telecom tests need the `reliquary-telecom-solo`
wheel (v2, 0.2.0a1) and a Qwen3.5 tokenizer, and are skipped without them.
"""

from __future__ import annotations

import asyncio
import copy
import dataclasses
import json
import os
from pathlib import Path

import pytest

from reliquary.corpus import suite_episode
from reliquary.corpus.job import JobError, parse_job
from reliquary.corpus.suite_episode import (
    SuiteSource,
    TokenizerCodec,
    episode_seed,
    export_tokens,
    suite_spec_refusal,
)
from reliquary.environment.agentic.chatml import CHATML_RENDERER_ID, ChatMLEpisodeRenderer
from reliquary.environment.agentic.runner import EpisodeRunner
from reliquary.environment.agentic.types import AssistantAction, GeneratedAction
from reliquary.environment.registry import ENVIRONMENT_SPECS
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest
from reliquary.protocol.toploc import MIN_CHUNK_TOKENS, span_chunk_count
from tests.unit.test_corpus_service import _r2_client, fake_r2  # noqa: F401  (fixtures)

ENV = "reliquary_stateful_tools_v1"
ENV_SHA = ENVIRONMENT_SPECS[ENV].environment_manifest_sha256
JOB_ID = "suite-stateful-v1"
CHUNK = 32
PROOF = "A" * 200
QWEN35_TOKENIZER = Path(os.environ.get(
    "RELIQUARY_TEST_QWEN35_TOKENIZER", "/home/ubuntu/bench-rl-throughput/dryrun/tiny-qwen35"))


# -- a tokenizer whose ChatML markers are single tokens ---------------------------

SPECIALS = {"<|im_start|>": 256, "<|im_end|>": 257, "<|endoftext|>": 258}
IM_END, EOT = 257, 258


class ByteTokenizer:
    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            for marker, token in SPECIALS.items():
                if text.startswith(marker, i):
                    ids.append(token)
                    i += len(marker)
                    break
            else:
                ids.extend(text[i].encode("utf-8"))
                i += 1
        return ids

    def decode(self, ids, skip_special_tokens=False, **_):
        names = {v: k for k, v in SPECIALS.items()}
        out, pending = [], bytearray()
        for token in ids:
            if token in names:
                out.append(pending.decode("utf-8", errors="replace"))
                pending = bytearray()
                out.append(names[token])
            else:
                pending.append(token)
        out.append(pending.decode("utf-8", errors="replace"))
        return "".join(out)


CODEC = TokenizerCodec(ByteTokenizer())


@pytest.fixture
def chatml_registry(monkeypatch):
    """The registry with the stateful-tools environment declared in ChatML."""
    specs = dict(ENVIRONMENT_SPECS)
    specs[ENV] = dataclasses.replace(ENVIRONMENT_SPECS[ENV], renderer_id=CHATML_RENDERER_ID)
    monkeypatch.setattr(suite_episode, "_registry", lambda given: given or specs)
    return specs


def _episode(**overrides):
    episode = {"env": {"package": "reliquary-suite", "name": ENV, "version": ENV_SHA,
                       "split": "train"},
               "harness": "suite", "renderer": CHATML_RENDERER_ID, "max_turns": 8,
               "max_tokens_per_turn": 4096, "max_total_tokens": 60000}
    episode.update(overrides)
    return episode


def _manifest(**overrides):
    raw = {"schema": "reliquary/corpus-job/v1", "job_id": JOB_ID,
           "checkpoint_repo": "Org/Model", "checkpoint_revision": "rev1",
           "checkpoint_sha256": "c" * 64, "prompt_source": ENV, "prompt_count": 50,
           "renderer_id": CHATML_RENDERER_ID, "eos_token_id": IM_END,
           "sampling": {"temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_new_tokens": 2,
                        "max_new_tokens": 4096, "n": 1},
           "slots_per_prompt": 2, "filter": None, "prompt_order": "free",
           "deadline_round": None, "episode": _episode()}
    raw.update(overrides)
    return raw


# -- the manifest -----------------------------------------------------------------


def test_a_suite_job_parses_and_round_trips():
    job = parse_job(_manifest())
    assert job.episode.is_suite and job.episode.env.name == ENV
    assert job.episode.verifiers is None and job.episode.replay_fraction_failed is None
    assert job.to_contract()["episode"] == _episode()
    assert parse_job(job.to_contract()) == job


def _env(**overrides):
    env = dict(_episode()["env"])
    env.update(overrides)
    return env


@pytest.mark.parametrize("episode", [
    _episode(verifiers="b" * 40),                                   # a SWE field
    _episode(replay_fraction_failed=0.1),
    _episode(env=_env(num_images=1)),
    _episode(env=_env(package="reliquary-swe")),
    _episode(env=_env(version="a" * 40)),                           # a commit, not a manifest
    _episode(env=_env(split="eval")),
    _episode(env=_env(name="Not A Name")),
    _episode(max_turns=65), _episode(max_total_tokens=60001),
    _episode(renderer=" x"),
])
def test_bad_suite_episodes_are_refused(episode):
    with pytest.raises(JobError):
        parse_job(_manifest(episode=episode))


def test_a_bash_harness_cannot_run_the_suite_package():
    from tests.unit.test_corpus_job_episode import _manifest as swe_manifest, _episode as swe

    episode = swe()
    episode["env"] = {**episode["env"], "package": "reliquary-suite"}
    with pytest.raises(JobError, match="harness runs"):
        parse_job(swe_manifest(episode=episode))


@pytest.mark.parametrize("overrides,match", [
    ({"prompt_source": "reliquary_retrieval_tools_v1"}, "prompt_source"),
    ({"slots_per_prompt": 1}, "slots_per_prompt"),
    ({"prompt_order": "miner_walk"}, "prompt_order"),
    ({"filter": {"grader_id": "g", "threshold": 0.5}}, "filter"),
    ({"renderer_id": "reliquary-jsonl-tools-v1"}, "renderer_id"),
])
def test_the_episode_job_rules_hold_for_suite_jobs(overrides, match):
    with pytest.raises(JobError, match=match):
        parse_job(_manifest(**overrides))


def test_the_registry_must_have_the_environment_at_its_manifest_and_renderer(chatml_registry):
    episode = parse_job(_manifest()).episode
    assert suite_spec_refusal(episode) is None
    other = dataclasses.replace(episode, env=dataclasses.replace(episode.env, version="0" * 64))
    assert "manifest" in suite_spec_refusal(other)
    missing = dataclasses.replace(episode, env=dataclasses.replace(episode.env, name="nope_v1"))
    assert "not in this binary's registry" in suite_spec_refusal(missing)
    single = dataclasses.replace(episode, env=dataclasses.replace(
        episode.env, name="openmathinstruct"))
    assert "not an episode environment" in suite_spec_refusal(single)


def test_a_jsonl_environment_is_not_a_corpus_suite_source():
    # The real registry: stateful tools render JSONL there.
    episode = parse_job(_manifest(episode=_episode(renderer="reliquary-jsonl-tools-v1"),
                                  renderer_id="reliquary-jsonl-tools-v1")).episode
    assert "takes only" in suite_spec_refusal(episode)


def test_the_turn_limit_is_the_lower_of_the_jobs_and_the_environments(chatml_registry):
    environment = ENVIRONMENT_SPECS[ENV].create()
    assert environment.max_turns == 8
    first = _reference_actions(environment, 0)[0]
    rendered, tokens, spans, stop, _ = miner_episode(environment, 0, [first] * 8, CODEC)
    assert stop == "max_turns" and len(spans) == 8
    job = parse_job(_manifest(episode=_episode(max_turns=64)))
    facts = _intake(job).check(_request(rendered, tokens, spans, stop))
    assert facts.success is False and facts.task_id is not None, facts


def test_telecom_corpus_ranges():
    from reliquary.eval.sets import refuse_held_out_overlap

    refuse_held_out_overlap("reliquary_telecom_solo_v2", 0, 1220)
    with pytest.raises(ValueError, match="corpus ranges"):
        refuse_held_out_overlap("reliquary_telecom_solo_v2", 0, 1221)
    with pytest.raises(ValueError, match="serves no corpus job"):
        refuse_held_out_overlap("reliquary_telecom_solo_v1", 0, 10)


# -- an honest episode, as a miner runs it ----------------------------------------


def _policy(actions, codec, *, terminators=None):
    texts = []
    for turn, action in enumerate(actions):
        end = (terminators or {}).get(turn, "<|im_end|>")
        texts.append(ChatMLEpisodeRenderer.action_text(
            action, reasoning=f"Step {turn}: I will act on what I know.") + end)
    pieces = iter(texts)

    class Policy:
        def generate(self, **_):
            text = next(pieces)
            return GeneratedAction(text=text, tokens=tuple(codec.encode(text)))

    return Policy()


def _reference_actions(environment, index):
    return list(environment.get_task(index).private["reference_actions"])


def miner_episode(environment, index, actions, codec, *, job_id=JOB_ID, max_turns=8, **kw):
    """What a miner submits for one prompt: the trajectory after the initial
    prompt, cut at the end of the last span."""
    task = environment.get_task(index)
    renderer = ChatMLEpisodeRenderer(codec.encode)
    trace = EpisodeRunner(renderer=renderer, max_turns=max_turns).run(
        environment, task, seed=episode_seed(job_id, index), policy=_policy(actions, codec, **kw))
    offset = len(renderer.encode_initial(task))
    last = trace.assistant_spans[-1][1]
    tokens = list(trace.tokens[offset:last])
    spans = [(s - offset, e - offset) for s, e in trace.assistant_spans]
    stop = "max_turns" if trace.termination_reason == "turn_limit" else "agent_completed"
    return renderer.initial_text(task), tokens, spans, stop, trace


def _request(rendered, tokens, spans, stop, *, index=0, job_id=JOB_ID, final_diff=""):
    return CorpusSubmissionRequest(
        job_id=job_id, miner_hotkey="5Hot", cursor=0, prompt_index=index,
        checkpoint_sha256="c" * 64, rendered_prompt=rendered,
        trajectory={"tokens": tokens, "turns": [
            {"start": s, "end": e, "proofs": [PROOF] * span_chunk_count(e - s, CHUNK,
                                                                         MIN_CHUNK_TOKENS)}
            for s, e in spans], "final_diff": final_diff, "stop": stop},
        signature="ok")


def _intake(job=None):
    from reliquary.validator.suite_intake import InlineSuiteReplayer, SuiteEpisodeIntake

    job = job or parse_job(_manifest())
    source = suite_episode.suite_source(job.episode)
    return SuiteEpisodeIntake(job=job, source=source, tokenizer=ByteTokenizer(), vocab_size=None,
                              chunk_tokens=CHUNK,
                              replayer=InlineSuiteReplayer(encode=CODEC.encode,
                                                           decode=CODEC.decode))


@pytest.fixture
def honest(chatml_registry):
    environment = ENVIRONMENT_SPECS[ENV].create()
    return miner_episode(environment, 0, _reference_actions(environment, 0), CODEC)


def test_an_honest_episode_is_accepted_with_the_replay_verdict(honest):
    from reliquary.validator.agentic_intake import IntakeFacts

    rendered, tokens, spans, stop, trace = honest
    assert stop == "agent_completed" and len(spans) >= 2
    facts = _intake().check(_request(rendered, tokens, spans, stop))
    assert isinstance(facts, IntakeFacts), facts
    assert facts.task_id == trace.task_id
    assert facts.success is True and facts.reward == 1.0
    assert facts.token_count == sum(e - s for s, e in spans)


def test_a_failing_episode_is_accepted_and_says_so(chatml_registry):
    environment = ENVIRONMENT_SPECS[ENV].create()
    rendered, tokens, spans, stop, _ = miner_episode(
        environment, 0, [AssistantAction.final("I could not do it.")], CODEC)
    facts = _intake().check(_request(rendered, tokens, spans, stop))
    assert facts.success is False and facts.reward == 0.0


def _first_observation_token(spans):
    return spans[0][1] + 5


def test_a_forged_observation_is_refused(honest):
    rendered, tokens, spans, stop, _ = honest
    forged = list(tokens)
    at = _first_observation_token(spans)
    forged[at] = ord("#") if forged[at] != ord("#") else ord("$")
    refusal = _intake().check(_request(rendered, forged, spans, stop))
    assert refusal.reason == "bad_observation" and refusal.detail["turn"] == 1


def test_an_action_the_observation_does_not_answer_is_refused(chatml_registry):
    """The miner changes an argument after the fact: the environment answers
    the changed call differently, so the segment that follows is not its."""
    environment = ENVIRONMENT_SPECS[ENV].create()
    actions = _reference_actions(environment, 0)
    rendered, tokens, spans, stop, _ = miner_episode(environment, 0, actions, CODEC)
    first = actions[0]
    changed = dataclasses.replace(first, arguments={k: "C-9999" for k in first.arguments})
    _, other, other_spans, _, _ = miner_episode(environment, 0, [changed, *actions[1:]], CODEC)
    # The changed first turn, then the honest observation and turns after it.
    s0, e0 = other_spans[0]
    swapped = other[s0:e0] + tokens[spans[0][1]:]
    shift = (e0 - s0) - (spans[0][1] - spans[0][0])
    swapped_spans = [(0, e0)] + [(s + shift, e + shift) for s, e in spans[1:]]
    refusal = _intake().check(_request(rendered, swapped, swapped_spans, stop))
    assert refusal.reason in ("bad_observation", "bad_turns"), refusal


def test_a_trajectory_cut_short_is_refused(honest):
    rendered, tokens, spans, stop, _ = honest
    end = spans[-2][1]
    refusal = _intake().check(_request(rendered, tokens[:end], spans[:-1], stop))
    assert refusal.reason == "bad_turns"


def test_a_stop_the_replay_denies_is_refused(honest):
    rendered, tokens, spans, _, _ = honest
    refusal = _intake().check(_request(rendered, tokens, spans, "max_turns"))
    assert refusal.reason == "bad_stop"
    refusal = _intake().check(_request(rendered, tokens, spans, "context_length"))
    assert refusal.reason == "bad_stop"


def test_an_unfaithful_prompt_or_a_diff_is_refused(honest):
    rendered, tokens, spans, stop, _ = honest
    assert _intake().check(_request("x" + rendered, tokens, spans, stop)).reason \
        == "prompt_not_faithful"
    assert _intake().check(_request(rendered, tokens, spans, stop, final_diff="d")).reason \
        == "malformed_submission"


def test_endoftext_mid_episode_is_refused_and_ends_the_last_turn(chatml_registry):
    environment = ENVIRONMENT_SPECS[ENV].create()
    actions = _reference_actions(environment, 0)
    mid = miner_episode(environment, 0, actions, CODEC, terminators={0: "<|endoftext|>"})
    assert _intake().check(_request(*mid[:4])).reason == "bad_termination"
    last = miner_episode(environment, 0, actions, CODEC,
                         terminators={len(actions) - 1: "<|endoftext|>"})
    assert _intake().check(_request(*last[:4])).success is True


def test_the_turn_limit_is_the_jobs(chatml_registry):
    """A job's max_turns below the environment's: the replay stops there too."""
    environment = ENVIRONMENT_SPECS[ENV].create()
    actions = _reference_actions(environment, 0)
    job = parse_job(_manifest(episode=_episode(max_turns=1)))
    rendered, tokens, spans, stop, _ = miner_episode(environment, 0, actions, CODEC, max_turns=1)
    assert stop == "max_turns" and len(spans) == 1
    facts = _intake(job).check(_request(rendered, tokens, spans, stop))
    assert facts.success is False


# -- the route --------------------------------------------------------------------


def test_an_honest_episode_through_the_route_is_recorded(honest, fake_r2):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from reliquary.infrastructure import corpus_job_store as job_store
    from reliquary.validator.corpus_service import build_corpus_router
    from tests.unit.test_corpus_route_records import _Records
    from tests.unit.test_corpus_service import _CountingStore

    asyncio.run(job_store.write_job(_manifest(), None, **fake_r2))
    records, accepted = _Records(), []
    app = FastAPI()
    app.include_router(build_corpus_router(
        job_id=JOB_ID, store=_CountingStore(fake_r2), tokenizer=ByteTokenizer(), renderer=None,
        verify_signature=lambda r: True, records=records, on_accepted=accepted.append,
        proof_chunk_tokens=CHUNK, episode_intake=_intake()))
    rendered, tokens, spans, stop, _ = honest
    body = TestClient(app).post("/corpus/submit",
                                json=_request(rendered, tokens, spans, stop).model_dump()).json()
    assert body["accepted"] is True, body
    (record,) = records.written.values()
    assert record["schema"] == "reliquary/corpus-submission-record/v2"
    assert record["completions"][0]["prompt_tokens"] == list(_intake().initial_ids(0))


# -- the grader -------------------------------------------------------------------


class _GradeRecords:
    def __init__(self, records, verdicts=None):
        self.records, self.verdicts, self.grades = records, verdicts or {}, {}

    async def read_submission(self, job_id, sid):
        return self.records.get(sid)

    async def read_verdict(self, job_id, sid):
        return self.verdicts.get(sid)

    async def write_grade(self, job_id, sid, document):
        self.grades[sid] = document

    async def list_grade_ids(self, job_id):
        return list(self.grades)

    async def list_submission_ids(self, job_id):
        return list(self.records)


def _record(request, prompt_ids):
    trajectory = request.trajectory.model_dump()
    trajectory["prompt_tokens"] = list(prompt_ids)
    return {"schema": "reliquary/corpus-submission-record/v2", "submission_id": "s1",
            "job_id": JOB_ID, "hotkey": "5Hot", "cursor": 0,
            "prompt_index": request.prompt_index, "rendered_prompt": request.rendered_prompt,
            "received_at": 1.0, "token_count": 1, "completions": [trajectory]}


def _grader(records):
    from reliquary.validator.suite_grading import LocalEpisodeGrader
    from reliquary.validator.suite_intake import InlineSuiteReplayer

    return LocalEpisodeGrader(job=parse_job(_manifest()), records=records,
                              replayer=InlineSuiteReplayer(encode=CODEC.encode,
                                                           decode=CODEC.decode))


def test_the_local_grader_writes_a_certified_grade(honest):
    from reliquary.corpus.delivery import certified

    rendered, tokens, spans, stop, trace = honest
    record = _record(_request(rendered, tokens, spans, stop), _intake().initial_ids(0))
    records = _GradeRecords({"s1": record})
    grader = _grader(records)

    async def run():
        document = await grader.grade_one("s1")
        return document, await grader.ready({"s1", "s2"})

    document, ready = asyncio.run(run())
    assert document == records.grades["s1"]
    assert document["status"] == "ok" and document["graded_success"] is True
    assert document["replay"]["certified"] is True and document["replay_certified"] is True
    assert document["instance_id"] == trace.task_id and certified(document)
    assert ready == {"s1"}
    grader.hold_executor("x")                         # no executor: nothing held
    assert asyncio.run(grader.regrade_executor("x")) == []
    assert asyncio.run(grader.ready({"s1"})) == {"s1"}


def test_the_local_grader_respects_a_failed_audit(honest):
    rendered, tokens, spans, stop, _ = honest
    record = _record(_request(rendered, tokens, spans, stop), _intake().initial_ids(0))
    records = _GradeRecords({"s1": record}, {"s1": {"passed": False}})
    document = asyncio.run(_grader(records).grade_one("s1"))
    assert document["status"] == "audit_failed" and document["replay_certified"] is False


# -- the export -------------------------------------------------------------------


def _row(honest_episode, codec=CODEC, environment_name=ENV, job=None):
    from reliquary.corpus.delivery import suite_episode_row

    rendered, tokens, spans, stop, trace = honest_episode
    job = job or parse_job(_manifest())
    source = suite_episode.suite_source(job.episode)
    prompt = source.renderer(codec.encode).encode_initial(source.task(0))
    request = _request(rendered, tokens, spans, stop, job_id=job.job_id)
    grade = {"status": "ok", "graded_success": True, "replay_certified": True,
             "instance_id": trace.task_id}
    return suite_episode_row(job=job, submission_id="s1", record=_record(request, prompt),
                             grade=grade, source=source, encode=codec.encode,
                             decode=codec.decode), prompt


def test_the_export_row_drops_the_duplicate_im_end(honest):
    row, prompt = _row(honest)
    _, tokens, spans, _, _ = honest
    raw = list(prompt) + tokens
    doubled = sum(1 for a, b in zip(raw, raw[1:]) if a == b == IM_END)
    assert doubled == len(spans) - 1                  # one per answered turn, before the transform
    assert not any(a == b == IM_END for a, b in zip(row["tokens"], row["tokens"][1:]))
    assert len(row["tokens"]) == len(raw) - doubled + 1          # + the closing "\n"
    assert row["tokens"][-2:] == [IM_END, ord("\n")]
    assert sum(row["assistant_mask"]) == sum(e - s for s, e in spans)
    assert len(row["assistant_mask"]) == len(row["tokens"])
    assert row["tokens_transform"] == "chatml-single-im-end-v1"
    assert row["graded_success"] is True and row["replay_certified"] is True
    assert row["reward"] == 1.0 and row["prompt_index"] == 0


def test_the_export_messages_and_tools(honest):
    row, _ = _row(honest)
    messages, tools = json.loads(row["messages"]), json.loads(row["tools"])
    assert [m["role"] for m in messages[:2]] == ["system", "user"]
    assistants = [m for m in messages if m["role"] == "assistant"]
    assert len(assistants) == len(honest[2])
    call = assistants[0]["tool_calls"][0]
    assert call["type"] == "function"
    assert call["function"]["name"] in {tool["function"]["name"] for tool in tools}
    assert assistants[0]["reasoning_content"].startswith("Step 0")
    # The last turn (here a `finish` call) ends the conversation: what the
    # environment said after it is not in the trajectory.
    assert messages[-1]["role"] == "assistant"
    assert any(m["role"] == "tool" for m in messages)


def test_export_tokens_keep_a_capped_turns_only_terminator():
    # Turn 0 cut at the cap (no <|im_end|>): the observation's is kept.
    full = [1, 2, 3, IM_END, 10, 5, 6, IM_END, IM_END, 10, 7, IM_END]
    spans = [(1, 3), (5, 8), (10, 12)]
    tokens, mask = export_tokens(full, spans, im_end_id=IM_END, newline_ids=[10])
    assert tokens == [1, 2, 3, IM_END, 10, 5, 6, IM_END, 10, 7, IM_END, 10]
    assert mask == [0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0]


@pytest.mark.skipif(not QWEN35_TOKENIZER.is_dir(), reason="no Qwen3.5 tokenizer on this host")
def test_exported_tokens_are_the_chat_template_rendering(chatml_registry):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(QWEN35_TOKENIZER))
    codec = TokenizerCodec(tokenizer)
    environment = ENVIRONMENT_SPECS[ENV].create()
    episode = miner_episode(environment, 0, _reference_actions(environment, 0), codec)
    row, _ = _row(episode, codec=codec)
    rendered = tokenizer.apply_chat_template(
        json.loads(row["messages"]), tools=json.loads(row["tools"]), tokenize=False)
    assert row["tokens"] == codec.encode(rendered)


# -- telecom v2 (the real wheel) -------------------------------------------------

TELECOM = "reliquary_telecom_solo_v2"


def _telecom_or_skip():
    try:
        environment = ENVIRONMENT_SPECS[TELECOM].create()
    except Exception as exc:  # noqa: BLE001 - absent or another version installed
        pytest.skip(f"reliquary-telecom-solo 0.2.0a1 not installed: {exc}")
    if not QWEN35_TOKENIZER.is_dir():
        pytest.skip("no Qwen3.5 tokenizer on this host")
    return environment


def _telecom_golden(index=83):
    from importlib import resources

    text = resources.files("reliquary_telecom_solo").joinpath("goldens/reference.jsonl").read_text()
    for line in text.splitlines():
        row = json.loads(line)
        if row["split"] == "train" and row["index"] == index:
            return [AssistantAction(kind="tool", tool=a["tool"], arguments=a["arguments"])
                    for a in row["actions"]], row
    raise LookupError(index)


def _telecom_job():
    episode = _episode(env=_env(name=TELECOM,
                                version=ENVIRONMENT_SPECS[TELECOM].environment_manifest_sha256),
                       max_turns=40)
    return parse_job(_manifest(job_id="telecom-v2-sft", prompt_source=TELECOM,
                               prompt_count=1220, episode=episode))


def test_an_honest_telecom_episode_replays_in_the_worker_pool():
    from transformers import AutoTokenizer

    from reliquary.validator.suite_intake import PoolSuiteReplayer, SuiteEpisodeIntake

    environment = _telecom_or_skip()
    tokenizer = AutoTokenizer.from_pretrained(str(QWEN35_TOKENIZER))
    codec = TokenizerCodec(tokenizer)
    actions, golden = _telecom_golden()
    job = _telecom_job()
    rendered, tokens, spans, stop, trace = miner_episode(
        environment, 83, actions, codec, job_id=job.job_id, max_turns=40)
    assert trace.reward.success is True and stop == "agent_completed"
    replayer = PoolSuiteReplayer.for_tokenizer(tokenizer, workers=1)
    try:
        intake = SuiteEpisodeIntake(job=job, source=suite_episode.suite_source(job.episode),
                                    tokenizer=tokenizer, vocab_size=None, chunk_tokens=CHUNK,
                                    replayer=replayer)
        facts = intake.check(_request(rendered, tokens, spans, stop, index=83,
                                      job_id=job.job_id))
        assert facts.success is True and facts.task_id == golden["task_id"], facts
        forged = list(tokens)
        forged[spans[0][1] + 8] = codec.encode("X")[0]
        refusal = intake.check(_request(rendered, forged, spans, stop, index=83,
                                        job_id=job.job_id))
        assert refusal.reason == "bad_observation"
    finally:
        replayer.close()


def test_a_telecom_v2_job_can_be_declared_on_its_sft_share_only():
    from reliquary.cli.main import prepare_corpus_job

    _telecom_or_skip()
    episode = copy.deepcopy(_telecom_job().to_contract()["episode"])

    def declare(count):
        return prepare_corpus_job(
            job_id="telecom-v2-sft", task_id=None, model="Org/Model", model_revision="r" * 40,
            model_architecture="qwen3_5", checkpoint_sha256="c" * 64, from_profile=None,
            prompt_encoding=None, prompt_source=TELECOM, prompt_count=count, prompt_start=0,
            renderer_id=CHATML_RENDERER_ID, eos_token_id=248046, slots_per_prompt=2,
            max_new_tokens=4096, cap=0.05, min_incentive_share=0.5, audit_params={},
            episode=episode)

    manifest, entry = declare(1220)
    assert parse_job(manifest).episode.is_suite
    contract = entry.contract["environments"][TELECOM]
    assert contract["episode"]["renderer_id"] == CHATML_RENDERER_ID
    with pytest.raises(ValueError, match="corpus ranges"):
        declare(1221)


def test_suite_source_reads_tasks(chatml_registry):
    source = suite_episode.suite_source(parse_job(_manifest()).episode)
    assert isinstance(source, SuiteSource)
    assert source.instance_id(0) == source.task(0).id
    assert source.initial_text(0).endswith("<|im_start|>assistant\n<think>\n")
