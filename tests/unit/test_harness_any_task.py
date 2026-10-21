"""A harness on any task: a corpus or eval job pins it in `episode.harness_spec`, an RL window or lease env
in `episode.harness`, and the miner, the validator and the export all play it. Only a harness with
surface variants stays an RL lease's (no other task has a validator seed to draw one from), and the
legacy Docker replay plays the compatibility harness only."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from reliquary_harness.spec import BASH_SURFACE_V1, BASH_VF_B2E4E81, MINI_SWE_V2

from reliquary.corpus.job import parse_job
from reliquary.environment import agentic_swe
from tests.unit.test_corpus_job_episode import _episode, _manifest
from tests.unit.test_corpus_job_signed_sandbox import signed_episode
from tests.unit.test_harness_goldens import TOKENIZERS


def _pin(spec):
    return {"id": spec.id, "version": spec.version, "digest": spec.digest}


MINI_MANIFEST = _manifest(job_id="swe-agentic-mini-v1", prompt_count=3,
                          episode=signed_episode(harness_spec=_pin(MINI_SWE_V2)))
MINI_JOB = parse_job(MINI_MANIFEST)


def test_a_corpus_or_eval_job_plays_any_harness_without_variants():
    assert agentic_swe.harness_pin_refusal(_pin(MINI_SWE_V2), variants=False) is None
    assert agentic_swe.job_harness(MINI_JOB) is MINI_SWE_V2
    assert agentic_swe.job_harness(parse_job(_manifest(episode=signed_episode()))) is BASH_VF_B2E4E81
    surface = agentic_swe.harness_pin_refusal(_pin(BASH_SURFACE_V1), variants=False)
    assert "an RL lease's, not a corpus job's" in surface


def test_a_replay_job_plays_the_compatibility_harness_only():
    replay = parse_job(_manifest(episode=_episode(harness_spec=_pin(MINI_SWE_V2))))
    refusal = agentic_swe.episode_support_refusal(replay.episode, need_verifiers=False)
    assert refusal == "a replay (non signed_sandbox) job plays bash-vf-b2e4e81@1 only, not mini-swe@2"


def test_jobs_create_pins_the_harness_its_signed_episode_plays():
    from reliquary.cli.main import with_job_harness

    episode = signed_episode()
    pinned = with_job_harness(episode, "mini-swe@2")
    assert pinned["harness_spec"] == _pin(MINI_SWE_V2) and "harness_spec" not in episode
    assert parse_job(_manifest(episode=pinned)).episode.harness_spec == _pin(MINI_SWE_V2)
    assert with_job_harness(episode, None) is episode
    for bad, value, why in [(episode, "mini-swe", "id@version"), (episode, "bash-surface@1", "RL lease"),
                            (episode, "openhands@1", "unknown harness"), (_episode(), "mini-swe@2", "signed_sandbox")]:
        with pytest.raises(ValueError, match=why):
            with_job_harness(bad, value)


def _policy(spec):
    return SimpleNamespace(environment="swe_v1", tools=("bash", "edit"),
                           harness_pin=lambda: None if spec is None else _pin(spec))


def test_an_rl_window_plays_any_harness_without_variants():
    from reliquary.miner.episode_mining import _episode_policies
    from reliquary.validator.service import ValidationService

    for spec, served in [(None, True), (MINI_SWE_V2, True), (BASH_SURFACE_V1, False)]:
        contract = SimpleNamespace(episode_policy=lambda name, spec=spec: _policy(spec))
        if served:
            assert _episode_policies(contract, ["swe_v1"], lambda policy: None)["swe_v1"].harness_pin() == \
                (None if spec is None else _pin(spec))
            continue
        with pytest.raises(ValueError, match="an RL lease's"):
            _episode_policies(contract, ["swe_v1"], lambda policy: None)
        with pytest.raises(ValueError, match="an RL lease's"):
            ValidationService._default_episode_renderer(SimpleNamespace(tokenizer=None), _policy(spec))


needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")


@needs_tokenizer
def test_the_window_validator_renders_in_the_orders_harness():
    from reliquary.validator.service import ValidationService

    tokenizer = SimpleNamespace(name_or_path=next(iter(TOKENIZERS.values())))
    renderer = ValidationService._default_episode_renderer(SimpleNamespace(tokenizer=tokenizer),
                                                           _policy(MINI_SWE_V2))
    assert renderer.spec is MINI_SWE_V2 and [t["function"]["name"] for t in renderer.tools] == ["bash"]


# -- a mini-swe eval/corpus job end to end ----------------------------------------------------------------

e2e = pytest.importorskip("tests.unit.test_signed_episode_e2e")
from reliquary_harness import mini  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402

from reliquary.corpus.delivery import episode_rows  # noqa: E402
from reliquary.infrastructure import corpus_job_store as job_store  # noqa: E402
from reliquary.miner.harness_episode import HarnessEpisodeRunner  # noqa: E402
from reliquary.protocol.corpus_submission import CorpusSubmissionRequest  # noqa: E402
from reliquary.protocol.signatures import build_corpus_binding  # noqa: E402
from reliquary.validator.signed_grading import SignedEpisodeGrader  # noqa: E402
from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.test_harness_variants_e2e import _trajectory  # noqa: E402

gateway, r2 = e2e.gateway, e2e.r2      # the signed e2e's fixtures  # noqa: F811
_r2_client, fake_r2 = e2e._r2_client, e2e.fake_r2  # noqa: F811


SCRIPT = [
    parity._turn("Write it.", parity._call("bash", command="write /work/answer.txt 42"), content="THOUGHT: write the answer."),
    parity._turn("Hmm.", content="I think it is written."),
    parity._turn("Done.", parity._call("bash", command=mini.SUBMIT + " && cat patch.txt"), content="THOUGHT: submit."),
]


@needs_tokenizer
async def test_a_mini_swe_job_is_mined_admitted_graded_and_exported_in_mini(gateway, r2):  # noqa: F811
    """The job pins mini-swe@2. The miner's loop plays it against the real gateway (through the
    validator's session route); the validator's corpus intake renders it in mini and accepts it once;
    the grader grades it; the export rebuilds mini's conversation from the proven tokens."""
    await job_store.write_job(MINI_MANIFEST, None, **r2.kwargs)
    directory = next(iter(TOKENIZERS.values()))
    renderer = agentic_swe.job_turn_renderer(MINI_JOB, directory)
    assert renderer.spec is MINI_SWE_V2
    tokenizer = renderer._tokenizer
    chunk = 4096
    await e2e.register(gateway)
    world = await e2e.started(gateway, r2, renderer=renderer, tokenizer=tokenizer, chunk=chunk, job=MINI_JOB)
    engine = parity.FakeEngine(tokenizer, SCRIPT, renderer.stop_ids, 1 << 20)
    prompt_text = e2e.SOURCE.prompt(0)
    with serve_app(engine.app) as engine_url:
        async with e2e.http(world) as validator:
            grant = await e2e.grant_for(validator, world, "6" * 32, job_id=MINI_JOB.job_id)
            async with HarnessEpisodeRunner(gateway_url=grant["gateway_url"], model=parity.MODEL,
                                            renderer_dir=directory, generate_url=engine_url,
                                            sampling=SimpleNamespace(temperature=1.0, top_p=1.0),
                                            max_tokens=MINI_JOB.episode.max_tokens_per_turn,
                                            max_turns=MINI_JOB.episode.max_turns,
                                            spec=agentic_swe.job_harness(MINI_JOB)) as runner:
                result = await runner.run(token=grant["token"], prompt=prompt_text)
            assert result.error is None and result.trace.stop_condition == "agent_completed"
            assert result.final["body"]["status"] == "graded" and result.final["body"]["reward"] == 1.0
            prompt, tokens, spans = _trajectory(engine)
            assert prompt == renderer.initial_ids(prompt_text)          # the validator's own render
            request = CorpusSubmissionRequest(
                job_id=MINI_JOB.job_id, miner_hotkey=e2e.HOTKEY, cursor=0, prompt_index=0,
                checkpoint_sha256=MINI_JOB.checkpoint_sha256,
                rendered_prompt=tokenizer.decode(prompt, skip_special_tokens=False,
                                                 clean_up_tokenization_spaces=False),
                trajectory={"tokens": tokens, "stop": result.trace.stop_condition, "transcript": result.transcript,
                            "final_diff": (result.state or b"").decode("utf-8"),
                            "turns": [{"start": a, "end": b,
                                       "proofs": [e2e.PROOF] * e2e.span_chunk_count(b - a, chunk)}
                                      for a, b in spans]},
                signature="unsigned")
            request = request.model_copy(update={"signature": e2e.sign(build_corpus_binding(request))})
            answer = (await validator.post("/corpus/submit", json=request.model_dump())).json()
            assert answer["reason"] == "accepted", answer
            again = (await validator.post("/corpus/submit", json=request.model_dump())).json()
            assert again["reason"] != "accepted"
    (submission_id,) = world.accepted
    await world.records.write_verdict(MINI_JOB.job_id, submission_id, {"passed": True})
    grade = await SignedEpisodeGrader(job=MINI_JOB, records=world.records, source=e2e.SOURCE).grade_one(
        submission_id)
    assert grade["status"] == "ok"
    counts = {}
    rows = [row async for row in episode_rows(job=MINI_JOB, records=world.records, renderer=renderer,
                                              source=e2e.SOURCE, counts=counts, quarantined=())]
    assert counts["rows"] == 1, counts
    messages = json.loads(rows[0]["messages"])
    assert messages[0] == {"role": "system", "content": mini.SYSTEM}
    assert messages[1]["role"] == "user" and messages[1]["content"].startswith("<pr_description>\nConsider the following")
    # mini's results are tool messages, its format error a user message.
    assert [m["role"] for m in messages[2:]] == ["assistant", "tool", "assistant", "user", "assistant"]
    assert messages[3]["content"].startswith("<returncode>0</returncode>")
    assert messages[5]["content"].startswith("Tool call error:")
    assert [c["function"]["name"] for c in messages[2]["tool_calls"]] == ["bash"]
    assert rows[0]["tokens"] == prompt + tokens
