"""A malformed model turn on an async RL lease: the miner's loop ends the episode on it (`format_error`), the
validator admits the episode with reward 0 and the lease keeps its 16 episodes; a corpus or window episode still
refuses it. The model's turn is malformed when it writes turn markup or a stop before its end (its own made-up
observation, its own next turn), or ends a turn with calls on the end-of-text token."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from reliquary.corpus.trajectory_parse import STOP_FORMAT_ERROR, turn_format_violation
from tests.unit.test_trajectory_parse import CALL, EOT, TERM, TEXT, TR, FakeRenderer


def test_a_turn_is_malformed_by_its_markup_or_its_end_of_text_with_calls():
    r = FakeRenderer()
    assert turn_format_violation(r, [TEXT, CALL, TERM]) is None                 # a closed turn
    assert turn_format_violation(r, [TEXT, EOT]) is None                         # end of text, no call: the end
    assert turn_format_violation(r, [TEXT, TEXT]) is None                        # cut at a cap: not the model's form
    assert turn_format_violation(r, [TEXT, TR, TEXT, TERM]) == "turn markup or stop inside the turn"
    assert turn_format_violation(r, [TEXT, TERM, TEXT, TERM]) == "turn markup or stop inside the turn"
    assert turn_format_violation(r, [TEXT, CALL, TR]) == "turn markup ends the turn"
    assert turn_format_violation(r, [TEXT, CALL, EOT]) == \
        "a turn with calls ended on another stop than the turn terminator"


def test_the_miner_maps_a_format_error_for_a_lease_only():
    from reliquary.miner.episode_commit import episode_stop

    tokens, spans = [0] * 10 + [TEXT, TR, TEXT], [(10, 13)]
    kwargs = dict(tokens=tokens, spans=spans, max_turns=4, max_tokens_per_turn=64, max_episode_tokens=256,
                  stop_ids={TERM, EOT})
    assert episode_stop(STOP_FORMAT_ERROR, lease=True, **kwargs) == STOP_FORMAT_ERROR
    assert episode_stop(STOP_FORMAT_ERROR, lease=False, **kwargs) is None


def _checker(monkeypatch, exhausted=()):
    from reliquary.validator import episode_admission
    from reliquary.validator.rl_episode import LeaseEpisodeChecker

    def verified(policy, transcript, expected, **kwargs):
        index = int(expected.engagement.rsplit(":", 1)[1])
        status = "budget_exhausted" if index in exhausted else "graded"
        return SimpleNamespace(claims=SimpleNamespace(session_id=f"s{index}"),
                               final=SimpleNamespace(reward=1.0, status=status), calls=(), open=None)

    monkeypatch.setattr(episode_admission, "verified_transcript", verified)
    monkeypatch.setattr(episode_admission, "episode_shape_refusal", lambda *a, **k: None)
    policy = SimpleNamespace(environment="swe_v1", sandbox_env="e", split="train", tools=("bash",), max_turns=8,
                             max_tokens_per_turn=64, max_episode_tokens=4096)
    renderer = SimpleNamespace(initial_ids=lambda prompt: [1, 2, 3])
    checker = LeaseEpisodeChecker(policy=policy, renderer=renderer, source=SimpleNamespace(prompt=lambda i: "p"),
                                  directory=lambda now: object(), token_verifier=None)
    lease = SimpleNamespace(nonce="ab" * 32, lease_id="L1", prompt_idx=0, hotkey="h", checkpoint=1)
    return checker, lease


def _rollouts(stops):
    return [SimpleNamespace(force_span=None, tokens=[TEXT, TERM], spans=((0, 2),),
                            episode={"transcript": {}, "stop": stops.get(i, "agent_completed")}) for i in range(16)]


def test_the_lease_checker_scores_a_format_error_episode_0_and_keeps_the_lease(monkeypatch):
    checker, lease = _checker(monkeypatch)
    graded = checker.grade(lease, _rollouts({3: STOP_FORMAT_ERROR, 9: STOP_FORMAT_ERROR}), received=0.0)
    assert getattr(graded, "reject", None) is None
    assert [graded.rewards[i] for i in range(16)] == [0.0 if i in (3, 9) else 1.0 for i in range(16)]


def test_a_format_error_and_a_budget_close_in_one_lease_both_score_0(monkeypatch):
    checker, lease = _checker(monkeypatch, exhausted=(5,))
    graded = checker.grade(lease, _rollouts({3: STOP_FORMAT_ERROR, 5: "episode_closed"}), received=0.0)
    assert getattr(graded, "reject", None) is None
    assert [graded.rewards[i] for i in range(16)] == [0.0 if i in (3, 5) else 1.0 for i in range(16)]
    assert [i for i, cut in enumerate(graded.truncated) if cut] == [5]       # only the budget close counts as cut


def test_a_budget_close_must_be_what_the_episode_says_it_stopped_on(monkeypatch):
    checker, lease = _checker(monkeypatch, exhausted=(5,))
    graded = checker.grade(lease, _rollouts({5: STOP_FORMAT_ERROR}), received=0.0)
    assert graded.reject == "episode_transcript"


# -- end to end: the miner's loop against the real gateway, the validator's shared checks -----------------------

pytest.importorskip("renderers")
pytest.importorskip("reliquary_sandbox_service.episodes.testing")

from reliquary_harness.spec import BASH_VF_B2E4E81  # noqa: E402
from reliquary_sandbox_service.episodes import testing  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402

from reliquary.corpus.signed_parse import signed_records  # noqa: E402
from reliquary.miner.harness_episode import HarnessEpisodeRunner  # noqa: E402
from reliquary.validator.episode_admission import episode_shape_refusal  # noqa: E402
from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.test_harness_goldens import TOKENIZERS  # noqa: E402
from tests.unit.test_harness_variants_e2e import _renderer, _trajectory  # noqa: E402

needs_tokenizer = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
WRITE = parity._turn("Write it.", parity._call("bash", command="write /work/answer.txt 42"))
# The model writes its own observation and next turn inside its turn (the smoke's 2B did).
MADE_UP = ("Check it.\n</think>\n\n" + parity._call("bash", command="cat /work/answer.txt")
           + "\n<|im_start|>user\n<tool_response>\n42\n</tool_response><|im_end|>")
# The model ends a turn with a call on the end-of-text token.
END_OF_TEXT = "Check it.\n</think>\n\n" + parity._call("bash", command="cat /work/answer.txt") + "<|endoftext|>"
DONE = parity._turn("Done.", content="Written.")
POLICY = SimpleNamespace(max_turns=6, max_tokens_per_turn=512, max_episode_tokens=1 << 16)


async def _play(tmp_path, script, *, format_stop):
    directory = next(iter(TOKENIZERS.values()))
    tokenizer = parity._tokenizer(directory)
    renderer = _renderer(BASH_VF_B2E4E81, 0)
    engine = parity.FakeEngine(tokenizer, script, renderer.stop_ids, 1 << 20)
    with testing.fake_gateway(tmp_path) as (gateway, _), serve_app(engine.app) as engine_url:
        token = gateway.issue(env="fake-env", split="train", index=1, image=testing.FAKE_IMAGE,
                              budgets=parity._budgets())
        async with HarnessEpisodeRunner(gateway_url=gateway.url, model=parity.MODEL, renderer_dir=directory,
                                        generate_url=engine_url, sampling=SimpleNamespace(temperature=1.0, top_p=1.0),
                                        max_tokens=POLICY.max_tokens_per_turn, max_turns=POLICY.max_turns,
                                        format_stop=format_stop) as runner:
            result = await runner.run(token=token, prompt=parity.PROMPT)
    return result, engine, renderer


def _shape(renderer, result, engine, *, lease, stop=None):
    prompt, tokens, spans = _trajectory(engine)
    found = signed_records(result.transcript)
    offset = len(prompt)
    return episode_shape_refusal(POLICY, renderer, prompt_ids=list(prompt), tokens=[*prompt, *tokens],
                                 spans=[(offset + a, offset + b) for a, b in spans],
                                 stop=stop or result.trace.stop_condition, calls=tuple(found.calls),
                                 opened=SimpleNamespace(tools=found.tools), final=found.final, lease=lease)


@needs_tokenizer
@pytest.mark.parametrize("malformed", [MADE_UP, END_OF_TEXT], ids=["made-up-observation", "end-of-text"])
def test_a_lease_episode_ends_at_the_malformed_turn_and_is_admitted(tmp_path, malformed):
    result, engine, renderer = asyncio.run(_play(tmp_path, [WRITE, malformed, DONE], format_stop=True))
    if malformed is END_OF_TEXT:
        assert engine.completions[1][-1] in renderer.stop_ids and engine.completions[1][-1] != renderer.terminator_id
    assert result.error is None and result.trace.stop_condition == STOP_FORMAT_ERROR
    assert len(engine.requests) == 2                                       # no turn after the malformed one
    calls = [r["body"] for r in result.transcript["records"] if r["body"]["kind"] == "call"]
    assert [c["arguments"]["command"] for c in calls] == ["write /work/answer.txt 42"]   # its call never ran
    assert result.final["body"]["status"] == "graded"
    assert _shape(renderer, result, engine, lease=True) is None             # the validator's and the miner's check
    refused = _shape(renderer, result, engine, lease=False)                  # a window or corpus episode: refused
    assert refused is not None and refused.stage == "episode_termination"


@needs_tokenizer
def test_without_the_format_stop_the_episode_goes_on_and_is_refused(tmp_path):
    """The corpus and window behaviour, kept: the loop goes on past the malformed turn, and the trajectory is
    refused (the whole lease was given up for it before this change)."""
    result, engine, renderer = asyncio.run(_play(tmp_path, [WRITE, MADE_UP, DONE], format_stop=False))
    assert result.error is None and result.trace.stop_condition == "agent_completed" and len(engine.requests) == 3
    refused = _shape(renderer, result, engine, lease=True)
    assert refused is not None and refused.detail["check"] == "bad_turns"


@needs_tokenizer
def test_a_well_formed_episode_cannot_claim_a_format_error(tmp_path):
    result, engine, renderer = asyncio.run(_play(tmp_path, [WRITE, DONE], format_stop=True))
    assert result.trace.stop_condition == "agent_completed"
    refused = _shape(renderer, result, engine, lease=True, stop=STOP_FORMAT_ERROR)
    assert refused is not None and refused.detail["check"] == "bad_stop"
