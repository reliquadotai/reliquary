"""reliquary_automationbench_v1 through the suite path: loaded from its verified
wheel, played in ChatML as a miner plays it, replayed from tokens as the
validator does. Skipped without the `reliquary-automationbench` 0.1.0a1 wheel."""

from __future__ import annotations

import json

import pytest

from reliquary.corpus.suite_episode import episode_seed
from reliquary.environment.agentic.chatml import CHATML_RENDERER_ID, ChatMLEpisodeRenderer
from reliquary.environment.agentic.replay import replay_tokenized_episode
from reliquary.environment.agentic.runner import EpisodeRunner
from reliquary.environment.agentic.types import AssistantAction
from reliquary.environment.registry import ENVIRONMENT_SPECS
from tests.unit.test_corpus_suite_episode import CODEC, _policy

NAME = "reliquary_automationbench_v1"


@pytest.fixture(scope="module")
def environment():
    try:
        return ENVIRONMENT_SPECS[NAME].create(split="eval")
    except Exception as exc:  # noqa: BLE001 - absent or another version installed
        pytest.skip(f"reliquary-automationbench 0.1.0a1 not installed: {exc}")


def _fetch(method, url, body=None, params=None):
    arguments = {"method": method, "url": url}
    if params is not None:
        arguments["params"] = json.dumps(params)
    if body is not None:
        arguments["body"] = json.dumps(body)
    return AssistantAction(kind="tool", tool="api_fetch", arguments=arguments)


def _actions():
    return [
        AssistantAction(kind="tool", tool="api_search", arguments={"query": "gmail send message", "top_k": 2}),
        _fetch("GET", "https://gmail.googleapis.com/gmail/v1/users/me/messages", params={"maxResults": 3}),
        _fetch("POST", "https://slack.com/api/chat.postMessage", body={"channel": "general", "text": "hello"}),
        _fetch("POST", "https://gmail.googleapis.com/gmail/v1/users/me/messages/send",
               body={"to": "team@example.com", "subject": "Update", "body": "Done."}),
        AssistantAction.final("Done."),
    ]


@pytest.mark.parametrize("index", [0, 150, 599])
def test_a_chatml_episode_replays_from_its_tokens(environment, index):
    task = environment.get_task(index)
    assert task.metadata["domain"] == ("sales", "marketing", "hr")[[0, 150, 599].index(index)]
    renderer = ChatMLEpisodeRenderer(CODEC.encode)
    seed = episode_seed("automationbench-order", index)
    trace = EpisodeRunner(renderer=renderer, max_turns=environment.max_turns).run(
        environment, task, seed=seed, policy=_policy(_actions(), CODEC))
    assert trace.termination_reason == "finished"
    assert [a.tool for a in trace.actions[:-1]] == ["api_search", "api_fetch", "api_fetch", "api_fetch"]
    assert trace.reward is not None and trace.reward.reward in (0.0, 1.0)

    replayed = replay_tokenized_episode(
        environment, task_index=index, seed=seed, tokens=list(trace.tokens),
        assistant_spans=trace.assistant_spans, decode=CODEC.decode, encode=CODEC.encode,
        renderer_id=CHATML_RENDERER_ID)
    assert replayed.tokens == trace.tokens
    assert replayed.observation_digests == trace.observation_digests
    assert replayed.reward == trace.reward


def test_the_spec_binds_the_wheel_and_has_no_train_split(environment):
    spec = ENVIRONMENT_SPECS[NAME]
    assert spec.interaction_mode == "episode" and spec.renderer_id == CHATML_RENDERER_ID
    assert len(environment) == 600
    assert len(spec.create(split="qualification")) == 200
    with pytest.raises(Exception):
        spec.create()  # train: AutomationBench is held out
