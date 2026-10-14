"""envscaler_tools_v2 episodes: contained, replayable, ended in ChatML, binary.

Every test here drives the real path — the world process, the runner, the
ChatML renderer, the validator's tokenized replay — over the two-world corpus
in ``envscaler_v2_fixtures``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from reliquary.environment.agentic.chatml import (
    CHATML_RENDERER_ID,
    IM_END,
    ChatMLEpisodeRenderer,
)
from reliquary.environment.agentic.envs.envscaler_tools_v2 import sandbox
from reliquary.environment.agentic.envs.envscaler_tools_v2.environment import (
    EnvScalerToolsV2Environment,
)
from reliquary.environment.agentic.renderers import renderer_for
from reliquary.environment.agentic.replay import replay_tokenized_episode
from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
from reliquary.environment.agentic.types import AssistantAction, GeneratedAction
from tests.unit import envscaler_v2_fixtures as fixtures

REPOSITORY = Path(__file__).resolve().parents[2]


def _encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def _decode(ids: list[int]) -> str:
    return bytes(ids).decode("utf-8")


class _ChatMLPolicy:
    """Writes each turn the way a ChatML-trained model would, or verbatim."""

    def __init__(self, turns: list[AssistantAction | str]) -> None:
        self._turns = iter(turns)

    def generate(self, **_: object) -> GeneratedAction:
        turn = next(self._turns)
        if isinstance(turn, str):
            return GeneratedAction(text=turn)
        return GeneratedAction(text=ChatMLEpisodeRenderer.action_text(
            turn, reasoning="Check the shop, then act."
        ))


def _environment() -> EnvScalerToolsV2Environment:
    return EnvScalerToolsV2Environment(fixtures.corpus())


def _call(tool: str, **arguments) -> AssistantAction:
    return AssistantAction(kind="tool", tool=tool, arguments=arguments)


def _run(index: int, actions: list[AssistantAction], environment=None):
    environment = environment or _environment()
    return EpisodeRunner().run(
        environment, environment.get_task(index), seed=3,
        policy=ScriptedPolicy(actions),
    )


def _chatml_run(index: int, turns: list[AssistantAction | str]):
    environment = _environment()
    return EpisodeRunner(renderer=renderer_for(CHATML_RENDERER_ID, _encode)).run(
        environment, environment.get_task(index), seed=11,
        policy=_ChatMLPolicy(turns),
    )


# ── reward mode ──────────────────────────────────────────────────────────────


def test_the_task_is_won_when_every_required_check_holds_and_it_finishes() -> None:
    trace = _run(fixtures.SHOP, [
        _call("add_item", name="apple", qty=2), AssistantAction.final("Added."),
    ])

    assert trace.termination_reason == "finished"
    assert trace.reward.reward == 1.0 and trace.reward.success
    weights = {c.name: c.weight for c in trace.reward.checks}
    # Only the check false at reset counts: one already true, one that cannot
    # run on the initial state.
    assert weights["check_0"] == 1.0
    assert weights["check_1"] == 0.0 and weights["check_2"] == 0.0


def test_a_no_op_agent_scores_zero() -> None:
    trace = _run(fixtures.SHOP, [AssistantAction.final("Nothing to do.")])

    assert trace.termination_reason == "finished"
    assert trace.reward.reward == 0.0


def test_a_scenario_with_nothing_to_do_cannot_be_won() -> None:
    """Every check already true: no required check, so no success to claim."""
    trace = _run(fixtures.SATISFIED, [AssistantAction.final("Done.")])

    assert trace.reward.reward == 0.0
    assert all(c.weight == 0.0 for c in trace.reward.checks)


def test_the_work_without_a_final_answer_is_not_a_win() -> None:
    trace = _run(fixtures.SHOP, [_call("add_item", name="apple")] * 3)
    environment = _environment()
    capped = EpisodeRunner(max_turns=1).run(
        environment, environment.get_task(fixtures.SHOP), seed=0,
        policy=ScriptedPolicy([_call("add_item", name="apple")]),
    )

    assert trace.termination_reason == "finished"  # the script ran out
    assert capped.termination_reason == "turn_limit"
    assert capped.reward.reward == 0.0
    assert {c.name: c.passed for c in capped.reward.checks}["check_0"] is True


def test_a_tool_that_raises_ends_the_episode_at_zero() -> None:
    trace = _run(fixtures.SHOP, [
        _call("add_item", name="apple"), _call("explode"),
        AssistantAction.final("Done."),
    ])

    assert trace.termination_reason == "tool_raised"
    assert "ValueError: boom" in trace.events[-1].content
    assert trace.reward.reward == 0.0


def test_an_unknown_tool_is_recoverable() -> None:
    trace = _run(fixtures.SHOP, [
        _call("teleport"), _call("add_item", name="apple"),
        AssistantAction.final("Done."),
    ])

    assert trace.termination_reason == "finished"
    assert "unknown tool" in trace.events[2].content
    assert trace.reward.reward == 1.0


def test_the_reward_mode_is_not_read_from_the_host(monkeypatch) -> None:
    """v1 switched modes on RELIQUARY_ENVSCALER_REWARD; v2 has one mode."""
    actions = [AssistantAction.final("Nothing to do.")]
    baseline = _run(fixtures.SHOP, actions).reward

    monkeypatch.setenv("RELIQUARY_ENVSCALER_REWARD", "upstream")

    assert _run(fixtures.SHOP, actions).reward == baseline


def test_the_ungraded_range_always_scores_zero() -> None:
    trace = _run(fixtures.UNGRADED, [
        _call("add_item", name="apple"), AssistantAction.final("Stocked."),
    ])

    assert trace.termination_reason == "finished"
    assert trace.reward.reward == 0.0
    assert trace.reward.checks[0].name == "ungraded"
    assert len(trace.reward.state_digest) == 64


# ── ChatML ───────────────────────────────────────────────────────────────────


def test_a_chatml_answer_without_a_call_ends_the_episode() -> None:
    trace = _chatml_run(fixtures.SHOP, [
        _call("add_item", name="apple", qty=3),
        AssistantAction.final("The apple is in stock."),
    ])

    assert trace.termination_reason == "finished"
    assert trace.reward.reward == 1.0
    # The model's turn stops on <|im_end|>; the runner closes the last one.
    assert _decode(list(trace.tokens)).endswith(
        "The apple is in stock." + IM_END + "\n"
    )
    # Arguments are typed from the tool schema, not left as text.
    assert trace.actions[0].arguments == {"name": "apple", "qty": 3}


def test_a_turn_ending_on_its_own_terminator_is_read() -> None:
    turn = ChatMLEpisodeRenderer.action_text(
        _call("add_item", name="apple"), reasoning="Add it."
    ) + IM_END
    trace = _chatml_run(fixtures.SHOP, [turn, AssistantAction.final("Done.")])

    assert trace.actions[0].tool == "add_item"
    assert trace.reward.reward == 1.0


def test_an_unreadable_chatml_turn_ends_the_episode_at_zero() -> None:
    unclosed = "I will add the apple <tool_call>\n<function=add_item>"
    trace = _chatml_run(fixtures.SHOP, [unclosed])

    assert trace.termination_reason == "invalid_action"
    assert trace.reward.reward == 0.0


def test_the_validator_replays_a_chatml_episode_token_for_token() -> None:
    miner = _chatml_run(fixtures.SHOP, [
        _call("add_item", name="apple", qty=2), _call("list_items"),
        AssistantAction.final("Done."),
    ])

    replayed = replay_tokenized_episode(
        _environment(), task_index=fixtures.SHOP, seed=11,
        tokens=list(miner.tokens), assistant_spans=miner.assistant_spans,
        decode=_decode, encode=_encode, renderer_id=CHATML_RENDERER_ID,
    )

    assert replayed.tokens == miner.tokens
    assert replayed.observation_digests == miner.observation_digests
    assert replayed.reward == miner.reward


# ── determinism ──────────────────────────────────────────────────────────────


_REPLAY_IN_A_FRESH_PROCESS = """
import json, sys
from reliquary.environment.agentic.chatml import CHATML_RENDERER_ID, ChatMLEpisodeRenderer
from reliquary.environment.agentic.envs.envscaler_tools_v2.environment import EnvScalerToolsV2Environment
from reliquary.environment.agentic.renderers import renderer_for
from reliquary.environment.agentic.runner import EpisodeRunner
from reliquary.environment.agentic.types import AssistantAction, GeneratedAction
from tests.unit import envscaler_v2_fixtures as fixtures

class Policy:
    def __init__(self, actions):
        self._actions = iter(actions)
    def generate(self, **_):
        return GeneratedAction(text=ChatMLEpisodeRenderer.action_text(next(self._actions), reasoning="go"))

actions = [
    AssistantAction(kind="tool", tool="add_item", arguments={"name": "apple", "qty": 2}),
    AssistantAction(kind="tool", tool="add_item", arguments={"name": "pear", "qty": 5}),
    AssistantAction.final("Done."),
]
environment = EnvScalerToolsV2Environment(fixtures.corpus())
encode = lambda text: list(text.encode("utf-8"))
trace = EpisodeRunner(renderer=renderer_for(CHATML_RENDERER_ID, encode)).run(
    environment, environment.get_task(fixtures.SHOP), seed=5, policy=Policy(actions))
json.dump({
    "tokens": list(trace.tokens),
    "observations": [e.content for e in trace.events if e.role == "tool"],
    "reward": trace.reward.reward,
    "state_digest": trace.reward.state_digest,
}, sys.stdout)
"""


def _episode_in_a_fresh_process(hash_seed: str) -> dict:
    environment = {
        key: value for key, value in os.environ.items()
        if key not in ("PYTHONHASHSEED", "TZ")
    }
    environment.update({"PYTHONHASHSEED": hash_seed, "TZ": "Asia/Tokyo"})
    completed = subprocess.run(
        [sys.executable, "-c", _REPLAY_IN_A_FRESH_PROCESS],
        cwd=REPOSITORY, env=environment, capture_output=True, text=True,
        timeout=120, check=True,
    )
    return json.loads(completed.stdout)


def test_an_episode_is_byte_identical_across_processes() -> None:
    """Different hash seeds and time zones on the host change nothing.

    The world reads the clock, draws uuids and random numbers, and returns a
    set; each would differ between two hosts if the world process did not pin
    or seed it.
    """
    first = _episode_in_a_fresh_process("1")
    second = _episode_in_a_fresh_process("4242")

    assert first == second
    assert first["reward"] == 1.0
    observation = json.loads(first["observations"][0])
    assert observation["data"]["id"] == "00000000-0000-0000-0000-000000000001"
    assert observation["data"]["at"] == "2023-11-14T22:13:20"


def test_two_rollouts_in_one_process_see_the_same_world() -> None:
    """v1 cached the class, so the second rollout's uuids continued the first's."""
    environment = _environment()
    actions = [_call("add_item", name="apple"), AssistantAction.final("Done.")]
    first = _run(fixtures.SHOP, actions, environment)
    second = _run(fixtures.SHOP, actions, environment)

    assert first.events == second.events
    assert first.reward.state_digest == second.reward.state_digest


# ── isolation ────────────────────────────────────────────────────────────────


def _hostile(*actions: AssistantAction):
    return _run(fixtures.HOSTILE, [*actions, AssistantAction.final("Done.")])


def _error(trace) -> str:
    return json.loads(trace.events[-1].content)["error"]


def test_a_world_cannot_write_a_file(tmp_path) -> None:
    target = tmp_path / "owned.txt"
    trace = _hostile(_call("write_file", path=str(target)))

    assert trace.termination_reason == "tool_raised"
    assert _error(trace).startswith("PermissionError")
    assert not target.exists()
    assert trace.reward.reward == 0.0


def test_a_world_cannot_read_the_host(tmp_path) -> None:
    secret = tmp_path / "token"
    secret.write_text("hf_secret")
    trace = _hostile(_call("read_file", path=str(secret)))

    assert trace.termination_reason == "tool_raised"
    assert "hf_secret" not in trace.events[-1].content


def test_a_world_cannot_open_a_socket() -> None:
    trace = _hostile(_call("connect", host="127.0.0.1", port=22))

    assert trace.termination_reason == "tool_raised"
    assert _error(trace).startswith("PermissionError")


def test_a_world_cannot_start_a_process(tmp_path) -> None:
    marker = tmp_path / "ran"
    by_shell = _hostile(_call("spawn", command=f"touch {marker}"))
    by_fork = _hostile(_call("fork"))

    assert by_shell.termination_reason == "tool_raised"
    assert by_fork.termination_reason == "tool_raised"
    assert not marker.exists()


def test_a_world_sees_none_of_the_host_environment(monkeypatch) -> None:
    monkeypatch.setenv("HF_TOKEN", "hf_secret")
    trace = _hostile(_call("environment"))

    seen = json.loads(trace.events[2].content)["data"]
    assert "HF_TOKEN" not in seen
    assert seen == dict(sandbox.WORKER_ENVIRONMENT)


def test_a_memory_bomb_is_a_clean_failure() -> None:
    trace = _hostile(_call("hog"))

    assert trace.termination_reason == "tool_raised"
    assert _error(trace).startswith("MemoryError")


def test_an_infinite_loop_is_a_clean_failure(monkeypatch) -> None:
    """The CPU limit is the bound in production; the wall clock is quicker here."""
    monkeypatch.setattr(sandbox, "REQUEST_SECONDS", 2.0)
    trace = _hostile(_call("spin"))

    assert trace.termination_reason == "world_error"
    assert _error(trace) == sandbox.TERMINATED
    assert trace.reward.reward == 0.0
    assert trace.reward.environment_error == sandbox.TERMINATED


def test_the_cpu_limit_kills_a_spinning_world() -> None:
    process = sandbox.WorldProcess()
    try:
        assert process.request({
            "op": "init", "source": fixtures.HOSTILE_SOURCE,
            "class_name": "Hostile", "init_config": {},
        })["ok"]
        with pytest.raises(sandbox.WorldProcessError):
            process.request({"op": "call", "tool": "spin", "arguments": {}})
        assert not process.alive
    finally:
        process.close()


def test_private_methods_are_not_tools() -> None:
    process = sandbox.WorldProcess()
    try:
        process.request({
            "op": "init", "source": fixtures.SHOP_SOURCE,
            "class_name": "Shop", "init_config": {},
        })
        reply = process.request({"op": "call", "tool": "__init__", "arguments": {}})
        assert reply["ok"] is False
    finally:
        process.close()


def test_closing_the_episode_ends_the_world() -> None:
    environment = _environment()
    task = environment.get_task(fixtures.SHOP)
    state = environment.reset(task, 0).state
    process = state.process
    assert process.alive

    environment.close(state)

    assert not process.alive


_CLOCK_PROBE = """
import random, time, uuid
from datetime import datetime


class Probe:
    def __init__(self):
        self.n = 0

    def probe(self):
        own = random.Random()
        random.seed()
        return {
            "now": time.time(),
            "converted": time.strftime("%Y-%m-%d", time.gmtime(0)),
            "ctime": time.ctime(86400),
            "own": own.random(),
            "module": random.random(),
            "named": str(uuid.uuid5(uuid.NAMESPACE_DNS, "reliquary.ai")),
            "counted": str(uuid.uuid4()),
            "today": datetime.now().isoformat(),
        }
"""


def test_the_stand_ins_freeze_the_clock_but_keep_conversions() -> None:
    import random as host_random
    import uuid as host_uuid

    process = sandbox.WorldProcess()
    try:
        process.request({
            "op": "init", "source": _CLOCK_PROBE, "class_name": "Probe",
            "init_config": {},
        })
        result = process.request(
            {"op": "call", "tool": "probe", "arguments": {}}
        )["result"]
    finally:
        process.close()

    assert result["now"] == 1_700_000_000.0
    assert result["today"] == "2023-11-14T22:13:20"
    # Given an instant, the clock functions convert it rather than ignore it.
    assert result["converted"] == "1970-01-01"
    assert result["ctime"] == "Fri Jan  2 00:00:00 1970"
    # A world's own generator, and a reseed, both use the constant seed.
    assert result["own"] == host_random.Random(0).random()
    assert result["module"] == host_random.Random(0).random()
    # uuid5 is a hash and stays one; uuid4 counts.
    assert result["named"] == str(
        host_uuid.uuid5(host_uuid.NAMESPACE_DNS, "reliquary.ai")
    )
    assert result["counted"] == "00000000-0000-0000-0000-000000000001"
