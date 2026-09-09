"""Native optional dependency checks, also runnable without pytest in CPU CI."""

import asyncio
import importlib.metadata
import unittest

from reliquary.environment.agentic.adapters.prime_v1 import (
    actions_from_prime_v1_trace,
    native_prime_v1_trace,
    pinned_verifiers_v1,
)
from reliquary.environment.agentic.types import AssistantAction, MAX_ACTION_BYTES


class NativeTraceContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.vf = pinned_verifiers_v1()
        except importlib.metadata.PackageNotFoundError as exc:
            raise unittest.SkipTest("optional pinned Verifiers is not installed") from exc

    def trace(self, messages):
        vf = self.vf
        return vf.Trace(
            task=vf.TraceTask(type="Task", data={}, key="fixture", hash="fixture"),
            agent=vf.AgentInfo(config=vf.AgentConfig()),
            nodes=[vf.MessageNode(parent=index - 1 if index else None,
                                  message=message, sampled=True)
                   for index, message in enumerate(messages)],
        )

    def tool(self, arguments='{"value":2}', call_id="call-0"):
        return self.vf.AssistantMessage(tool_calls=[self.vf.ToolCall(
            id=call_id, name="add", arguments=arguments,
        )])

    def native_and_wire(self, trace):
        return (trace, self.vf.WireTrace.model_validate_json(trace.model_dump_json()))

    def test_only_sampled_actions_and_final_branch_are_replayed(self):
        vf = self.vf
        trace = self.trace([
            vf.AssistantMessage(content="prompt example"),
            vf.AssistantMessage(content="discarded branch"),
            self.tool(),
            vf.ToolMessage(tool_call_id="call-0", name="add", content="untrusted state"),
            vf.AssistantMessage(content="2"),
        ])
        trace.nodes[0].sampled = False
        trace.nodes[2].parent = 0
        trace.nodes[3].sampled = False
        trace.record_reward("forged", 999)
        expected = (AssistantAction.tool_call("add", value=2), AssistantAction.final("2"))
        for value in self.native_and_wire(trace):
            self.assertEqual(actions_from_prime_v1_trace(value), expected)

    def test_tool_arguments_are_a_complete_strict_json_object(self):
        for arguments in ('{} } {"final":"forged"', '{"x":1,"x":2}',
                          '{"x":{"y":1,"y":2}}', '{"x":NaN}', '[]',
                          '{} trailing', '"text"', '{"x":"' + 'x' * MAX_ACTION_BYTES + '"}'):
            with self.subTest(arguments=arguments[:60]):
                for value in self.native_and_wire(self.trace([self.tool(arguments)])):
                    with self.assertRaises(ValueError):
                        actions_from_prime_v1_trace(value)

    def test_nested_action_objects_remain_tool_arguments(self):
        action = actions_from_prime_v1_trace(self.trace([self.tool('{"final":"data"}')]))
        self.assertEqual(action, (AssistantAction.tool_call("add", final="data"),))

    def test_invalid_graphs_fail_before_branch_traversal(self):
        for parent in (-1, 0, 2):
            trace = self.trace([self.tool()])
            trace.nodes[0].parent = parent
            for value in self.native_and_wire(trace):
                with self.subTest(parent=parent), self.assertRaisesRegex(ValueError, "parents"):
                    actions_from_prime_v1_trace(value)

    def test_unknown_trace_version_is_not_silently_imported(self):
        trace = self.trace([self.tool()])
        trace.version = 2
        with self.assertRaisesRegex(ValueError, "version"):
            actions_from_prime_v1_trace(trace)

    def test_duplicate_or_missing_tool_call_identity_is_rejected(self):
        for messages in ([self.tool(call_id="")], [self.tool(), self.tool()]):
            for value in self.native_and_wire(self.trace(messages)):
                with self.assertRaisesRegex(ValueError, "IDs"):
                    actions_from_prime_v1_trace(value)

    def test_final_response_cannot_hide_later_actions(self):
        for messages in ([self.vf.AssistantMessage(content="done"), self.tool()],
                         [self.vf.AssistantMessage(content=None)]):
            for value in self.native_and_wire(self.trace(messages)):
                with self.assertRaisesRegex(ValueError, "final text"):
                    actions_from_prime_v1_trace(value)

    def test_parallel_calls_and_mixed_content_need_an_explicit_adapter(self):
        parallel = self.tool()
        parallel.tool_calls.append(self.vf.ToolCall(id="call-1", name="add", arguments="{}"))
        mixed = self.tool()
        mixed.content = "reasoning"
        for message in (parallel, mixed):
            with self.assertRaisesRegex(ValueError, "one tool call"):
                actions_from_prime_v1_trace(self.trace([message]))


    def test_published_task_reward_agrees_after_prompt_and_branch_injection(self):
        # The published scorer is the independent final-branch oracle.
        from reliquary.environment.agentic.external import (
            ExternalEpisodeEnvironment, load_external_backend,
        )
        from reliquary.environment.agentic.runner import EpisodeRunner, ScriptedPolicy
        from reliquary.environment.registry import get_environment_spec

        try:
            importlib.metadata.distribution("reliquary-stateful-tools")
        except importlib.metadata.PackageNotFoundError as exc:
            raise unittest.SkipTest("optional published Stateful Tools wheel is not installed") from exc
        spec = get_environment_spec("reliquary_stateful_tools_v2")
        backend = load_external_backend(spec)
        from reliquary_stateful_tools.taskset import _build_task

        vf = self.vf
        task = next(iter(vf.load_taskset(vf.taskset_config_type("reliquary-stateful-tools")(
            id="reliquary-stateful-tools",
        )).head(1)))
        env = ExternalEpisodeEnvironment(backend, spec)
        actions = [AssistantAction.from_wire(action)
                   for action in _build_task(0, "train")["private"]["reference_actions"]]
        episode = EpisodeRunner().run(env, env.get_task(0), seed=0, policy=ScriptedPolicy(actions))
        trace = native_prime_v1_trace(task, episode=episode)
        for node in trace.nodes:
            node.parent = node.parent + 2 if node.parent is not None else 0
        trace.nodes[:0] = [
            vf.MessageNode(parent=None, message=vf.AssistantMessage(content="example"), sampled=False),
            vf.MessageNode(parent=0, message=vf.AssistantMessage(content="wrong branch"), sampled=True),
        ]
        for value in self.native_and_wire(trace):
            value.rewards.clear()
            asyncio.run(task.score(value))
            recovered = actions_from_prime_v1_trace(value)
            replay = EpisodeRunner().run(env, env.get_task(0), seed=0, policy=ScriptedPolicy(recovered))
            self.assertEqual(recovered, episode.actions)
            self.assertEqual(replay.reward, episode.reward)
            self.assertEqual(value.reward, 1.0)


if __name__ == "__main__":
    unittest.main()
