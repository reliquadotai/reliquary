"""Source-bound signed episode evaluation through the existing sandbox harness."""

from __future__ import annotations

from dataclasses import asdict
import asyncio
import base64
import hashlib
import time

from reliquary.eval.protocol import digest
from reliquary.eval.sets import prompt_sha256


async def resolve_task(source, index: int) -> dict:
    """The same actual task identity used by set freezing and session issuance."""
    prompt = await asyncio.to_thread(source.prompt, index)
    task = asdict(await source.resolve(index))
    return {"source_index": index, "prompt_sha256": prompt_sha256(prompt),
            "reference_sha256": digest({"prompt": prompt, "task": task}), "task": task}


def checkpoint_identity(model: dict) -> str:
    from reliquary.eval.runner import validate_model

    validate_model(model["repo"], model["revision"])
    return digest({"schema": "hf-snapshot/v1", **model})


def model_client(generator, *, task, source, deadline, seed, cli_bridge=None):
    """The pinned harness's existing interception calls the owned local model only."""
    from verifiers.v1.clients import Client
    from verifiers.v1.dialects import AnthropicDialect, ChatDialect, ResponsesDialect
    from verifiers.v1.clients.client import RelayReply
    from verifiers.v1.clients.train import message_to_wire, tool_to_wire
    from verifiers.v1.types import Response, TurnTokens
    from reliquary.eval.model_wire import completion_events, native_completion, restore_local_thinking

    model = task["model"]["repo"] + "@" + task["model"]["revision"]
    limits = task["protocol"]
    dialect_type = (ChatDialect if cli_bridge is None else {
        "signed-codex-cli-v1": ResponsesDialect,
        "signed-claude-code-cli-v1": AnthropicDialect,
    }[cli_bridge.capability.protocol_id])

    class LocalClient(Client):
        turns = 0
        tool_calls = 0
        token_ids = ()
        generated_spans = ()
        transcript_tokens = 0
        finish_reason = None

        def __init__(self):
            self._serial = asyncio.Lock()

        def _request(self, dialect, body, turn=None):
            if type(dialect) is not dialect_type or body.get("model") != model:
                raise ValueError("signed harness requested another model or dialect")
            request = dialect.parse_request(body)
            if len(body.get("tools") or []) != len(request.tools or []):
                raise ValueError("signed harness offered unsupported native tools")
            messages = [message_to_wire(message) for message in (turn.prompt if turn is not None else request.messages)]
            messages = restore_local_thinking(messages, dialect)
            tools = [tool_to_wire(tool) for tool in request.tools] if request.tools else None
            if digest(tools or []) != source["tools_sha256"]:
                raise ValueError("signed harness tool schemas changed")
            if cli_bridge is not None:
                cli_bridge.capability.require_model_tools(tools or [])
                cli_bridge.validate_messages(messages)
            return messages, tools

        async def get_response(self, dialect, body, sampling, session_id=None, turn=None, headers=None):
            async with self._serial:
                if self.turns >= limits["max_turns"] or time.monotonic() >= deadline:
                    raise TimeoutError("signed episode turn or time budget exhausted")
                messages, tools = self._request(dialect, body, turn)
                try:
                    ids = await asyncio.to_thread(generator.render_messages, messages, tools=tools,
                        thinking=task["thinking"], renderer_id=source["renderer_id"])
                    remaining = limits["max_episode_tokens"] - len(ids)
                    if remaining <= 0:
                        raise ValueError("episode transcript token budget exhausted")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("signed episode rendering exceeded its deadline")
                    seed_for_turn = (seed + self.turns) % 2**31
                    self.turns += 1
                    maximum = min(task["max_new_tokens"], remaining)
                    result = await asyncio.to_thread(generator.generate_turn, ids,
                        max_new_tokens=maximum, seed=seed_for_turn,
                        sampling=task["sampling"], absolute_deadline=deadline)
                except asyncio.CancelledError:
                    generator.close()
                    raise
                if time.monotonic() >= deadline:
                    raise TimeoutError("signed episode inference exceeded its deadline")
                tokens = result["token_ids"]
                if len(tokens) > maximum or any(type(token) is not int or token < 0 for token in tokens):
                    raise ValueError("signed model exceeded its frozen token budget")
                response = Response.model_validate({"id": f"evaluation-turn-{self.turns}", "created": 0,
                    "model": model, "message": result["message"], "finish_reason": result["finish_reason"],
                    "usage": {"prompt_tokens": len(ids), "completion_tokens": len(tokens)}})
                response.tokens = TurnTokens(prompt_ids=ids, completion_ids=tokens)
                response.raw = native_completion(response, dialect)
                if cli_bridge is not None:
                    cli_bridge.register_turn(self.turns - 1, [
                        (call.id, call.name, call.arguments) for call in response.message.tool_calls or ()],
                        assistant_message=message_to_wire(response.message))
                self.generated_spans += ((len(self.token_ids), len(self.token_ids) + len(tokens)),)
                self.token_ids += tuple(tokens)
                self.transcript_tokens = max(self.transcript_tokens, len(ids) + len(tokens))
                self.finish_reason = result["finish_reason"]
                self.tool_calls += len(response.message.tool_calls or ())
                return response

        async def relay(self, dialect, body, session_id=None, headers=None):
            response = await self.get_response(dialect, body, None, session_id=session_id, headers=headers)

            async def chunks():
                for chunk in completion_events(response.raw, dialect):
                    yield chunk

            async def close():
                pass

            return RelayReply("text/event-stream", chunks(), close)

        async def relay_aux(self, dialect, route, body, headers=None):
            if type(dialect) is not AnthropicDialect or route != "/v1/messages/count_tokens":
                raise ValueError("unsupported local auxiliary request")
            async with self._serial:
                if time.monotonic() >= deadline:
                    raise TimeoutError("signed episode time budget exhausted")
                messages, tools = self._request(dialect, body)
                ids = await asyncio.to_thread(generator.render_messages, messages, tools=tools,
                    thinking=task["thinking"], renderer_id=source["renderer_id"])
                return {"input_tokens": len(ids)}

    return LocalClient()


async def run_signed_episode(*, generator, task, row, sample_index, grant, seed, deadline):
    from reliquary.environment.agentic_swe import SUPPORTED_VERIFIERS, _dist_commit
    from reliquary_sandbox_verifiers import SandboxEpisodeRunner
    from reliquary_sandbox_verifiers.runner import env_config
    from verifiers.v1.clients import ModelContext
    from verifiers.v1.configs.client import EvalClientConfig
    from verifiers.v1.interception import InterceptionServer
    from verifiers.v1.types import Sampling
    from verifiers.v1.utils.loaders import load_environment, resolve_env_config

    if _dist_commit("verifiers") != SUPPORTED_VERIFIERS:
        raise ValueError("signed evaluation requires the pinned Verifiers implementation")
    expected = {"eval_id": task["eval_id"], "task_id": task["task_id"], "set_id": row["set_id"],
        "problem_id": row["problem_id"], "sample_index": sample_index,
        "model": task["model"], "protocol_sha256": task["protocol_sha256"],
        "identity_kind": "hf-snapshot/v1", "checkpoint": checkpoint_identity(task["model"])}
    binding = grant.get("binding", {})
    if any(binding.get(key) != value for key, value in expected.items()):
        raise ValueError("signed session does not bind the actual evaluation sample")
    grant = grant.get("grant")
    if (not isinstance(grant, dict) or not isinstance(grant.get("token"), dict) or not grant["token"]
            or not all(isinstance(grant.get(key), str) and grant[key]
                       for key in ("gateway_url", "session_id"))):
        raise ValueError("signed session grant is invalid")
    source = task["set_protocols"][row["set_id"]]["protocol"]["source"]["episode"]
    if time.monotonic() >= deadline:
        raise TimeoutError("signed episode session preparation exceeded its deadline")
    client = model_client(generator, task=task, source=source, deadline=deadline, seed=seed)
    # Explicit unused endpoint config is a key only; every model call is injected locally.
    config = EvalClientConfig(base_url="http://127.0.0.1", api_key_var="RELIQUARY_EVAL_UNUSED_KEY")
    context = ModelContext(model=task["model"]["repo"] + "@" + task["model"]["revision"],
                           client=config, sampling=Sampling(max_tokens=task["max_new_tokens"]))

    class PinnedRunner(SandboxEpisodeRunner):
        async def _env(self, tools):
            key = tuple(sorted(tools))
            if key not in self._envs:
                settings = {**env_config(tools=key, max_turns=task["protocol"]["max_turns"]),
                            "interception": {"type": "server"}}
                env = load_environment(resolve_env_config(settings))
                await self._stack.enter_async_context(env.serving())
                if not isinstance(env._interception, InterceptionServer):
                    raise ValueError("signed harness interception implementation changed")
                env._interception.clients[config.model_dump_json()] = client
                self._envs[key] = env
            return self._envs[key]

    async with PinnedRunner(ctx=context, gateway_url=grant["gateway_url"],
                            max_turns=task["protocol"]["max_turns"]) as runner:
        async with asyncio.timeout_at(deadline):
            result = await runner.run(token=grant["token"], prompt=row["messages"][0]["content"])
    if result.error or result.final.get("body", {}).get("status") != "graded":
        raise ValueError("signed episode did not produce a graded final")
    return {"completion": "", "completion_tokens": len(client.token_ids),
            "finish_reason": client.finish_reason or "stop", "turns": client.turns,
            "tool_calls": client.tool_calls, "transcript_tokens": client.transcript_tokens,
            "episode": {"binding": binding, "transcript": result.transcript,
                        "stop_reason": result.final["body"]["status"],
                        "model_token_ids": list(client.token_ids),
                        "generated_spans": list(client.generated_spans),
                        "state_base64": None if result.state is None else base64.b64encode(result.state).decode("ascii"),
                        "state_sha256": None if result.state is None else hashlib.sha256(result.state).hexdigest()}}
