"""Frozen-corpus search environment with exact answer and citation checks."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import random
import re
from typing import Any

from reliquary.environment.agentic.types import (
    AssistantAction,
    EpisodeEvent,
    EpisodeTask,
    EpisodeTrace,
    ResetResult,
    RewardCheck,
    RewardReport,
    StepResult,
    ToolSpec,
    canonical_json,
    sha256_json,
)


TASK_COUNT = 1 << 31
GENERATOR_VERSION = "reliquary-retrieval-tools-generator-v1"


def _schema(properties: dict, required: list[str]) -> dict:
    return {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }


TOOLS = (
    ToolSpec(
        "search_documents",
        "Search the frozen document collection and return matching document IDs.",
        _schema({"query": {"type": "string"}}, ["query"]),
    ),
    ToolSpec(
        "open_document",
        "Open one document by document ID.",
        _schema({"document_id": {"type": "string"}}, ["document_id"]),
    ),
    ToolSpec(
        "finish",
        "Return the answer and the document IDs that directly support it.",
        _schema(
            {
                "response": {"type": "string"},
                "citations": {"type": "array", "items": {"type": "string"}},
            },
            ["response", "citations"],
        ),
    ),
)


@dataclass(slots=True)
class RetrievalState:
    documents: dict[str, str]
    opened: set[str]
    final_response: str | None = None
    citations: tuple[str, ...] = ()
    invalid_actions: int = 0
    closed: bool = False


def _build_task(index: int) -> EpisodeTask:
    index = int(index) % TASK_COUNT
    digest = hashlib.sha256(f"{GENERATOR_VERSION}:{index}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:16], "big"))
    project = f"Project-{rng.randrange(1000, 9999)}"
    code = f"ZX-{rng.randrange(10000, 99999)}"
    target_id = f"doc-{rng.randrange(100000, 999999)}"
    distractor_ids = [f"doc-{rng.randrange(100000, 999999)}" for _ in range(4)]
    documents = {
        target_id: (
            f"Operations memo for {project}. The approved recovery code is {code}. "
            "This supersedes all earlier drafts."
        ),
        distractor_ids[0]: f"Meeting notes for {project}; recovery approval is pending.",
        distractor_ids[1]: "General incident response handbook with no project codes.",
        distractor_ids[2]: f"An obsolete draft mentions code ZZ-{rng.randrange(10000, 99999)}.",
        distractor_ids[3]: "Procurement schedule and unrelated delivery dates.",
    }
    return EpisodeTask(
        id=hashlib.sha256(
            f"reliquary_retrieval_tools_v1:{index}".encode()
        ).hexdigest()[:16],
        prompt=(
            f"Find the currently approved recovery code for {project}. Answer with "
            "the code and cite only documents that directly support it."
        ),
        tools=TOOLS,
        metadata={
            "family": "frozen_evidence_retrieval",
            "generator_version": GENERATOR_VERSION,
            "generator_index": index,
            "difficulty": 1 + index % 3,
        },
        private={
            "documents": documents,
            "target_document_id": target_id,
            "answer": code,
            "reference_actions": (
                AssistantAction.tool_call("search_documents", query=project),
                AssistantAction.tool_call("open_document", document_id=target_id),
                AssistantAction.tool_call(
                    "finish", response=code, citations=[target_id]
                ),
            ),
        },
    )


class RetrievalToolsEnvironment:
    name = "reliquary_retrieval_tools_v1"
    validator_authoritative_reward = True
    max_turns = 6

    def __len__(self) -> int:
        return TASK_COUNT

    def get_task(self, index: int) -> EpisodeTask:
        return _build_task(index)

    def get_problem(self, index: int) -> dict[str, Any]:
        from reliquary.environment.agentic.compat import episode_problem

        return episode_problem(self, index)

    def reset(self, task: EpisodeTask, seed: int) -> ResetResult:
        del seed
        return ResetResult(
            state=RetrievalState(documents=dict(task.private["documents"]), opened=set())
        )

    @staticmethod
    def _event(name: str, value: dict[str, Any]) -> tuple[EpisodeEvent, ...]:
        return (EpisodeEvent(role="tool", name=name, content=canonical_json(value)),)

    def step(
        self,
        task: EpisodeTask,
        state: RetrievalState,
        action: AssistantAction,
    ) -> StepResult:
        del task
        if state.closed:
            raise RuntimeError("episode state is closed")
        if action.kind == "final":
            state.final_response = action.content or ""
            return StepResult(
                state,
                self._event("final", {"accepted": True}),
                done=True,
                termination_reason="finished",
            )
        name = str(action.tool)
        arguments = dict(action.arguments)
        try:
            if name == "search_documents":
                if set(arguments) != {"query"}:
                    raise ValueError("search_documents requires query")
                terms = set(re.findall(r"[a-z0-9-]+", str(arguments["query"]).lower()))
                if not terms:
                    raise ValueError("query must not be empty")
                scored = []
                for document_id, content in state.documents.items():
                    haystack = set(re.findall(r"[a-z0-9-]+", content.lower()))
                    score = len(terms & haystack)
                    if score:
                        scored.append((score, document_id))
                scored.sort(key=lambda item: (-item[0], item[1]))
                value = {"results": [item[1] for item in scored[:5]]}
            elif name == "open_document":
                if set(arguments) != {"document_id"}:
                    raise ValueError("open_document requires document_id")
                document_id = str(arguments["document_id"])
                if document_id not in state.documents:
                    raise ValueError("unknown document_id")
                state.opened.add(document_id)
                value = {
                    "document_id": document_id,
                    "content": state.documents[document_id],
                }
            elif name == "finish":
                if set(arguments) != {"response", "citations"}:
                    raise ValueError("finish requires response and citations")
                citations = arguments["citations"]
                if not isinstance(citations, list) or not all(
                    isinstance(value, str) for value in citations
                ):
                    raise ValueError("citations must be a list of strings")
                state.final_response = str(arguments["response"])
                state.citations = tuple(citations)
                return StepResult(
                    state,
                    self._event("finish", {"accepted": True}),
                    done=True,
                    termination_reason="finished",
                )
            else:
                raise ValueError(f"unknown tool: {name}")
            return StepResult(state, self._event(name, {"ok": True, **value}))
        except (TypeError, ValueError) as exc:
            state.invalid_actions += 1
            return StepResult(
                state,
                self._event(name, {"ok": False, "error": str(exc)[:512]}),
            )

    def grade(
        self,
        task: EpisodeTask,
        state: RetrievalState,
        trace: EpisodeTrace,
    ) -> RewardReport:
        target = str(task.private["target_document_id"])
        answer = str(task.private["answer"])
        citations = set(state.citations)
        checks = (
            RewardCheck("answer_exact", answer.lower() in (state.final_response or "").lower(), 2.0),
            RewardCheck("target_document_opened", target in state.opened, 1.0),
            RewardCheck("target_document_cited", target in citations, 2.0),
            RewardCheck("citations_are_opened", citations <= state.opened, 1.0),
            RewardCheck("citations_are_minimal", citations == {target}, 1.0),
            RewardCheck("no_invalid_tool_calls", state.invalid_actions == 0, 0.5),
            RewardCheck("finished_explicitly", trace.termination_reason == "finished", 0.5),
        )
        digest = sha256_json(
            {
                "opened": sorted(state.opened),
                "response": state.final_response,
                "citations": list(state.citations),
            }
        )
        return RewardReport.from_checks(checks, state_digest=digest)

    def close(self, state: RetrievalState) -> None:
        state.closed = True
