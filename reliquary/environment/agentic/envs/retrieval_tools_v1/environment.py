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
    base_id = rng.randrange(100000, 999990)
    document_ids = [f"doc-{base_id + offset}" for offset in range(6)]
    family = index % 3
    if family == 0:
        target_id = document_ids[0]
        documents = {
            target_id: (
                f"Operations memo for {project}. The approved recovery code is "
                f"{code}. This supersedes all earlier drafts."
            ),
            document_ids[1]: (
                f"Meeting notes for {project}; recovery approval is pending."
            ),
            document_ids[2]: "General incident response handbook with no codes.",
            document_ids[3]: (
                "An obsolete draft mentions code "
                f"ZZ-{rng.randrange(10000, 99999)}."
            ),
            document_ids[4]: "Procurement schedule and unrelated delivery dates.",
        }
        prompt = (
            f"Find the currently approved recovery code for {project}. Answer "
            "with the code and cite only documents that directly support it."
        )
        required_opened = (target_id,)
        expected_citations = (target_id,)
        reference = (
            AssistantAction.tool_call("search_documents", query=project),
            AssistantAction.tool_call("open_document", document_id=target_id),
            AssistantAction.tool_call(
                "finish", response=code, citations=[target_id]
            ),
        )
        family_name = "single_evidence"
    elif family == 1:
        alias = f"Cluster-{rng.randrange(100, 999)}"
        link_id, code_id = document_ids[:2]
        documents = {
            link_id: (
                f"Deployment registry: {project} currently maps to {alias}."
            ),
            code_id: (
                f"Recovery ledger for {alias}: approved code {code}."
            ),
            document_ids[2]: f"Archive: {project} previously used Cluster-001.",
            document_ids[3]: "General recovery policy without project mappings.",
            document_ids[4]: "Unrelated cluster maintenance schedule.",
        }
        prompt = (
            f"Resolve the current deployment alias for {project}, then find its "
            "approved recovery code. Answer with the code and cite both documents "
            "needed to establish the chain."
        )
        required_opened = (link_id, code_id)
        expected_citations = (link_id, code_id)
        reference = (
            AssistantAction.tool_call("search_documents", query=project),
            AssistantAction.tool_call("open_document", document_id=link_id),
            AssistantAction.tool_call("search_documents", query=alias),
            AssistantAction.tool_call("open_document", document_id=code_id),
            AssistantAction.tool_call(
                "finish", response=code, citations=[link_id, code_id]
            ),
        )
        family_name = "multi_hop_alias"
    else:
        old_id, current_id = document_ids[:2]
        old_code = f"ZZ-{rng.randrange(10000, 99999)}"
        documents = {
            old_id: (
                f"Revision 1 for {project}: recovery code {old_code}. Superseded."
            ),
            current_id: (
                f"Revision 2 for {project}: recovery code {code}. Current and "
                "approved."
            ),
            document_ids[2]: f"Meeting agenda for {project}; no approval decision.",
            document_ids[3]: "General document-retention policy.",
            document_ids[4]: "Unrelated incident report.",
        }
        prompt = (
            f"Compare the available revisions for {project} and return the current "
            "approved recovery code. Cite only the current authoritative revision."
        )
        required_opened = (old_id, current_id)
        expected_citations = (current_id,)
        reference = (
            AssistantAction.tool_call("search_documents", query=project),
            AssistantAction.tool_call("open_document", document_id=old_id),
            AssistantAction.tool_call("open_document", document_id=current_id),
            AssistantAction.tool_call(
                "finish", response=code, citations=[current_id]
            ),
        )
        family_name = "revision_resolution"
    return EpisodeTask(
        id=hashlib.sha256(
            f"reliquary_retrieval_tools_v1:{index}".encode()
        ).hexdigest()[:16],
        prompt=prompt,
        tools=TOOLS,
        metadata={
            "family": family_name,
            "generator_version": GENERATOR_VERSION,
            "generator_index": index,
            "difficulty": 1 + index % 3,
        },
        private={
            "documents": documents,
            "required_opened": required_opened,
            "expected_citations": expected_citations,
            "answer": code,
            "reference_actions": reference,
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
                query = str(arguments["query"])
                if not query or len(query) > 512:
                    raise ValueError("query must contain 1..512 characters")
                terms = set(re.findall(r"[a-z0-9-]+", query.lower()))
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
                if len(citations) > 16:
                    raise ValueError("citations must contain at most 16 IDs")
                response = str(arguments["response"])
                if not response or len(response) > 4096:
                    raise ValueError("response must contain 1..4096 characters")
                state.final_response = response
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
        required_opened = set(task.private["required_opened"])
        expected_citations = tuple(task.private["expected_citations"])
        answer = str(task.private["answer"])
        citations = set(state.citations)
        checks = (
            RewardCheck(
                "answer_exact",
                answer.casefold() == (state.final_response or "").strip().casefold(),
                2.0,
            ),
            RewardCheck(
                "required_documents_opened",
                required_opened <= state.opened,
                1.0,
            ),
            RewardCheck(
                "required_documents_cited",
                set(expected_citations) <= citations,
                2.0,
            ),
            RewardCheck("citations_are_opened", citations <= state.opened, 1.0),
            RewardCheck(
                "citations_are_exact",
                state.citations == expected_citations,
                1.0,
            ),
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
        return RewardReport.from_checks(
            checks,
            state_digest=digest,
            binary=True,
        )

    def close(self, state: RetrievalState) -> None:
        state.closed = True
