"""Single-turn admission helpers shared by the async RL submission path and the environment scorers:
termination classification, the trusted problem materials, and the registry's scoring adapters."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from reliquary.constants import (
    BFT_ANSWER_BUDGET,
    BFT_THINKING_BUDGET,
    GRADER_EVAL_TIMEOUT_SECONDS,
    max_new_tokens_for_environment,
)
from reliquary.environment.grader_client import GraderClient
from reliquary.environment.opencodeinstruct import _entry_function_name, _extract_python
from reliquary.environment.openmathinstruct import _compute_omi_reward
from reliquary.environment.registry import get_environment_spec


@dataclass(frozen=True)
class AdmissionContext:
    randomness: str
    environment: str
    vocab_size: int | None
    max_sequence_length: int
    eos_token_ids: tuple[int, ...]
    canonical_force_ids: tuple[int, ...]
    think_close_ids: tuple[int, ...]
    bootstrap: bool
    enforce_envelope_signature: bool
    enforce_legacy_merkle: bool
    service_policy: dict | None = None
    # This env's groups are signed episodes (the parent process verifies their transcripts) and
    # the contract's per-env maximum episode length. The defaults leave every other context unchanged.
    signed_episode: bool = False
    episode_max_tokens: int | None = None



@dataclass(frozen=True)
class AdmissionProblemMaterials:
    problem: dict[str, Any]
    rendered_prompt: str
    code_cases: list[dict[str, Any]] | None = None
    reward_materials: Any = None

    @property
    def effective_reward_materials(self) -> Any:
        return (
            self.reward_materials
            if self.reward_materials is not None
            else self.code_cases
        )


_ADMISSION_DEADLINE = ContextVar("admission_deadline", default=None)



def _force_span_valid(
    tokens: list[int],
    meta: dict[str, Any],
    context: AdmissionContext,
) -> bool:
    if not meta.get("forced"):
        return True
    # Mirror of verifier.validate_force_span: without BFT there is no
    # profile-sanctioned force span, so a forced claim is tampering.
    from reliquary.constants import BFT_ENABLED

    if not BFT_ENABLED:
        return False
    span = meta.get("force_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    try:
        start, end = int(span[0]), int(span[1])
        prompt_length = int(meta.get("prompt_length", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if not (prompt_length <= start < end <= len(tokens)):
        return False
    if start - prompt_length != BFT_THINKING_BUDGET:
        return False
    think_close = set(context.think_close_ids)
    if any(int(token) in think_close for token in tokens[prompt_length:start]):
        return False
    return list(tokens[start:end]) == list(context.canonical_force_ids)



def _forced_cap_termination(meta: dict[str, Any]) -> bool:
    if not meta.get("forced"):
        return False
    span = meta.get("force_span")
    if not isinstance(span, (list, tuple)) or len(span) != 2:
        return False
    try:
        start, end = int(span[0]), int(span[1])
        completion_length = int(meta.get("completion_length", 0))
    except (TypeError, ValueError, OverflowError):
        return False
    if end <= start:
        return False
    return completion_length == (
        BFT_THINKING_BUDGET + (end - start) + BFT_ANSWER_BUDGET
    )



def _natural_cap_termination(
    tokens: list[int],
    meta: dict[str, Any],
    context: AdmissionContext,
) -> bool:
    try:
        uses_math_bft = (
            get_environment_spec(context.environment).termination_policy
            == "math_bft"
        )
    except ValueError:
        uses_math_bft = False
    if not uses_math_bft:
        return False
    if meta.get("forced") or meta.get("force_span") not in (None, []):
        return False
    try:
        prompt_length = int(meta.get("prompt_length", 0))
        completion_length = int(meta.get("completion_length", 0))
    except (TypeError, ValueError):
        return False
    if completion_length != BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET:
        return False
    if prompt_length + completion_length != len(tokens):
        return False
    phase_one = tokens[
        prompt_length: prompt_length + BFT_THINKING_BUDGET
    ]
    think_close = set(context.think_close_ids)
    return any(int(token) in think_close for token in phase_one)



def _classify_termination(rollout, context: AdmissionContext) -> str:
    """Classify one rollout's termination.

    Returns ``"ok"`` (terminated, or an accepted BFT/natural cap shape),
    ``"truncated"`` (no EOS, ran to the protocol cap), or a reject reason name
    (``"bad_schema"`` / ``"tampered"`` / ``"bad_termination"``). Shared by the
    admission gate and the truncated-rollout count so both read the same
    predicate.
    """
    eos_ids = set(context.eos_token_ids)
    commit = rollout.commit
    tokens = list(commit.get("tokens") or [])
    meta = commit.get("rollout", {}) or {}
    # Episode termination is an environment state transition, not a trailing
    # tokenizer EOS. The authoritative replay gate below validates it.
    if isinstance(meta.get("episode"), dict):
        return "ok"
    try:
        prompt_length = int(meta.get("prompt_length", 0))
        completion_length = int(meta.get("completion_length", 0))
    except (TypeError, ValueError, OverflowError):
        return "bad_schema"
    completion = tokens[prompt_length: prompt_length + completion_length]
    if not completion:
        return "bad_schema"
    eos_positions = [
        index
        for index, token in enumerate(completion)
        if int(token) in eos_ids
    ]
    if eos_positions:
        if len(eos_positions) > 1 or eos_positions[0] != len(completion) - 1:
            return "bad_termination"
        return "ok"
    try:
        uses_math_bft = (
            get_environment_spec(context.environment).termination_policy
            == "math_bft"
        )
    except ValueError:
        uses_math_bft = False
    if uses_math_bft and _forced_cap_termination(meta):
        if not _force_span_valid(tokens, meta, context):
            return "tampered"
        return "ok"
    if _natural_cap_termination(tokens, meta, context):
        return "ok"
    if (
        prompt_length + completion_length
        < max_new_tokens_for_environment(context.environment)
    ):
        return "bad_termination"
    return "truncated"



def _compute_code_rewards(
    completion_texts: list[str],
    cases: list[dict[str, Any]],
) -> list[float]:
    client = GraderClient()
    client.deadline_monotonic = _ADMISSION_DEADLINE.get()
    # Must grade the same span as OpenCodeInstructEnvironment.compute_reward:
    # a divergence here rejects honest miners on reward_mismatch.
    entry_name = _entry_function_name(cases)

    def _grade(text: str) -> float:
        return float(
            client.evaluate_cases(
                _extract_python(text, entry_name=entry_name),
                cases,
                timeout_s=GRADER_EVAL_TIMEOUT_SECONDS,
            )
        )

    executor = ThreadPoolExecutor(max_workers=len(completion_texts))
    try:
        return list(executor.map(_grade, completion_texts))
    finally:
        # A process deadline must not block in context-manager shutdown while
        # a bounded grader call is still unwinding.
        executor.shutdown(wait=False, cancel_futures=True)



def _score_openmath_adapter(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    del reward_materials
    return [
        float(_compute_omi_reward(problem, text))
        for text in completion_texts
    ]



def _score_opencode_adapter(
    problem: dict[str, Any],
    completion_texts: list[str],
    reward_materials: Any = None,
) -> list[float]:
    del problem
    return _compute_code_rewards(
        completion_texts,
        list(reward_materials or []),
    )


def render_canonical_prompt(
    tokenizer: Any, prompt: str, *, thinking: bool = True
) -> str:
    """Render a trusted user prompt exactly as admission will tokenize it."""
    # Mirror of encode_prompt's guard. Both must switch on the same constant:
    # this render feeds prompt_content_sha256 while encode_prompt feeds the
    # generated tokens, and a disagreement between them is dropped silently at
    # seal rather than rejected.
    from reliquary.constants import RAW_COMPLETION_PROMPTS

    if RAW_COMPLETION_PROMPTS:
        return str(prompt)

    rendered = str(prompt)
    chat_template = getattr(tokenizer, "chat_template", None)
    if (
        isinstance(chat_template, str)
        and chat_template
        and hasattr(tokenizer, "apply_chat_template")
    ):
        kwargs: dict[str, Any] = {
            "add_generation_prompt": True,
            "tokenize": False,
        }
        if "enable_thinking" in chat_template:
            kwargs["enable_thinking"] = bool(thinking)
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            **kwargs,
        )
    return rendered


def _environment_policy(env: Any, field: str, default: Any = None) -> Any:
    """Resolve registered policy while retaining local test-env embedding."""

    try:
        return getattr(
            get_environment_spec(str(getattr(env, "name", ""))),
            field,
        )
    except (AttributeError, ValueError):
        return getattr(env, field, default)


def _render_environment_prompt(env: Any, tokenizer: Any, prompt_idx: int) -> str:
    mode = _environment_policy(env, "interaction_mode", "single_turn")
    if mode == "signed_episode":
        # The task source's prompt verbatim (exactly what the harness renders, which the
        # admission compared token for token with the trajectory's prompt); never a chat template here.
        return str(env.get_problem(prompt_idx)["prompt"])
    if mode == "episode":
        from reliquary.environment.agentic.compat import rendered_episode_prompt

        return rendered_episode_prompt(env, prompt_idx)
    problem = env.get_problem(prompt_idx)
    from reliquary.constants import thinking_for_environment

    return render_canonical_prompt(
        tokenizer,
        str(problem["prompt"]),
        thinking=thinking_for_environment(str(getattr(env, "name", ""))),
    )


def admission_problem_materials(env: Any, tokenizer: Any, prompt_idx: int):
    """The trusted problem data admission grades against: the problem, its rendered prompt and its reward
    materials, for the async submission path (``rl_submit``)."""
    problem = env.get_problem(prompt_idx)
    rendered_prompt = _render_environment_prompt(env, tokenizer, prompt_idx)
    code_cases = None
    reward_materials = None
    try:
        from reliquary.environment.registry import get_environment_spec

        spec = get_environment_spec(getattr(env, "name", ""))
    except ValueError:
        spec = None
    materializer_name = (
        spec.reward_materializer_method if spec is not None else None
    )
    cases_loader = (
        getattr(env, materializer_name, None)
        if materializer_name
        else None
    )
    if callable(cases_loader):
        reward_materials = cases_loader(problem)
        # `code_cases` is the older, narrower channel; what fills it is the
        # environment asking for the sandbox, not its name. The name test
        # this replaces would never have matched a packaged code
        # environment, which is the same environment under another name.
        if spec is not None and spec.admission_resource_class == "sandbox":
            code_cases = list(reward_materials)
    return AdmissionProblemMaterials(
        problem=dict(problem),
        rendered_prompt=rendered_prompt,
        code_cases=code_cases,
        reward_materials=reward_materials,
    )
