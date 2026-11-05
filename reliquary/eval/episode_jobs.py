"""What an eval job on a frozen set is declared from, shared by `jobs create --eval-set`
and the admin service's orders: its prompt source, its contract environment, and, for a
set of episode or SWE references, the episode that plays them (checked against the set)."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping

# The turn renderer a signed episode order plays with, by model architecture: only the
# Qwen3.5/3.8 family has one (`renderers:qwen38`).
EPISODE_RENDERER_FAMILIES = {
    "Qwen3_5ForConditionalGeneration": "renderers:qwen38@",
    "Qwen3_5ForCausalLM": "renderers:qwen38@",
}


def set_contract_environment(card: Mapping) -> str:
    """The catalog environment an eval job on this set declares in its contract."""
    interaction = card.get("interaction", "single_turn")
    if interaction == "swe":
        from reliquary.protocol.agentic_source import AGENTIC_SWE_ENVIRONMENT

        return AGENTIC_SWE_ENVIRONMENT   # as for a signed corpus job
    if interaction == "signed_bridged":
        from reliquary.protocol.agentic_source import AGENTIC_SIGNED_ENVIRONMENT

        return AGENTIC_SIGNED_ENVIRONMENT
    from reliquary.protocol.external_eval import contract_environment_for

    return contract_environment_for(card)


def episode_renderer_refusal(architecture: str | None, renderer: str | None) -> str | None:
    """Why a model of ``architecture`` cannot play an episode rendered by ``renderer``, or None."""
    family = EPISODE_RENDERER_FAMILIES.get(str(architecture))
    if family is None:
        return (f"episode_renderer_unavailable: no episode renderer serves {architecture!r} "
                f"(only {sorted(EPISODE_RENDERER_FAMILIES)})")
    if not str(renderer or "").startswith(family):
        return f"episode_renderer_mismatch: {architecture!r} plays {family}*, not {renderer!r}"
    return None


def eval_job_source(*, job_id, eval_set, prompt_count, prompt_start, renderer_id, audit_q,
                    grader_id, threshold, from_profile, episode=None):
    """What an evaluation job served by our own validator is declared from: the
    set's first N problems as its prompt source (checked against the lines'
    sha256), the set's environment as its contract's, a seed from its id.
    Refuses what would make it something else than a measurement.

    A set of episode or SWE tasks is references: the job is the episode job
    (``episode``, from ``--episode-file``) that plays them, at the set's split."""
    from reliquary.eval.prompt_source import eval_source_for, register_eval_prompts
    from reliquary.eval.storage import read_published_set
    from reliquary.validator.corpus_service import CHAT_TEMPLATE_RENDERERS

    card, prompts = read_published_set(eval_set)
    interaction = card.get("interaction", "single_turn")
    if interaction == "single_turn":
        if episode is not None:
            raise ValueError(f"set {eval_set} holds single-turn problems: omit --episode-file")
        if renderer_id not in CHAT_TEMPLATE_RENDERERS:
            raise ValueError(f"an eval set's rows render only through the model's chat template "
                             f"({sorted(CHAT_TEMPLATE_RENDERERS)}), not {renderer_id!r}")
    else:
        refuse_episode_disagreeing_with_set(card, eval_set, episode, renderer_id)
    if prompt_start:
        raise ValueError("an eval job starts at the set's first problem (--prompt-start 0)")
    if audit_q != 1.0:
        raise ValueError("an eval job audits every submission (--audit-q 1.0): an unaudited "
                         "completion would be graded as the model's")
    if grader_id is not None or threshold is not None:
        raise ValueError("an eval job has no filter: every completion is graded, "
                         "right or wrong")
    if from_profile is not None:
        raise ValueError("an eval job's contract is composed from the model and the set's "
                         "environment: omit --from-profile")
    count = int(card["count"]) if prompt_count is None else int(prompt_count)
    if not 1 <= count <= int(card["count"]):
        raise ValueError(f"set {eval_set} holds {card['count']} problems, not {count}")
    source = eval_source_for(eval_set, prompts, count)
    register_eval_prompts(source, prompts)
    seed = int(hashlib.sha256(job_id.encode()).hexdigest()[:15], 16)
    return source.name, count, set_contract_environment(card), seed


def _refuse_bridge_disagreeing_with_set(card, episode, env):
    """A bridged eval env's episode serves the set's package, at the identity, options and
    image lock the set was frozen from."""
    bridge, frozen = episode.get("bridge"), card.get("bridge") or {}
    sandbox = episode.get("sandbox") if isinstance(episode.get("sandbox"), dict) else {}
    if env.get("package") != card["source"]:
        raise ValueError(f"the set's environment is {card['source']!r}, the episode names "
                         f"{env.get('package')!r}")
    if not isinstance(bridge, dict) or {k: bridge.get(k) for k in ("options", "image_lock_sha256")} \
            != {k: frozen.get(k) for k in ("options", "image_lock_sha256")}:
        raise ValueError("the episode's bridge (options, image lock) is not the one the set "
                         "was frozen from")
    if sandbox.get("env_package") != frozen.get("env_package"):
        raise ValueError(f"the set was frozen from {frozen.get('env_package')!r}, the episode "
                         f"pins {sandbox.get('env_package')!r}")


def refuse_episode_disagreeing_with_set(card, eval_set, episode, renderer_id):
    """A set of references is played by an episode job whose environment, split and
    renderer are the set's own: anything else would grade other tasks than the set names."""
    interaction = card["interaction"]
    if interaction not in ("episode", "swe", "signed_bridged"):
        raise ValueError(f"set {eval_set} holds {interaction!r} problems: no eval job plays them")
    if not isinstance(episode, dict):
        raise ValueError(f"set {eval_set} holds {interaction} tasks: pass the episode that plays "
                         "them with --episode-file")
    env = episode.get("env") if isinstance(episode.get("env"), dict) else {}
    if env.get("split") != card["split"]:
        raise ValueError(f"the set's split is {card['split']!r}, the episode names "
                         f"{env.get('split')!r}")
    if interaction in ("swe", "signed_bridged"):
        if episode.get("execution") != "signed_sandbox":
            raise ValueError("an eval job on SWE runs in the signed sandbox: the episode's "
                             f"execution is {episode.get('execution')!r}, not 'signed_sandbox'")
        if interaction == "signed_bridged":
            _refuse_bridge_disagreeing_with_set(card, episode, env)
        return
    if episode.get("harness") != "suite":
        raise ValueError(f"set {eval_set} holds suite episode tasks: the episode's harness is "
                         f"{episode.get('harness')!r}, not 'suite'")
    if env.get("name") != card["source"]:
        raise ValueError(f"the set's environment is {card['source']!r}, the episode names "
                         f"{env.get('name')!r}")
    if not (renderer_id == episode.get("renderer") == card.get("renderer_id")):
        raise ValueError(f"the renderer disagrees: --renderer-id {renderer_id!r}, the episode "
                         f"{episode.get('renderer')!r}, the set {card.get('renderer_id')!r}")


__all__ = ["EPISODE_RENDERER_FAMILIES", "episode_renderer_refusal", "eval_job_source",
           "refuse_episode_disagreeing_with_set", "set_contract_environment"]
