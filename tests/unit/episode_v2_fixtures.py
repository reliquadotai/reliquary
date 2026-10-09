# tests/unit/episode_v2_fixtures.py
"""Builders for signed-episode environments in the v2 service RL path (phase 2, plan 2C)."""
from __future__ import annotations

import time

from reliquary.protocol.service_contract import ServiceContract
from reliquary.services.runtime import ServiceRuntime, protocol_slot_geometry
from tests.unit.service_v2_fixtures import MATH, contract_v2_dict, qualification_v2

EPISODE = "reliquary_test_episode_v1"
SANDBOX_ENV = "reliquary-swe"            # the fake gateway's env name (test_signed_episode_e2e)
SPLIT = "train:rl"                       # accepted by reliquary_sandbox_service.episodes.testing
ENV_PACKAGE = "reliquary-swe==0.1.0a1+g0123456789abcdef"   # test_corpus_job_signed_sandbox.ENV_PACKAGE
REVISION = "d" * 40                      # contract_v2_dict's checkpoint revision
WINDOW_BEACON = "ab" * 32
POOL = 0.25
TASK = 3
GIB = 1024**3
BUDGETS = {"max_calls": 64, "per_call_timeout_s": 600, "cpu_s": 3600, "wall_s": 3600,
           "memory_bytes": 4 * GIB, "pids": 1024, "disk_bytes": 10 * GIB}


def episode_block(**overrides) -> dict:
    value = {"kind": "signed-sandbox-episode/v1", "sandbox_env": SANDBOX_ENV, "split": SPLIT,
             "env_package": ENV_PACKAGE, "tools": ["bash", "edit"], "max_turns": 8,
             "max_tokens_per_turn": 512, "max_episode_tokens": 4096, "budgets": dict(BUDGETS)}
    value.update(overrides)
    return value


def episode_contract_dict(*, episode=None, **kwargs) -> dict:
    kwargs.setdefault("envs", (MATH, EPISODE))
    value = contract_v2_dict(**kwargs)
    value["environments"][EPISODE]["episode"] = episode_block() if episode is None else episode
    return value


def episode_contract(**kwargs) -> ServiceContract:
    return ServiceContract.from_dict(episode_contract_dict(**kwargs))


def episode_runtime(tmp_path, contract=None, *, window=1) -> ServiceRuntime:
    """A real runtime with ``window`` frozen and announced on the episode order."""
    contract = contract or episode_contract()
    tmp_path.mkdir(parents=True, exist_ok=True)
    rt = ServiceRuntime(tmp_path / "runtime.sqlite3", contract, qualification_v2(contract),
                        now=time.time() - 10, drand_round_at=lambda instant: 1_000)
    rt.ensure_checkpoint(checkpoint_n=0, repo="models/test", revision=REVISION)
    picks, slots = protocol_slot_geometry()
    rt.open_window(window, pools={name: POOL for name in contract.environments}, picks_target=picks,
                   batch_slots=slots, now=time.time() - 5)
    rt.announcement(window=window, randomness=WINDOW_BEACON)
    return rt


from reliquary.protocol.seed_pool import SeedPool  # noqa: E402


def episode_pool(contract, *, task=TASK, window=1, randomness=WINDOW_BEACON, checkpoint=REVISION) -> SeedPool:
    """The public pool of (EPISODE, task, window), exactly as the runtime announces it."""
    return SeedPool.from_contract(contract, environment=EPISODE, prompt_idx=task, checkpoint_hash=checkpoint,
                                  pool_epoch=window, randomness=randomness)


def episode_precommit(contract, *, hotkey, task=TASK, window=1):
    from reliquary.protocol.service_episode import EpisodePrecommit

    return EpisodePrecommit(order=contract.sha256, window=window, environment=EPISODE, task_index=task,
                            checkpoint=REVISION,
                            pool_sha256=episode_pool(contract, task=task, window=window).sha256, hotkey=hotkey)


def signed_episode_metadata(*, precommit_sha256, seed_index, spans, transcript, stop="agent_completed") -> dict:
    from reliquary.protocol.submission import SIGNED_EPISODE_SCHEMA

    return {"schema_version": SIGNED_EPISODE_SCHEMA, "precommit_sha256": precommit_sha256,
            "seed_index": seed_index, "assistant_spans": [list(span) for span in spans], "stop": stop,
            "transcript": transcript}


def signed_episode_commit(*, tokens, spans, episode, selection, index, contract, purpose="training",
                          chunk_tokens=32, signature="aa", randomness="cd" * 32) -> dict:
    """A commit dict shaped like the miner's (stub proofs: one per span chunk, never checked here)."""
    from reliquary.protocol.service_submission import ServiceBinding
    from reliquary.protocol.toploc import span_chunk_count

    model_tokens = sum(end - start for start, end in spans)
    proofs = ["AAAA"] * sum(span_chunk_count(end - start, chunk_tokens) for start, end in spans)
    return {"tokens": list(tokens), "commitments": [{} for _ in tokens],
            "proof_version": "public-group-proof/v1", "model": {"name": "model", "layer_index": -1},
            "signature": signature, "beacon": {"randomness": randomness},
            "rollout": {"prompt_length": spans[0][0], "completion_length": len(tokens) - spans[0][0],
                        "success": False, "total_reward": 0.0, "advantage": 0.0,
                        "token_logprobs": [-1.0] * model_tokens, "episode": episode,
                        "seed_pool": selection.rollout_binding(index),
                        "service_binding": ServiceBinding(contract.sha256, purpose).rollout_binding(index)},
            "toploc_proofs": proofs}


PROMPT_TEXT = "Write 42 to /work/answer.txt."


class FixedSource:
    """An episode task source with one prompt for every task (plan 2A provides the real ones)."""

    def __init__(self, text: str = PROMPT_TEXT, rows: int = 1000, image: str | None = None) -> None:
        self.text, self.rows, self.image = text, rows, image

    def __len__(self) -> int:
        return self.rows

    def prompt(self, index: int) -> str:
        return self.text

    async def resolve(self, index: int):
        from reliquary_sandbox_service.episodes.testing import FAKE_IMAGE

        from reliquary.sandbox.tasks import ResolvedTask

        return ResolvedTask(self.image or FAKE_IMAGE, {})


def make_test_episode_env():
    from reliquary.environment.signed_episode import SignedEpisodeEnvironment

    return SignedEpisodeEnvironment(EPISODE, FixedSource())


def episode_spec():
    from reliquary.environment.registry import EnvironmentSpec

    return EnvironmentSpec(
        name=EPISODE, factory_path="tests.unit.episode_v2_fixtures:make_test_episode_env",
        scorer_path="reliquary.environment.signed_episode:signed_episode_scorer",
        validator_authoritative_reward=True, admission_resource_class="sandbox",
        termination_policy="eos_or_cap", final_answer_policy="text",
        reward_lattice_policy="final-record/v1", attainable_rewards=(),
        contract_version="reliquary/signed-episode/v1", interaction_mode="signed_episode")


def register_episode_env(monkeypatch) -> None:
    from types import MappingProxyType

    from reliquary.environment import registry

    monkeypatch.setattr(registry, "ENVIRONMENT_SPECS",
                        MappingProxyType({**registry.ENVIRONMENT_SPECS, EPISODE: episode_spec()}))


def episode_signers(tmp_path):
    """(validator, machine) Ed25519 signers, as the sandbox tests build them."""
    from tests.unit.sandbox_fixtures import signer

    tmp_path.mkdir(parents=True, exist_ok=True)
    return signer(tmp_path, "v", "v1"), signer(tmp_path, "m", "k1")


def play_episode(*, validator, machine, precommit, seed, session_id, reward, calls=1, output="ok",
                 last=None, issued_at=None):
    """One honest two-turn episode in the fake renderer's ids (``tests.unit.test_trajectory_parse``):
    turn 1 makes ``calls`` bash calls, each answered by one signed call record; turn 2 (``last``,
    default nine text tokens and the terminator) ends it. Returns ``(tokens, spans, transcript)``:
    tokens with the prompt, absolute spans, the gateway-signed transcript (graded ``reward``)."""
    from reliquary_sandbox.observation import render_observation

    from reliquary.corpus.signed_parse import signed_records
    from reliquary.protocol.service_episode import rl_engagement
    from tests.unit.sandbox_fixtures import NOW, claims, transcript
    from tests.unit.test_trajectory_parse import CALL, TERM, TEXT, FakeRenderer

    renderer = FakeRenderer()
    issued_at = NOW if issued_at is None else issued_at
    prompt = renderer.initial_ids(PROMPT_TEXT)
    first = [TEXT] * 9 + [CALL] * calls + [TERM]
    session = claims(session_id=session_id, hotkey=precommit.hotkey,
                     engagement=rl_engagement(precommit.window, precommit.sha256, seed), split=SPLIT,
                     index=precommit.task_index, checkpoint=precommit.checkpoint, issued_at=issued_at,
                     expires_at=issued_at + 4500)
    signed = transcript(validator, machine, session, status="graded", reward=float(reward),
                        env_package=ENV_PACKAGE,
                        calls=[{"turn": 0, "k": k, "arguments": {"command": f"c{k}"}, "output": output}
                               for k in range(calls)])
    observations = [render_observation(body.to_dict()) for body in signed_records(signed).calls]
    full = renderer.next_prompt(prompt, first, observations)
    tokens = full + (list(last) if last is not None else [TEXT] * 9 + [TERM])
    spans = [(len(prompt), len(prompt) + len(first)), (len(full), len(tokens))]
    return tokens, spans, signed


def half_rewards() -> list[float]:
    from reliquary.constants import M_ROLLOUTS

    return [1.0] * (M_ROLLOUTS // 2) + [0.0] * (M_ROLLOUTS - M_ROLLOUTS // 2)


def episode_group(contract, *, validator, machine, hotkey="5Hot", rewards=None, seeds=None, task=TASK,
                  window=1, calls=1, last=None, stop="agent_completed"):
    """M honest episodes of one precommit, one per chosen seed (rollout i plays seed i of the selection)."""
    from types import SimpleNamespace

    from reliquary.constants import M_ROLLOUTS
    from reliquary.protocol.service_submission import ServiceBinding

    pool = episode_pool(contract, task=task, window=window)
    precommit = episode_precommit(contract, hotkey=hotkey, task=task, window=window)
    selection = pool.selection(list(range(M_ROLLOUTS)) if seeds is None else list(seeds))
    rewards = half_rewards() if rewards is None else list(rewards)
    rollouts = []
    for index, seed in enumerate(selection.seeds):
        tokens, spans, signed = play_episode(validator=validator, machine=machine, precommit=precommit, seed=seed,
                                             session_id=f"s-{seed}", reward=rewards[index], calls=calls, last=last)
        episode = signed_episode_metadata(precommit_sha256=precommit.sha256, seed_index=seed, spans=spans,
                                          transcript=signed, stop=stop)
        commit = signed_episode_commit(tokens=tokens, spans=spans, episode=episode, selection=selection,
                                       index=index, contract=contract)
        rollouts.append(SimpleNamespace(tokens=tokens, reward=0.0, commit=commit, env_name=EPISODE))
    request = SimpleNamespace(
        miner_hotkey=hotkey, prompt_idx=task, window_start=window, checkpoint_hash=REVISION,
        pool_selection=selection.to_dict(), rollouts=rollouts,
        service_binding=ServiceBinding(contract.sha256, "training").to_dict())
    return SimpleNamespace(request=request, precommit=precommit, selection=selection, pool=pool, rewards=rewards)


def batch_request(group):
    """The group as a real ``BatchSubmissionRequest`` (its rollouts take private attributes)."""
    from reliquary.protocol.submission import BatchSubmissionRequest, RolloutSubmission

    request = group.request
    return BatchSubmissionRequest(
        miner_hotkey=request.miner_hotkey, prompt_idx=request.prompt_idx, window_start=request.window_start,
        merkle_root="00" * 32, checkpoint_hash=request.checkpoint_hash, pool_selection=request.pool_selection,
        service_binding=request.service_binding,
        rollouts=[RolloutSubmission(tokens=r.tokens, reward=r.reward, commit=r.commit, env_name=r.env_name)
                  for r in request.rollouts])
