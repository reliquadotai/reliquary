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
