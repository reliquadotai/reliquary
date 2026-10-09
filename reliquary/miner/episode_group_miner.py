"""The episode-group miner (phase 2, plan 2C; spec §4.1).

For one (env, task) of the current window: precommit; play every seed of the task's 2M public pool (the
runner bounds the live sessions; each generate session is bound to its seed's forced draw when its trace
is minted, and its log is taken when the episode ends); withdraw every episode the validator's admission
would refuse (``episode_commit.withdraw_inadmissible``); choose ``group_size`` of the rest
(``choose_episodes``: the hook a miner replaces, any subset of distinct seeds, cherry-picking is intended);
prove them and submit ONE group; withdraw every other kept episode (not waste for a submitted group). A
refused or failed group withdraws them all. Every session ends: submitted, withdrawn, or closed by the
runner.

Production wiring: ``sessions`` = ``HttpRlEpisodes``; ``runner_factory(policy, precommit, prompt=,
open_until=)`` = ``RlSignedEpisodeRunner(...)`` over the forced generate endpoint; ``engine`` = its
``GenerateEngine`` (built with the same ``draws``); ``renderer_for(policy)`` =
``agentic_swe.load_turn_renderer(dir, tools=tuple(policy.tools))``; ``task_prompt(env, task)`` = the env's
task text (the validator's source); ``engine_caps`` = the generate engine's ``max_total_tokens`` /
``max_tokens_per_turn`` and the harness's ``max_model_len``; ``prove`` = ``hf_prover(...)``; ``submit`` =
``lambda request: submitter.submit_batch_v2(url, request, wallet=wallet, randomness=randomness)``."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.protocol.sandbox_session import SessionRefused

logger = logging.getLogger(__name__)

SUBMIT_RETRY_REASONS = frozenset({"worker_dropped"})
"""The only verdict a group is sent again after: the validator's own capacity (refunded, no identity kept).
Any other refusal is final for the prompt this window (spec: never resubmit after one)."""
SUBMIT_ATTEMPTS = 3
SUBMIT_RETRY_S = 5.0
PRECOMMIT_ATTEMPTS = 4
"""The precommit route's 429 / 503 (rate, busy) are waited out (their Retry-After) this many times."""


@dataclass
class EpisodeOutcome:
    seed_index: int
    result: Any                 # agentic_episode.EpisodeResult
    session: Any                # corpus_generate_server.SessionLog, or None


def hf_prover(*, model, wallet, toploc) -> Callable[[dict, str], dict]:
    """``prove(generation, randomness) -> commit`` over the miner's HF proof model."""
    from reliquary.miner.episode_commit import build_signed_episode_commit
    from reliquary.protocol.grail_verifier import GRAILVerifier
    from reliquary.shared.hf_compat import resolve_hidden_size

    verifier = GRAILVerifier(hidden_dim=resolve_hidden_size(model))

    def prove(generation: dict, randomness: str) -> dict:
        return build_signed_episode_commit(
            model=model, verifier=verifier, tokens=generation["tokens"], spans=generation["spans"],
            episode=generation["episode"], service_binding=generation["service_binding"],
            seed_pool=generation["seed_pool"], randomness=randomness, wallet=wallet, toploc=toploc)

    return prove


class EpisodeGroupMiner:
    def __init__(self, *, hotkey: str, sign_binding: Callable[[bytes], str], sessions,
                 runner_factory: Callable[..., Any], engine, draws, prove: Callable[[dict, str], dict],
                 submit: Callable[[Any], Awaitable[Any]], renderer_for: Callable[[Any], Any],
                 task_prompt: Callable[[str, int], str], engine_caps: Mapping[str, int],
                 clock: Callable[[], float] = time.time,
                 sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep) -> None:
        self._hotkey = hotkey
        self._sign = sign_binding
        self._sessions = sessions
        self._runner_factory = runner_factory
        self._engine = engine
        self._draws = draws
        self._prove = prove
        self._submit = submit
        self._renderer_for = renderer_for
        self._task_prompt = task_prompt
        self._engine_caps = dict(engine_caps)
        self._clock = clock
        self._sleep = sleep
        self._renderers: dict[Any, Any] = {}

    def choose_episodes(self, outcomes: list[EpisodeOutcome], *, group_size: int) -> list[EpisodeOutcome]:
        """HOOK: which ``group_size`` admissible episodes form the group. The reference keeps the lowest
        seeds; a real miner over-generates and keeps the subset it prefers. Return ``group_size`` outcomes of
        distinct seeds, taken from ``outcomes``."""
        return sorted(outcomes, key=lambda outcome: outcome.seed_index)[:group_size]

    def _renderer(self, policy):
        if policy not in self._renderers:
            from reliquary.miner.rl_episode_client import check_engine_caps

            # Once per episode policy: an engine whose caps are not the contract's plays turns the
            # validator refuses (or cuts episodes it reads as early stops).
            check_engine_caps(policy, **self._engine_caps)
            self._renderers[policy] = self._renderer_for(policy)
        return self._renderers[policy]

    async def _precommit(self, precommit, until: float | None) -> None:
        from reliquary.miner.rl_episode_client import signed_precommit_body
        from reliquary.sandbox.rl_routes import episode_precommit_path

        path = episode_precommit_path(self._sessions.prefix)
        for attempt in range(1, PRECOMMIT_ATTEMPTS + 1):
            body = signed_precommit_body(precommit=precommit, sign_binding=self._sign, now=self._clock(),
                                         validator_hotkey=self._sessions.validator_hotkey, path=path)
            try:
                answer = await asyncio.to_thread(self._sessions.precommit, body)
            except SessionRefused as refused:
                if (refused.reason == "precommit_exists"
                        and refused.detail.get("precommit_sha256") == precommit.sha256):
                    return
                wait = float(refused.retry_after or 0.0)
                if (refused.status not in (429, 503) or attempt == PRECOMMIT_ATTEMPTS
                        or (until is not None and self._clock() + wait >= until)):
                    raise
                logger.info("precommit %s throttled (%s): sent again in %.0f s", precommit.sha256[:12],
                            refused.reason, wait)
                await self._sleep(wait)
                continue
            if not isinstance(answer, Mapping) or answer.get("precommit_sha256") != precommit.sha256:
                raise ValueError("the validator recorded another precommit")
            return

    async def _play(self, runner, precommit, pool, played: list[EpisodeOutcome]) -> None:
        """Every seed of the pool, at once (the runner bounds its live sessions). Each outcome is appended
        to ``played`` as soon as its episode ends, so the caller can withdraw it whatever happens next."""
        from reliquary.miner.forced_draw import DrawBinding

        deadline = getattr(runner, "deadline", None)

        async def one(seed: int) -> None:
            bound: list[str] = []
            taken: str | None = None

            def on_session(session_id: str) -> None:
                # Before the trace's first generate request: the forced engine refuses an unbound session.
                self._draws.bind(session_id, DrawBinding(pool, seed))
                bound.append(session_id)

            try:
                try:
                    async with asyncio.timeout(deadline(seed) if deadline is not None else None):
                        result = await runner.run(seed, on_session=on_session)
                except SessionRefused as refused:
                    logger.info("seed %d of precommit %s: no session (%s)", seed, precommit.sha256[:12],
                                refused.reason)
                    return
                except TimeoutError:
                    logger.warning("seed %d of precommit %s passed its deadline", seed, precommit.sha256[:12])
                    return
                except Exception:
                    logger.exception("seed %d of precommit %s crashed", seed, precommit.sha256[:12])
                    return
                # The log is taken only now, at the episode's end (taking it also ends its binding).
                taken = result.session_id
                try:
                    session = self._engine.take_session(taken) if taken else None
                except Exception:
                    logger.exception("seed %d: taking its generate session failed", seed)
                    session = None
                played.append(EpisodeOutcome(seed, result, session))
            finally:
                for session_id in dict.fromkeys(bound):
                    self._draws.drop(session_id)
                    if session_id != taken:
                        drop = getattr(self._engine, "drop_session", None)
                        if drop is not None:
                            drop(session_id)

        await asyncio.gather(*(one(seed) for seed in range(pool.pool_seeds)))

    @staticmethod
    async def _release(outcomes) -> None:
        for outcome in outcomes:
            release = getattr(outcome.result, "release", None)
            if release is None:
                continue
            try:
                await release()
            except Exception:
                logger.exception("withdrawing the episode of seed %d failed", outcome.seed_index)

    async def mine_task(self, *, announcement: dict, randomness: str, window: int, environment: str,
                        task_index: int, open_until: float | None = None):
        """One group of (``environment``, ``task_index``) in ``window``: the validator's verdict, or None when
        no group was sent. ``open_until`` is the window's end (wall clock): no open or precommit is sent
        again past it."""
        from reliquary.miner.episode_commit import withdraw_inadmissible
        from reliquary.protocol.seed_pool import pool_from_service_policy
        from reliquary.protocol.service_contract import ServiceContract
        from reliquary.protocol.service_episode import EpisodePrecommit

        contract = ServiceContract.from_dict(announcement["contract"])
        policy = contract.episode_policy(environment)
        if policy is None:
            raise ValueError(f"{environment} is not an episode environment of this order")
        renderer = self._renderer(policy)
        prompt = self._task_prompt(environment, int(task_index))
        checkpoint = announcement["checkpoint"]["revision"]
        pool = pool_from_service_policy(announcement, environment=environment, prompt_idx=int(task_index),
                                        checkpoint_hash=checkpoint)
        if pool is None:
            raise ValueError(f"{environment} announces no public seed pool")
        precommit = EpisodePrecommit(order=contract.sha256, window=int(window), environment=environment,
                                     task_index=int(task_index), checkpoint=checkpoint, pool_sha256=pool.sha256,
                                     hotkey=self._hotkey)
        await self._precommit(precommit, open_until)
        runner = self._runner_factory(policy, precommit, prompt=prompt, open_until=open_until)
        played: list[EpisodeOutcome] = []
        held: list[EpisodeOutcome] = []        # kept episodes not yet submitted nor withdrawn
        async with runner:
            try:
                await self._play(runner, precommit, pool, played)
                kept = await withdraw_inadmissible(list(played), policy=policy, renderer=renderer, prompt=prompt)
                played.clear()
                held = [outcome for outcome, _ in kept]
                admissible = {id(outcome): value for outcome, value in kept}
                if len(held) < pool.group_size:
                    logger.info("precommit %s: %d admissible episodes, %d needed; nothing submitted",
                                precommit.sha256[:12], len(held), pool.group_size)
                    return None
                chosen = self._chosen(held, pool.group_size)
                response = await self._send(contract, policy, pool, precommit, chosen, admissible,
                                            randomness=randomness, environment=environment,
                                            task_index=int(task_index), window=int(window), checkpoint=checkpoint)
                if response is not None and getattr(response, "accepted", False):
                    for outcome in chosen:
                        if outcome.result.submitted is not None:
                            outcome.result.submitted()
                    held = [outcome for outcome in held if all(outcome is not c for c in chosen)]
                return response
            finally:
                # Whatever ended the task (a refusal, an error, a cancellation): every episode not submitted
                # is withdrawn, so its session and the hotkey's caps are freed now.
                await self._release(played + held)

    def _chosen(self, held: list[EpisodeOutcome], group_size: int) -> list[EpisodeOutcome]:
        chosen = sorted(self.choose_episodes(list(held), group_size=group_size),
                        key=lambda outcome: outcome.seed_index)
        seeds = [outcome.seed_index for outcome in chosen]
        if (len(chosen) != group_size or len(set(seeds)) != len(seeds)
                or any(all(outcome is not h for h in held) for outcome in chosen)):
            raise ValueError("choose_episodes must return group_size admissible episodes of distinct seeds")
        return chosen

    async def _send(self, contract, policy, pool, precommit, chosen, admissible, *, randomness: str,
                    environment: str, task_index: int, window: int, checkpoint: str):
        from reliquary.constants import ACTIVE_PROTOCOL_PROFILE, FORCED_SEED_PROTOCOL_VERSION
        from reliquary.miner.engine import _compute_merkle_root
        from reliquary.miner.episode_commit import episode_metadata
        from reliquary.protocol.service_submission import ServiceBinding
        from reliquary.protocol.submission import BatchSubmissionRequest, RolloutSubmission
        from reliquary.services.scoring import classify_signal

        selection = pool.selection([outcome.seed_index for outcome in chosen])
        rewards = [float(outcome.result.reward) for outcome in chosen]
        signal = classify_signal(rewards, expected=pool.group_size,
                                 sigma_min_bps=contract.to_dict()["scoring"]["sigma_min_bps"])
        # As the single-turn miner: a uniform group of an exploration env is sent as exploration.
        purpose = ("exploration" if contract.environment(environment)["exploration"] == 1
                   and signal.category in ("uniform-low", "uniform-high", "uniform-intermediate")
                   else "training")
        binding = ServiceBinding(contract.sha256, purpose)
        rollouts = []
        for index, outcome in enumerate(chosen):
            value = admissible[id(outcome)]
            episode = episode_metadata(precommit_sha256=precommit.sha256, seed_index=outcome.seed_index,
                                       spans=value.spans, stop=value.stop, transcript=outcome.result.transcript)
            generation = {"tokens": list(value.tokens), "spans": list(value.spans), "episode": episode,
                          "service_binding": binding.rollout_binding(index),
                          "seed_pool": selection.rollout_binding(index)}
            commit = await asyncio.to_thread(self._prove, generation, randomness)
            rollouts.append(RolloutSubmission(tokens=list(value.tokens), reward=rewards[index], commit=commit,
                                              env_name=environment))
        request = BatchSubmissionRequest(
            miner_hotkey=self._hotkey, prompt_idx=task_index, window_start=window,
            merkle_root=_compute_merkle_root(rollouts), rollouts=rollouts, checkpoint_hash=checkpoint,
            protocol_version=FORCED_SEED_PROTOCOL_VERSION,
            generation_profile_id=(ACTIVE_PROTOCOL_PROFILE.profile_id
                                   if ACTIVE_PROTOCOL_PROFILE.protocol_version >= 3 else ""),
            pool_selection=selection.to_dict(), service_binding=binding.to_dict())
        submit_by = min((float(o.result.submit_by) for o in chosen if o.result.submit_by is not None),
                        default=None)
        response = None
        for attempt in range(1, SUBMIT_ATTEMPTS + 1):
            if submit_by is not None and self._clock() > submit_by:
                logger.warning("precommit %s: past the validator's grading deadline, nothing submitted",
                               precommit.sha256[:12])
                return response
            response = await self._submit(request)
            reason = getattr(getattr(response, "reason", None), "value", getattr(response, "reason", None))
            if getattr(response, "accepted", False) or reason not in SUBMIT_RETRY_REASONS:
                break
            if attempt < SUBMIT_ATTEMPTS:
                logger.info("precommit %s: group refused %s (validator busy), sent again", precommit.sha256[:12],
                            reason)
                await self._sleep(SUBMIT_RETRY_S * attempt)
        logger.info("precommit %s: group %s %s", precommit.sha256[:12],
                    "accepted" if getattr(response, "accepted", False) else "refused",
                    getattr(response, "reason", None))
        return response


__all__ = ["EpisodeGroupMiner", "EpisodeOutcome", "hf_prover"]
