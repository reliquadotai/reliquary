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
``lambda request: submitter.submit_batch_v2(url, request, wallet=wallet, randomness=randomness)``;
``verdicts`` = ``http_verdicts(url, hotkey, client=client)``.

``/submit`` answers a queue receipt (``SUBMITTED``), not the verdict: the kept episodes stay held while the
group's verdict is polled (``GET /miner-verdicts/{hotkey}/{window}/{merkle_root}``), and only then are they
submitted (admitted) or withdrawn, or the group is sent again (``protocol.episode_retry``)."""
from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

from reliquary.protocol.episode_retry import group_resendable
from reliquary.protocol.sandbox_session import SessionRefused

logger = logging.getLogger(__name__)

SUBMIT_ATTEMPTS = 8
"""Sends of one group at most (each also bounded by its grading deadline and the window's end)."""
SUBMIT_RETRY_S = 5.0
"""Back-off before a group is sent again: ``SUBMIT_RETRY_S`` x attempt (a RATE_LIMITED resend spends quota)."""
PRECOMMIT_ATTEMPTS = 4
"""Without a window end, the precommit route's 429 / 503 are waited out this many times; with one, until it."""
PRECOMMIT_MIN_BACKOFF_S = 2.0
"""The least wait before a throttled precommit is sent again (a 503 may name no Retry-After)."""
VERDICT_POLL_S = 2.0
VERDICT_POLL_MAX_S = 15.0
"""The verdict is polled every VERDICT_POLL_S at first, then 1.5 x longer each time, up to VERDICT_POLL_MAX_S."""
VERDICT_WAIT_S = 600.0
"""How long a verdict is polled for when the group names no grading deadline and the window no end."""
PROOF_BUDGET_S = 120.0
"""Seeds still playing are cut this long before the earliest grading deadline (``submit_by``) of an
episode already played: the group must still be proved and sent."""
ABORT_REPLAYS = 1
"""A seed whose machine-signed final is ``aborted`` is played again this many times (the validator frees
its engagement once for that)."""


@dataclass(frozen=True)
class GroupVerdict:
    """What became of a sent group: ``accepted`` (admitted into the pool), else the refusal's reason and
    stage; ``verdict`` is the validator's verdict record when one was polled."""
    accepted: bool
    reason: str | None
    stage: str | None = None
    verdict: Mapping | None = None


def _reason(value) -> str | None:
    return getattr(value, "value", value)


def http_verdicts(url: str, hotkey: str, *, client, timeout: float = 10.0):
    """``verdicts(window, merkle_root)`` over ``GET /miner-verdicts/{hotkey}/{window}/{merkle_root}``: the
    route's JSON, or None when it does not answer 200."""
    from urllib.parse import quote

    async def fetch(window: int, merkle_root: str):
        response = await client.get(f"{url}/miner-verdicts/{quote(hotkey, safe='')}/{int(window)}/{merkle_root}",
                                    timeout=timeout)
        return response.json() if response.status_code == 200 else None

    return fetch


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
                 verdicts: Callable[[int, str], Awaitable[Any]], proof_budget_s: float = PROOF_BUDGET_S,
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
        self._verdicts = verdicts
        self._proof_budget_s = float(proof_budget_s)
        self._clock = clock
        self._sleep = sleep
        self._renderers: dict[Any, Any] = {}

    def choose_episodes(self, outcomes: list[EpisodeOutcome], *, group_size: int,
                        in_zone: Callable[[list[EpisodeOutcome]], bool] | None = None) -> list[EpisodeOutcome]:
        """HOOK: which ``group_size`` admissible episodes form the group (any subset of distinct seeds,
        cherry-picking is intended). ``in_zone(subset)`` says whether the validator would classify that
        subset in-zone (the order's sigma threshold). The reference keeps the lowest seeds when they are
        in-zone, else the in-zone subset mixing the lowest and the highest rewards closest to half and
        half, else the lowest seeds. Return ``group_size`` outcomes of distinct seeds, taken from
        ``outcomes``."""
        lowest = sorted(outcomes, key=lambda outcome: outcome.seed_index)[:group_size]
        if in_zone is None or in_zone(lowest):
            return lowest
        ranked = sorted(outcomes, key=lambda outcome: (float(outcome.result.reward), outcome.seed_index))
        for low in sorted(range(1, group_size), key=lambda count: (abs(2 * count - group_size), count)):
            mixed = ranked[:low] + ranked[len(ranked) - (group_size - low):]
            if in_zone(mixed):
                return mixed
        return lowest

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
        attempt = 0
        while True:
            attempt += 1
            body = signed_precommit_body(precommit=precommit, sign_binding=self._sign, now=self._clock(),
                                         validator_hotkey=self._sessions.validator_hotkey, path=path)
            try:
                answer = await asyncio.to_thread(self._sessions.precommit, body)
            except SessionRefused as refused:
                if (refused.reason == "precommit_exists"
                        and refused.detail.get("precommit_sha256") == precommit.sha256):
                    return
                # Sent again until the window's end (without one, PRECOMMIT_ATTEMPTS times), never at once.
                wait = max(float(refused.retry_after or 0.0), PRECOMMIT_MIN_BACKOFF_S)
                if (refused.status not in (429, 503)
                        or (until is None and attempt >= PRECOMMIT_ATTEMPTS)
                        or (until is not None and self._clock() + wait >= until)):
                    raise
                logger.info("precommit %s throttled (%s): sent again in %.0f s", precommit.sha256[:12],
                            refused.reason, wait)
                await self._sleep(wait)
                continue
            if not isinstance(answer, Mapping) or answer.get("precommit_sha256") != precommit.sha256:
                raise ValueError("the validator recorded another precommit")
            return

    async def _play_seed(self, runner, precommit, pool, seed: int, deadline):
        """One episode of ``seed``: ``(result, session)``, or None (no session, crash, deadline)."""
        from reliquary.miner.forced_draw import DrawBinding

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
                return None
            except TimeoutError:
                logger.warning("seed %d of precommit %s passed its deadline", seed, precommit.sha256[:12])
                return None
            except Exception:
                logger.exception("seed %d of precommit %s crashed", seed, precommit.sha256[:12])
                return None
            # The log is taken only now, at the episode's end (taking it also ends its binding).
            taken = result.session_id
            try:
                session = self._engine.take_session(taken) if taken else None
            except Exception:
                logger.exception("seed %d: taking its generate session failed", seed)
                session = None
            return result, session
        finally:
            for session_id in dict.fromkeys(bound):
                self._draws.drop(session_id)
                if session_id != taken:
                    drop = getattr(self._engine, "drop_session", None)
                    if drop is not None:
                        drop(session_id)

    def _cutoff(self, open_until: float | None, played: list[EpisodeOutcome]) -> float | None:
        """When the seeds still playing are cut: the window's end, or earlier, the proof budget before the
        earliest grading deadline of an episode already played."""
        deadlines = [float(o.result.submit_by) - self._proof_budget_s for o in played
                     if getattr(o.result, "submit_by", None) is not None]
        bounds = deadlines + ([float(open_until)] if open_until is not None else [])
        return min(bounds, default=None)

    async def _play(self, runner, precommit, pool, played: list[EpisodeOutcome], *,
                    open_until: float | None = None) -> None:
        """Every seed of the pool, at once (the runner bounds its live sessions). Each outcome is appended
        to ``played`` as soon as its episode ends, so the caller can withdraw it whatever happens next.
        A seed whose final is machine-signed ``aborted`` is played again (ABORT_REPLAYS); the seeds still
        playing at the cutoff (``_cutoff``) are cancelled."""
        deadline = getattr(runner, "deadline", None)

        async def one(seed: int) -> None:
            for replay in range(ABORT_REPLAYS + 1):
                played_seed = await self._play_seed(runner, precommit, pool, seed, deadline)
                if played_seed is None:
                    return
                result, session = played_seed
                if (not result.ok and getattr(result, "final_status", None) == "aborted"
                        and replay < ABORT_REPLAYS):
                    logger.info("seed %d of precommit %s aborted by its machine: played again", seed,
                                precommit.sha256[:12])
                    continue
                played.append(EpisodeOutcome(seed, result, session))
                return

        tasks = [asyncio.ensure_future(one(seed)) for seed in range(pool.pool_seeds)]
        pending = set(tasks)
        try:
            while pending:
                cutoff = self._cutoff(open_until, played)
                timeout = None if cutoff is None else max(0.0, cutoff - self._clock())
                done, pending = await asyncio.wait(pending, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
                if not done:
                    logger.info("precommit %s: %d seeds still playing at the cutoff, cancelled",
                                precommit.sha256[:12], len(pending))
                    break
        finally:
            for task in pending:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

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
        """One group of (``environment``, ``task_index``) in ``window``: its ``GroupVerdict``, or None when
        no group was sent. ``open_until`` is the window's end (wall clock): no open, precommit or group is
        sent past it, and the seeds still playing then are cancelled."""
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
        chosen: list[EpisodeOutcome] = []
        sent = {"pending": False}              # the validator holds the chosen group, its verdict unknown
        async with runner:
            try:
                await self._play(runner, precommit, pool, played, open_until=open_until)
                kept = await withdraw_inadmissible(list(played), policy=policy, renderer=renderer, prompt=prompt)
                played.clear()
                held = [outcome for outcome, _ in kept]
                admissible = {id(outcome): value for outcome, value in kept}
                if len(held) < pool.group_size:
                    logger.info("precommit %s: %d admissible episodes, %d needed; nothing submitted",
                                precommit.sha256[:12], len(held), pool.group_size)
                    return None
                chosen = self._chosen(held, pool.group_size,
                                      sigma_min_bps=contract.to_dict()["scoring"]["sigma_min_bps"])
                response = await self._send(contract, policy, pool, precommit, chosen, admissible,
                                            randomness=randomness, environment=environment,
                                            task_index=int(task_index), window=int(window), checkpoint=checkpoint,
                                            open_until=open_until, sent=sent)
                # Held until the group's verdict: admitted -> submitted; anything else -> withdrawn below.
                if response is not None and response.accepted:
                    for outcome in chosen:
                        if outcome.result.submitted is not None:
                            outcome.result.submitted()
                    held = [outcome for outcome in held if all(outcome is not c for c in chosen)]
                return response
            finally:
                # Whatever ended the task (a refusal, an error, a cancellation): every episode not submitted
                # is withdrawn, so its session and the hotkey's caps are freed now. A group the validator
                # holds whose verdict is unknown (poll timeout, cancellation) is never withdrawn under its
                # grading: its episodes stay held and lapse on their own.
                if sent["pending"]:
                    held = [outcome for outcome in held if all(outcome is not c for c in chosen)]
                await self._release(played + held)

    def _chosen(self, held: list[EpisodeOutcome], group_size: int, *, sigma_min_bps: int) -> list[EpisodeOutcome]:
        from reliquary.services.scoring import classify_signal

        def in_zone(subset: list[EpisodeOutcome]) -> bool:
            rewards = [float(outcome.result.reward) for outcome in subset]
            return (len(rewards) == group_size and classify_signal(
                rewards, expected=group_size, sigma_min_bps=sigma_min_bps).category == "in-zone")

        chosen = sorted(self.choose_episodes(list(held), group_size=group_size, in_zone=in_zone),
                        key=lambda outcome: outcome.seed_index)
        seeds = [outcome.seed_index for outcome in chosen]
        if (len(chosen) != group_size or len(set(seeds)) != len(seeds)
                or any(all(outcome is not h for h in held) for outcome in chosen)):
            raise ValueError("choose_episodes must return group_size admissible episodes of distinct seeds")
        return chosen

    async def _send(self, contract, policy, pool, precommit, chosen, admissible, *, randomness: str,
                    environment: str, task_index: int, window: int, checkpoint: str,
                    open_until: float | None = None, sent: dict | None = None) -> GroupVerdict | None:
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
        return await self._deliver(request, submit_by=submit_by, open_until=open_until,
                                   label=precommit.sha256[:12], sent=sent)

    async def _await_verdict(self, window: int, merkle_root: str, *, until: float,
                             after_ts: float | None) -> Mapping | None:
        """The group's verdict once it is known (admitted, or refused), or None past ``until``. A verdict
        recorded at or before ``after_ts`` (validator clock) answers an earlier send of the same group."""
        interval = VERDICT_POLL_S
        while True:
            try:
                body = await self._verdicts(window, merkle_root)
            except Exception as exc:
                logger.warning("verdict of %s unavailable: %s", merkle_root[:12], exc)
                body = None
            verdict = body.get("verdict") if isinstance(body, Mapping) else None
            if isinstance(verdict, Mapping):
                ts = verdict.get("ts")
                fresh = after_ts is None or (ts is not None and float(ts) > after_ts)
                known = (verdict.get("accepted") is True and _reason(verdict.get("reason")) != "submitted"
                         ) or verdict.get("is_final") is True
                if fresh and known:
                    return verdict
            now = self._clock()
            if now >= until:
                return None
            await self._sleep(min(interval, until - now))
            interval = min(interval * 1.5, VERDICT_POLL_MAX_S)

    async def _deliver(self, request, *, submit_by: float | None, open_until: float | None,
                       label: str, sent: dict | None = None) -> GroupVerdict | None:
        """Send the group and wait for its verdict; send it again (a new envelope each time) after a
        resendable refusal (``protocol.episode_retry``) or a 503, while before ``submit_by`` and the window's
        end. None when it could not be sent at all."""
        from reliquary.miner.signed_episode import SUBMIT_TRANSIT_S

        sent = {} if sent is None else sent
        outcome: GroupVerdict | None = None
        seen_ts: float | None = None
        for attempt in range(1, SUBMIT_ATTEMPTS + 1):
            now = self._clock()
            if (submit_by is not None and now > submit_by) or (open_until is not None and now >= open_until):
                logger.warning("precommit %s: past the grading deadline or the window, not sent", label)
                return outcome
            response = await self._submit(request)
            reason = _reason(getattr(response, "reason", None))
            wait = SUBMIT_RETRY_S * attempt
            if reason == "submitted":
                # A queue receipt: the verdict comes from the validator's verdict record. The validator
                # grades until the group's own deadline (submit_by): the window's end only stops sending.
                sent["pending"] = True
                if submit_by is not None:
                    until = submit_by + SUBMIT_TRANSIT_S
                elif open_until is not None:
                    until = open_until + SUBMIT_TRANSIT_S
                else:
                    until = self._clock() + VERDICT_WAIT_S
                verdict = await self._await_verdict(request.window_start, request.merkle_root, until=until,
                                                    after_ts=seen_ts)
                if verdict is None:
                    # Still held: never withdrawn under a grading that may yet admit it.
                    logger.warning("precommit %s: no verdict before the grading deadline; episodes left held", label)
                    return GroupVerdict(False, "submitted")
                sent["pending"] = False
                if verdict.get("ts") is not None:
                    seen_ts = float(verdict["ts"])
                outcome = GroupVerdict(verdict.get("accepted") is True, _reason(verdict.get("reason")),
                                       verdict.get("reject_stage"), dict(verdict))
                resend = not outcome.accepted and group_resendable(outcome.reason, outcome.stage)
            else:
                outcome = GroupVerdict(bool(getattr(response, "accepted", False)) and reason == "accepted", reason)
                resend = not outcome.accepted and (reason == "window_not_active"
                                                   or group_resendable(reason, None))
                if reason == "window_not_active":
                    wait = max(float(getattr(response, "_retry_after_seconds", None) or 0.0), SUBMIT_RETRY_S)
            logger.info("precommit %s: group %s %s %s", label, "admitted" if outcome.accepted else "refused",
                        outcome.reason, outcome.stage or "")
            if not resend:
                return outcome
            if attempt < SUBMIT_ATTEMPTS:
                logger.info("precommit %s: sent again in %.0f s", label, wait)
                await self._sleep(wait)
        return outcome


__all__ = ["EpisodeGroupMiner", "EpisodeOutcome", "GroupVerdict", "hf_prover", "http_verdicts"]
