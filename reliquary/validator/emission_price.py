"""Discovered emission price for one task, replayed from window outcomes.

The pool is wired to ``1.0`` per window today, so the subnet pays its full
miner share regardless of supply, of what the trainer can consume, or of how
fast the window filled. This module turns that number into a price the market
discovers, and the share it does not pay burns.

The controller tracks ``r = t_collect / max(t_training, t_validation)``: paying
for collection faster than the cycle's own incompressible time buys nothing.
The target is therefore our own performance, so repairing the proof plane
lowers the price with no constant to retune.

Every quantity is in DRAND ROUNDS -- quicknet, one every 3 seconds. Not
windows: see the module's test for why that is not a stylistic choice. Not
chain blocks either, though they would serve: the window loop makes no chain
call, while ``window_open_drand_round`` is already computed at window open and
the whole seal path already reasons in rounds. Rounds are three times finer and
cost nothing extra.

Whichever it is, the unit must be named honestly. Calling a round a "block"
would be the same unit confusion that quietly turned EMA_ALPHA into a 28-hour
time constant.

Purity matters as much as the arithmetic: any weight-only node must be able to
replay this from the published archives and land on the same number, exactly as
it already replays the EMA. Nothing here may read a clock, an environment
variable, or any state that is not an argument.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any
import logging
import math
import statistics

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class PriceParams:
    """Controller settings. Versioned in the image, never env-overridable."""

    start: float
    decay: float
    rounds_per_step: int
    deadband: float
    snap: float
    floor: float
    cap: float
    median_rounds: int
    # How many of the most recent FILLING windows ``last_good`` is the rolling
    # minimum over -- a count of fills, not of windows, so a run of shortages
    # does not shrink it and a single expensive fill cannot raise it. See
    # ``PriceState.recent_fill_prices``.
    last_good_fills: int
    # How many consecutive unfilled windows before an environment's price stops
    # escalating: past this, the likelier explanation is that the environment is
    # broken on our side, not that the market is short.
    breaker_timeouts: int = 3


@dataclass(frozen=True, slots=True)
class WindowOutcome:
    """What one archived window contributes to the price.

    ``collect_ready_round`` is when the N-th ADMISSIBLE candidate arrived, not
    when it was proven. The distinction is the whole sensor: a fill-closed
    window closes on PROVEN groups, so a proof-clock reading would measure our
    own plane (11 proofs/min) and never the market.
    """

    open_round: int
    close_round: int
    collect_ready_round: int | None
    incompressible_rounds: float

    @property
    def elapsed_rounds(self) -> int:
        return int(self.close_round) - int(self.open_round)

    @property
    def filled(self) -> bool:
        """Whether the window gathered its target at all."""
        return self.collect_ready_round is not None

    @property
    def ratio(self) -> float | None:
        """``r``, or None when no ratio can be computed.

        None means "no signal", NOT "shortage": a window that filled but whose
        stage durations were never recorded carries no information either way.
        Shortage is ``filled`` being false, and only that.
        """
        if self.collect_ready_round is None:
            return None
        if self.incompressible_rounds <= 0:
            return None
        return (
            int(self.collect_ready_round) - int(self.open_round)
        ) / self.incompressible_rounds


@dataclass(frozen=True, slots=True)
class PriceState:
    """What one archive carries forward. Bounded on purpose: see ``advance``.

    ``recent_fill_prices`` is capped at ``params.last_good_fills`` entries --
    fixed-size, so it costs ``replay()`` nothing to re-fold from genesis.
    ``last_good`` stays as the reported scalar so nothing downstream breaks;
    it is DERIVED as the minimum of the tuple when the tuple is non-empty.

    An archive written before this field existed carries none: the default,
    an empty tuple, makes that state read as "no fills recorded yet", and
    ``advance`` falls back to the carried scalar rather than treating the
    empty tuple as a fresh, unfilled controller.
    """

    price: float
    last_good: float
    recent_fill_prices: tuple[float, ...] = ()


@dataclass(frozen=True, slots=True)
class PriceDecision:
    """The price plus the inputs that produced it, so the archive can carry both."""

    price: float
    last_good: float
    recent_fill_prices: tuple[float, ...]
    r: float | None
    r_smoothed: float | None
    regime: str

    @property
    def state(self) -> PriceState:
        return PriceState(
            price=self.price,
            last_good=self.last_good,
            recent_fill_prices=self.recent_fill_prices,
        )


def _timed_out(outcome: Any) -> bool:
    """Whether this outcome's window timed out.

    Read leniently because both walks share ``advance``: only
    ``EnvironmentOutcome`` carries the field, and the scalar ``WindowOutcome``
    walk must stay byte-identical to what it was before it existed.
    """
    return bool(getattr(outcome, "timed_out", False))


def _smoothed_ratio(
    recent: Sequence[WindowOutcome], median_rounds: int
) -> float | None:
    """Median ``r`` over the trailing ``median_rounds``, or None if not finite.

    A window that never filled contributes ``inf``: it is maximally slow, and
    letting it be absent would allow a run of failures to leave the median
    claiming over-supply. When the median itself is infinite there is no usable
    ratio -- and no finite value could be archived either, since Reliquary's
    canonical JSON refuses non-finite floats.
    """
    cutoff = recent[-1].close_round - median_rounds
    samples: list[float] = []
    for outcome in recent:
        if outcome.close_round <= cutoff:
            continue
        if not outcome.filled:
            samples.append(math.inf)
            continue
        ratio = outcome.ratio
        if ratio is not None:
            samples.append(ratio)
    if not samples:
        return None
    value = statistics.median(samples)
    return value if math.isfinite(value) else None


def advance(
    state: PriceState,
    recent: Sequence[WindowOutcome],
    params: PriceParams,
) -> PriceDecision:
    """Decide the price after one window.

    ``recent`` is the trailing run of outcomes, oldest first, whose LAST element
    is the window being decided; anything outside ``median_rounds`` is dropped
    here rather than trusted from the caller.

    Taking a ``PriceState`` instead of the whole chain is what makes this
    replayable: ``_replay_ema`` reads a bounded slice of archives, so a price
    that needed folding back to genesis would put a weight-only node and the
    validator that wrote the archive on different numbers.
    """
    outcome = recent[-1]
    r = outcome.ratio
    r_smoothed = _smoothed_ratio(recent, params.median_rounds)
    price = state.price
    recent_fill_prices = state.recent_fill_prices
    # ``last_good`` is DERIVED from the rolling tuple, not carried as an
    # independent value: a fill at a raised price appends to the tuple, it
    # never overwrites it, so one expensive incident cannot lift the floor
    # the next snap escalates from. An empty tuple -- genesis, or a state
    # revived from an archive written before this field existed -- falls
    # back to the carried scalar instead of reading as "nothing ever filled".
    last_good = min(recent_fill_prices) if recent_fill_prices else state.last_good
    if not outcome.filled:
        # The window never gathered its target, so the trainer is stopped:
        # there is no graceful degradation to ride out. Escalating from the
        # CURRENT price rather than pinning to ``last_good`` is what keeps a
        # snap that fails to restore supply from parking there forever.
        price = max(price, last_good) * params.snap
        regime = "snap"
    elif r_smoothed is not None and r_smoothed < params.deadband:
        price *= params.decay ** (outcome.elapsed_rounds / params.rounds_per_step)
        regime = "descend"
    else:
        regime = "hold"
    price = min(max(price, params.floor), params.cap)
    # A timed-out window assembled no batch, so it is not evidence that this
    # price works no matter what its admission clock says.
    if outcome.filled and not _timed_out(outcome):
        # Only a window that actually filled proves a price works. It joins
        # the rolling window rather than replacing ``last_good`` outright, so
        # the minimum -- not the latest fill -- is what the next snap reads.
        recent_fill_prices = (*recent_fill_prices, price)[-params.last_good_fills:]
        last_good = min(recent_fill_prices)
    return PriceDecision(
        price=price,
        last_good=last_good,
        recent_fill_prices=recent_fill_prices,
        r=r,
        r_smoothed=r_smoothed,
        regime=regime,
    )


@dataclass(frozen=True, slots=True)
class EnvironmentOutcome:
    """One environment's contribution to one window's price signal.

    The numerator is this environment's own readiness; the denominator is the
    window's, because collecting one environment faster than the shared
    training step can consume buys nothing.
    """

    environment: str
    open_round: int
    close_round: int
    ready_round: int | None
    incompressible_rounds: float
    # Whether the WINDOW timed out: ``filled`` reads the ADMISSION clock while
    # a timeout is decided on PROVEN groups, so the two disagree in the
    # dominant shape (arrivals reached target, proofs did not). Last and
    # defaulted, so every existing construction keeps working.
    timed_out: bool = False

    @property
    def elapsed_rounds(self) -> int:
        return int(self.close_round) - int(self.open_round)

    @property
    def filled(self) -> bool:
        return self.ready_round is not None

    @property
    def ratio(self) -> float | None:
        if self.ready_round is None or self.incompressible_rounds <= 0:
            return None
        return (
            int(self.ready_round) - int(self.open_round)
        ) / self.incompressible_rounds


def advance_by_environment(
    states: Mapping[str, PriceState],
    recent_by_environment: Mapping[str, Sequence[EnvironmentOutcome]],
    params: PriceParams,
) -> dict[str, PriceDecision]:
    """Decide every environment's price independently, on one shared rule.

    ``advance`` is reused unchanged: one environment's price is decided by
    exactly the rule that decided the window's, so there is still one
    controller to calibrate, not one per environment. The one exception is
    the breaker below: a structurally dry environment must stop escalating
    without being dropped from the mix, which ``advance`` alone cannot do.
    """
    decisions: dict[str, PriceDecision] = {}
    for environment, recent in recent_by_environment.items():
        if not recent:
            continue
        state = states.get(
            environment,
            PriceState(price=params.start, last_good=params.start, recent_fill_prices=()),
        )
        # A breaker length of 0 disables the breaker. Without the guard,
        # ``[-0:]`` is the WHOLE list and ``all`` over an empty trailing run is
        # vacuously true, so 0 froze on the first window instead.
        breaker_length = max(int(params.breaker_timeouts), 0)
        trailing = list(recent)[-breaker_length:] if breaker_length else []
        # A timed-out window counts against the breaker even when its
        # admission clock reached target: the environment produced no trained
        # batch, which is the halting shape the breaker exists for -- and the
        # dominant one.
        if breaker_length and len(trailing) >= breaker_length and all(
            _timed_out(outcome) or not outcome.filled
            for outcome in trailing
        ):
            # The regime is the same either way -- go look -- but the two
            # causes are not: no arrivals means the market is not there, while
            # arrivals that never became proofs means we cannot keep up.
            starved = sum(1 for outcome in trailing if not outcome.filled)
            unproven = len(trailing) - starved
            if not unproven:
                cause = (
                    "no admissible candidate ever reached its target: the "
                    "MARKET is not supplying this environment"
                )
            elif not starved:
                cause = (
                    "arrivals reached its target but never became proofs: "
                    "OUR proof plane cannot keep up with the supply"
                )
            else:
                cause = (
                    f"mixed -- {starved} window(s) short on arrivals, "
                    f"{unproven} whose arrivals never became proofs"
                )
            # This is real subnet-halting news -- fill-closed cannot proceed
            # without this environment -- even though the frozen number is,
            # today, only ever a SHADOW price (this function's one production
            # caller is Phase 1's unapplied walk): say so explicitly, so the
            # critical does not read as money moving.
            logger.critical(
                "environment %s produced no trained batch for %d consecutive "
                "windows (%s); freezing its SHADOW price (applied: False) at "
                "%.4f -- treat the environment itself as broken on our side "
                "until shown otherwise",
                environment, breaker_length, cause, state.price,
            )
            decisions[environment] = PriceDecision(
                price=state.price,
                last_good=state.last_good,
                recent_fill_prices=state.recent_fill_prices,
                r=None,
                r_smoothed=None,
                regime="frozen",
            )
            continue
        decisions[environment] = advance(state, recent, params)
    return decisions


def replay(
    outcomes: Sequence[WindowOutcome], params: PriceParams
) -> PriceDecision:
    """Fold a whole history. The verification path, not the hot one."""
    state = PriceState(price=params.start, last_good=params.start)
    decision = PriceDecision(
        price=params.start,
        last_good=params.start,
        recent_fill_prices=(),
        r=None,
        r_smoothed=None,
        regime="hold",
    )
    for index in range(len(outcomes)):
        decision = advance(state, outcomes[: index + 1], params)
        state = decision.state
    return decision


# Absent from every archive written before this shipped. Their absence is how
# the replay tells "not instrumented" from "did not fill": the latter is an
# explicit ``collect_ready_round: null`` INSIDE an otherwise complete record.
_REQUIRED_ARCHIVE_FIELDS = (
    "window_open_round",
    "window_close_round",
)


def outcome_from_archive(record: Mapping[str, Any]) -> WindowOutcome | None:
    """Read one archive's price signal, or None when it carries none.

    Aborted windows are skipped for the same reason ``_replay_ema`` skips them:
    their timings describe the abort, not the market.
    """
    if record.get("window_status", "completed") == "aborted":
        return None
    if any(record.get(field) is None for field in _REQUIRED_ARCHIVE_FIELDS):
        return None
    ready = record.get("collect_ready_round")
    open_round = int(record["window_open_round"])
    close_round = int(record["window_close_round"])
    measured = max(
        float(record.get("training_rounds") or 0.0),
        float(record.get("validation_rounds") or 0.0),
    )
    # A fill-closed window closes when the LAST constraint is lifted, so its
    # span IS max(t_collect, t_stages) -- the denominator r wants, for free,
    # where per-stage timing does not exist in the window loop at all today.
    # The ratio it yields is capped at 1 and so cannot report how far past the
    # stages collection ran; that costs the controller nothing (it has no
    # raise-on-slow regime -- shortage is a window that never filled), and real
    # stage timings sharpen it without a schema change.
    incompressible = measured if measured > 0 else float(close_round - open_round)
    if record.get("window_status") == "timed_out":
        # ``collect_ready_round`` is built from ADMITTED candidates, while a
        # timeout is decided on PROVEN groups -- admission runs well ahead of
        # proof capacity (to ``2 * FILL_CLOSED_TARGET_GROUPS_PER_ENV``), so a
        # timed-out window routinely still shows a ready round despite
        # training on nothing: arrivals-full/proofs-short is the dominant
        # timeout shape, not the exception. No batch was assembled, so no
        # ``ratio`` this ``incompressible_rounds`` could produce would mean
        # anything; forcing it to 0 sends ``ratio`` through ``WindowOutcome``'s
        # own existing "no denominator" rule instead of a second special
        # case. ``filled`` is untouched -- it still reads off
        # ``collect_ready_round`` below, so a window that reached its ready
        # round is still a HOLD in ``advance``, not silence; only the ratio
        # this window would otherwise report is what gets suppressed, because
        # it is not a real measurement of anything. This is the one place the
        # rule lives -- ``outcomes_by_environment_from_archive`` reads it back
        # off this same ``WindowOutcome`` rather than repeating it. The
        # per-environment walk carries the timeout separately, as
        # ``EnvironmentOutcome.timed_out``, so the breaker can count it and
        # ``recent_fill_prices`` can refuse it.
        incompressible = 0.0
    return WindowOutcome(
        open_round=open_round,
        close_round=close_round,
        collect_ready_round=None if ready is None else int(ready),
        incompressible_rounds=incompressible,
    )


def outcomes_by_environment_from_archive(
    record: Mapping[str, Any],
) -> dict[str, EnvironmentOutcome] | None:
    """Per-environment outcomes, or None when the record carries no map.

    Archives written before the map existed carry only the scalar; they keep
    replaying through ``outcome_from_archive`` and contribute nothing here.

    Each environment's entry is converted independently: a hand-repaired
    archive that leaves ONE environment's ready round unreadable -- a typo, a
    bool, a list -- costs only that environment a signal. Failing the whole
    record for one bad entry would defeat the isolation this module exists to
    create, taking down every other environment's price along with it.
    """
    if record.get("window_status", "completed") == "aborted":
        return None
    by_environment = record.get("collect_ready_round_by_environment")
    if not isinstance(by_environment, dict):
        return None
    scalar = outcome_from_archive(record)
    if scalar is None:
        return None
    # ``outcome_from_archive`` already zeroes ``incompressible_rounds`` for a
    # "timed_out" record (one owner for that rule); every environment reads
    # the same scalar back rather than re-deriving it here.
    timed_out = record.get("window_status") == "timed_out"
    outcomes: dict[str, EnvironmentOutcome] = {}
    for environment, ready in by_environment.items():
        if not isinstance(environment, str):
            logger.warning(
                "archive %s carries a non-string environment key (%r); "
                "skipping that entry only",
                record.get("window_open_round"), environment,
            )
            continue
        try:
            # bool is an int subclass -- int(True) == 1 -- so it must be
            # refused explicitly rather than silently converted to a round.
            if isinstance(ready, bool):
                raise TypeError(f"bool is not a valid ready round: {ready!r}")
            resolved = None if ready is None else int(ready)
        except (TypeError, ValueError):
            # None means MEASURED and it did not fill -- the shortage that
            # snaps the price up. An unreadable entry means we do not know,
            # which is no signal at all, so it is omitted rather than given
            # None's shortage meaning: that would raise a price on a typo.
            logger.warning(
                "archive %s carries an unreadable ready round for "
                "environment %s (%r); skipping that environment only",
                record.get("window_open_round"), environment, ready,
            )
            continue
        outcomes[environment] = EnvironmentOutcome(
            environment=environment,
            open_round=scalar.open_round,
            close_round=scalar.close_round,
            ready_round=resolved,
            incompressible_rounds=scalar.incompressible_rounds,
            timed_out=timed_out,
        )
    return outcomes


def distinct_prompt_arrival_rounds(
    arrivals_by_prompt: Mapping[int, Sequence[int]],
) -> list[int]:
    """One round per distinct prompt: its earliest admissible arrival.

    A window's target counts GROUPS, and a group is one prompt.
    ``MAX_SUBMISSIONS_PER_PROMPT`` lets ten candidates chase the same prompt, so
    counting raw submissions would report supply the window cannot use.
    """
    return [
        min(rounds) for rounds in arrivals_by_prompt.values() if rounds
    ]


def ready_round(arrival_rounds: Sequence[int], target: int) -> int | None:
    """The round by which ``target`` admissible candidates had arrived.

    ``None`` means supply never reached the target -- the shortage that snaps
    the price up.
    """
    if target <= 0 or len(arrival_rounds) < target:
        return None
    return sorted(arrival_rounds)[target - 1]


def ready_rounds_by_environment(
    arrivals_by_environment: Mapping[str, Sequence[int]],
    targets_by_environment: Mapping[str, int],
) -> dict[str, int | None]:
    """When each environment reached its own target, or None for one that did not.

    ``window_ready_round`` computes exactly this and keeps only the maximum;
    a per-environment price needs the values it throws away.
    """
    return {
        environment: ready_round(
            arrivals_by_environment.get(environment, []),
            target,
        )
        for environment, target in targets_by_environment.items()
    }


def window_ready_round(
    arrivals_by_environment: Mapping[str, Sequence[int]],
    targets_by_environment: Mapping[str, int],
) -> int | None:
    """The round the SLOWEST environment reached its target.

    A fill-closed window is not ready until every environment holds its own
    target, so averaging across them would report a readiness neither one had.
    """
    rounds = ready_rounds_by_environment(
        arrivals_by_environment, targets_by_environment
    )
    if any(reached is None for reached in rounds.values()):
        return None
    return max(rounds.values(), default=None)


@dataclass(frozen=True)
class RestoredWalk:
    """The price walk as the archives left it, and the last decision they carry."""

    state: PriceState | None
    history: tuple[WindowOutcome, ...]
    states_by_environment: dict[str, PriceState]
    history_by_environment: dict[str, tuple[EnvironmentOutcome, ...]]
    latest_shadow: dict[str, Any] | None


def _archived_price(decision: Any, params: PriceParams) -> tuple[float, float] | None:
    """An archived decision's price and last_good, clamped, or None if unreadable."""
    if not isinstance(decision, Mapping):
        return None
    values = []
    for field in ("price", "last_good"):
        value = decision.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        if not math.isfinite(value):
            return None
        values.append(min(max(float(value), params.floor), params.cap))
    return values[0], values[1]


def restore_walk(
    archives: Sequence[Mapping[str, Any]], params: PriceParams
) -> RestoredWalk:
    """Rebuild the price walk from archived windows, oldest first.

    Each archive carries its window's decision and the signal behind it, so the
    walk resumes where it stopped; the rolling minimum is rebuilt with exactly
    the rule ``advance`` applies, one window at a time.
    """
    history: list[WindowOutcome] = []
    history_by_environment: dict[str, list[EnvironmentOutcome]] = {}
    state: PriceState | None = None
    states_by_environment: dict[str, PriceState] = {}
    fills: tuple[float, ...] = ()
    fills_by_environment: dict[str, tuple[float, ...]] = {}
    latest_shadow: dict[str, Any] | None = None
    for record in archives:
        if not isinstance(record, Mapping) or record.get("window_status") == "aborted":
            continue
        outcome = outcome_from_archive(record)
        if outcome is not None:
            history.append(outcome)
        environment_outcomes = outcomes_by_environment_from_archive(record) or {}
        for environment, environment_outcome in environment_outcomes.items():
            history_by_environment.setdefault(environment, []).append(environment_outcome)
        shadow = record.get("emission_price_shadow")
        decided = _archived_price(shadow, params)
        if decided is not None:
            latest_shadow = dict(shadow)
            price, last_good = decided
            if outcome is not None and outcome.filled and not _timed_out(outcome):
                fills = (*fills, price)[-params.last_good_fills:]
            state = PriceState(
                price=price,
                last_good=min(fills) if fills else last_good,
                recent_fill_prices=fills,
            )
        by_environment = shadow.get("by_environment") if isinstance(shadow, Mapping) else None
        if not isinstance(by_environment, Mapping):
            continue
        for environment, decision in by_environment.items():
            environment_decided = _archived_price(decision, params)
            if environment_decided is None:
                continue
            price, last_good = environment_decided
            environment_fills = fills_by_environment.get(environment, ())
            environment_outcome = environment_outcomes.get(environment)
            if (
                environment_outcome is not None
                and environment_outcome.filled
                and not _timed_out(environment_outcome)
            ):
                environment_fills = (*environment_fills, price)[-params.last_good_fills:]
                fills_by_environment[environment] = environment_fills
            states_by_environment[environment] = PriceState(
                price=price,
                last_good=min(environment_fills) if environment_fills else last_good,
                recent_fill_prices=environment_fills,
            )
    return RestoredWalk(
        state=state,
        history=tuple(history),
        states_by_environment=states_by_environment,
        history_by_environment={
            environment: tuple(trail) for environment, trail in history_by_environment.items()
        },
        latest_shadow=latest_shadow,
    )


def price_signal_fields(
    *,
    open_round: int | None,
    close_round: int | None,
    arrivals_by_environment: Mapping[str, Mapping[int, Sequence[int]]] | None,
    targets_by_environment: Mapping[str, int],
) -> dict[str, Any] | None:
    """The three fields a window contributes to the archive, or None.

    They travel together or not at all. A record carrying the window bounds but
    not the readiness reads as a SHORTAGE -- the one regime that needs no
    confirmation and snaps the price up -- when all that happened is that the
    validator could not measure. Silence has to look like silence.

    An explicit ``collect_ready_round: None`` INSIDE a complete record is the
    opposite: measured, and it did not fill. That is real news and it travels.
    """
    if open_round is None or close_round is None or arrivals_by_environment is None:
        return None
    collapsed = {
        environment: distinct_prompt_arrival_rounds(arrivals)
        for environment, arrivals in arrivals_by_environment.items()
    }
    return {
        "window_open_round": int(open_round),
        "window_close_round": int(close_round),
        "collect_ready_round": window_ready_round(
            collapsed,
            targets_by_environment,
        ),
        "collect_ready_round_by_environment": ready_rounds_by_environment(
            collapsed, targets_by_environment
        ),
    }


# Versioned in the image on purpose, and deliberately NOT in constants.py: the
# replay has to be reproducible from a release, while constants.py is where
# env-overridable settings live. An operator turning a knob at runtime would
# put two weight-only nodes on two different prices.
#
# None of these numbers is settled. They are a starting point whose whole job
# is to produce a shadow curve legible enough to calibrate them against.
PRODUCTION_PRICE_PARAMS = PriceParams(
    # Today's pool, so the first armed window changes nothing.
    start=1.0,
    # -2% per ~50 minutes of window time: about a week from 1.0 to 0.05 at a
    # 17-minute cycle, still ~5x slower than the ~12 h miners take to feel a price.
    decay=0.98,
    rounds_per_step=1000,
    # Collection close to the incompressible time is the target, not a signal.
    deadband=0.80,
    snap=1.20,
    # A liveness guard, not an economic opinion. Where the cliff actually sits
    # is what the shadow phase exists to find out.
    floor=0.05,
    cap=1.0,
    # ~4 hours: several windows of confirmation before spending less, against a
    # loop delay of one miner spin-up plus one window.
    median_rounds=4800,
    # A starting point, to be calibrated during the shadow phase like every
    # other number in this block: the rolling minimum tracks the last 50
    # filling windows.
    last_good_fills=50,
    # A starting point, calibrated in shadow like every other number here.
    breaker_timeouts=3,
)
