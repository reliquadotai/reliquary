"""Which service submissions a window admits. The server's announcement decides, never the miner."""
from __future__ import annotations

from dataclasses import dataclass

from reliquary.protocol.service_contract import PUBLIC_SEED_POOL, SUPPORTED_V2_CAPABILITIES, ServiceContract


def parse_service_announcement(announcement: dict, *, contract: ServiceContract | None = None):
    """``(contract, schedule)`` of a window's announcement, checked (v2 only, capabilities). A batcher
    parses its frozen announcement once per window and hands the pair to ``validate_submission_policy``.
    ``contract``: the announcement's contract when the caller already parsed it."""
    from reliquary.protocol.service_schedule import ServiceSchedule

    if contract is None:
        contract = ServiceContract.from_dict(announcement["contract"])
    if contract.version != 2:
        raise ValueError("only service-contract/v2 runs RL")
    contract.require_capabilities(set(announcement["supported_capabilities"]))
    contract.require_capabilities(set(SUPPORTED_V2_CAPABILITIES))
    return contract, ServiceSchedule.from_dict(announcement["schedule"], contract)


def validate_submission_policy(request, announcement: dict | None, *, parsed=None) -> ServiceContract | None:
    """``parsed``: the announcement's ``parse_service_announcement`` result, when the caller holds it."""
    from reliquary.protocol.seed_pool import PoolSelection, pool_from_service_policy, validate_rollout_selection
    from reliquary.protocol.service_submission import ServiceBinding, validate_service_rollout_bindings

    binding = getattr(request, "service_binding", None)
    selection = getattr(request, "pool_selection", None)
    metadata = [r.commit.get("rollout") or {} for r in request.rollouts]
    if any(not isinstance(row, dict) for row in metadata):
        raise ValueError("invalid rollout metadata")
    if announcement is None:
        if (binding is not None or selection is not None
                or any(row.get("service_binding") is not None or row.get("seed_pool") is not None for row in metadata)):
            raise ValueError("service metadata requires an active service task")
        return None
    contract, schedule = parse_service_announcement(announcement) if parsed is None else parsed
    if binding is None:
        raise ValueError("service task requires a signed service binding")
    intent = ServiceBinding.from_dict(binding)
    if intent.contract_sha256 != contract.sha256:
        raise ValueError("service contract revision mismatch")
    commits = [r.commit for r in request.rollouts]
    validate_service_rollout_bindings(intent, commits)
    if request.checkpoint_hash != announcement["checkpoint"]["revision"]:
        raise ValueError("service checkpoint mismatch")
    environments = {r.env_name for r in request.rollouts}
    if len(environments) != 1:
        raise ValueError("a service group belongs to one environment")
    environment = next(iter(environments))
    if environment not in schedule.active_environments():
        raise ValueError("service environment is not active")
    policy = contract.environment(environment)
    if policy["sampling"]["kind"] == PUBLIC_SEED_POOL:
        pool = pool_from_service_policy(announcement, environment=environment, prompt_idx=request.prompt_idx,
                                        checkpoint_hash=request.checkpoint_hash)
        validate_rollout_selection(pool, PoolSelection.from_dict(selection), commits)
    elif selection is not None or any(row.get("seed_pool") is not None for row in metadata):
        raise ValueError("pool metadata is not allowed by this contract")
    if intent.purpose == "exploration" and policy["exploration"] != 1:
        raise ValueError("exploration is not enabled for this environment")
    return contract


MISSING_BOX_UNCERTAIN = "uncertain"
MISSING_BOX_GRADED = "graded"
# Decision H / confirmed ruling 4: a missing ``\boxed`` is an uncertain outcome for maths only.
# Every other boxed environment (science) keeps it as a plain 0.
MATH_ENVIRONMENTS = frozenset({"openmathinstruct", "reliquary_dapo_math_v1", "reliquary_hard_math_v1"})
EXPLORATION_TRUNCATED = "truncated"


def default_missing_box(environment: str) -> str:
    """The ``missing_box`` value a contract should carry for this environment (its type decides)."""
    return MISSING_BOX_UNCERTAIN if environment in MATH_ENVIRONMENTS else MISSING_BOX_GRADED


def missing_box_problems(contract: ServiceContract) -> list[str]:
    """R22: every env's ``missing_box`` must be its by-type default; one line per env that is not."""
    return [
        f"{name}: missing_box is {env['missing_box']!r}, the default for its type is {default_missing_box(name)!r}"
        for name, env in sorted(contract.environments.items())
        if env["missing_box"] != default_missing_box(name)
    ]


def missing_box_is_uncertain(environment: str, contract: ServiceContract) -> bool:
    """Whether a properly terminated completion with no ``\\boxed`` is an uncertain outcome.

    It grades 0 either way. The contract's per-environment ``missing_box`` decides, and only
    for an environment whose final answer is boxed: elsewhere there is no box to miss.
    """
    from reliquary.environment.registry import get_environment_spec

    try:
        boxed = get_environment_spec(environment).final_answer_policy == "boxed"
    except ValueError:
        boxed = False
    return boxed and contract.environment(environment)["missing_box"] == MISSING_BOX_UNCERTAIN


def _indices(values, size: int) -> tuple[int, ...]:
    out = tuple(dict.fromkeys(values))
    if any(type(index) is not int or not 0 <= index < size for index in out):
        raise ValueError("a rollout index must identify one rollout of the group")
    return out


def uncertain_rollout_indices(*, truncated_indices=(), unboxed_indices=(), size: int) -> tuple[int, ...]:
    """Rollouts whose reward is not trusted: cut by the length cap (every environment), or
    terminated without a box where the caller established that a missing box is uncertain."""
    return _indices((*truncated_indices, *unboxed_indices), size)


@dataclass(frozen=True, slots=True)
class ExplorationEntitlement:
    entitled: bool
    reason: str | None
    rewards: tuple[float, ...]


def exploration_pay_entitlement(rewards, *, truncated_indices=(), uncertain_indices=()) -> ExplorationEntitlement:
    """Whether an exploration group may be paid, as far as termination goes, and what to record.

    The group is an observation and is published either way. It may be paid only when every
    rollout terminated properly: one rollout cut by the length cap makes it unpaid (public
    reason ``truncated``), never sanctioned. A terminated rollout with no box does not block
    pay and is recorded as reward 0. First scan, cap, audit and bans are decided elsewhere.
    """
    values = [float(reward) for reward in rewards]
    truncated = _indices(truncated_indices, len(values))
    for index in _indices(uncertain_indices, len(values)):
        if index not in truncated:
            values[index] = 0.0
    if truncated:
        return ExplorationEntitlement(False, EXPLORATION_TRUNCATED, tuple(values))
    return ExplorationEntitlement(True, None, tuple(values))


LANE_TRAINING = "training"
LANE_EXPLORATION = "exploration"
LANE_UNPROVEN = "unproven"
UNPROVEN_NOT_ROBUST = "not_robust"


@dataclass(frozen=True, slots=True)
class ServiceLane:
    """Which lane a group belongs to, decided by the validator from the graded vector alone.

    ``rewards`` is the vector the lane classified and the run log records (R24: for exploration,
    uncertain-but-terminated rollouts are already zeroed). ``reason`` is set for ``unproven``.
    """
    lane: str
    rewards: tuple[float, ...]
    reason: str | None = None


def _training_admits(contract: ServiceContract, graded: list[float], size: int, uncertain_indices,
                     attainable_rewards) -> bool:
    from reliquary.services.scoring import classify_signal

    sigma_min_bps = contract.to_dict()["scoring"]["sigma_min_bps"]
    if not classify_signal(graded, expected=size, sigma_min_bps=sigma_min_bps).in_zone:
        return False
    try:
        uncertain = _indices(uncertain_indices, len(graded))
    except ValueError:
        return False
    if not uncertain:
        return True
    lattice = tuple(round(float(r) * 10000) / 10000 for r in attainable_rewards)
    if not lattice:  # unknown lattice: no completion can be ruled out, so nothing is proven robust
        return False
    from reliquary.validator.admission import robust_utility_admits

    try:
        return robust_utility_admits(graded, sigma_min=sigma_min_bps / 10000,
                                     truncated_indices=uncertain, attainable_rewards=lattice)
    except (ValueError, TypeError, OverflowError):  # a malformed lattice proves nothing
        return False


def service_lane(request, contract: ServiceContract, rewards, *, truncated_indices=(), uncertain_indices=(),
                 attainable_rewards=()) -> ServiceLane | None:
    """The lane of a group. The miner's declared ``purpose`` plays no part: the vector decides.

    * observed vector in zone, and robust to its uncertain rollouts -> ``training``;
    * observed vector in zone but not robust (R23) -> ``unproven``: published, unpaid, not a scan;
    * otherwise, with the uncertain-but-terminated rollouts zeroed first (R24) the vector is still
      out of zone -> ``exploration`` (whether it is PAID is ``exploration_pay_entitlement``);
    * a vector that zeroing would put in zone can never be exploration -> ``unproven``;
    * an incomplete group is no observation -> None.
    """
    from reliquary.services.scoring import classify_signal

    sigma_min_bps = contract.to_dict()["scoring"]["sigma_min_bps"]
    size = len(request.rollouts)
    graded = [round(float(r) * 10000) / 10000 for r in rewards]  # the precision the run log records
    observed = classify_signal(graded, expected=size, sigma_min_bps=sigma_min_bps)
    if observed.category == "unknown":
        return None
    if observed.in_zone:
        if _training_admits(contract, graded, size, uncertain_indices, attainable_rewards):
            return ServiceLane(LANE_TRAINING, tuple(graded))
        return ServiceLane(LANE_UNPROVEN, tuple(graded), UNPROVEN_NOT_ROBUST)
    zeroed = [round(float(r) * 10000) / 10000 for r in exploration_pay_entitlement(
        graded, truncated_indices=truncated_indices, uncertain_indices=uncertain_indices).rewards]
    if classify_signal(zeroed, expected=size, sigma_min_bps=sigma_min_bps).in_zone:
        return ServiceLane(LANE_UNPROVEN, tuple(zeroed), UNPROVEN_NOT_ROBUST)
    return ServiceLane(LANE_EXPLORATION, tuple(zeroed))
