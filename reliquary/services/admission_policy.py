"""Which service submissions a window admits. The server's announcement decides, never the miner."""
from __future__ import annotations

from reliquary.protocol.service_contract import PUBLIC_SEED_POOL, SUPPORTED_V2_CAPABILITIES, ServiceContract

UNIFORM = frozenset({"uniform-low", "uniform-high", "uniform-intermediate"})


def validate_submission_policy(request, announcement: dict | None) -> ServiceContract | None:
    from reliquary.protocol.seed_pool import PoolSelection, pool_from_service_policy, validate_rollout_selection
    from reliquary.protocol.service_schedule import ServiceSchedule
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
    contract = ServiceContract.from_dict(announcement["contract"])
    if contract.version != 2:
        raise ValueError("only service-contract/v2 runs RL")
    contract.require_capabilities(set(announcement["supported_capabilities"]))
    contract.require_capabilities(set(SUPPORTED_V2_CAPABILITIES))
    schedule = ServiceSchedule.from_dict(announcement["schedule"], contract)
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


def service_signal_admits(request, contract: ServiceContract, rewards: list[float], *,
                          uncertain_indices=(), attainable_rewards=(0.0, 1.0)) -> bool:
    from reliquary.services.scoring import classify_signal

    if tuple(uncertain_indices):
        return False
    signal = classify_signal([round(r * 10000) / 10000 for r in rewards], expected=len(request.rollouts),
                             sigma_min_bps=contract.to_dict()["scoring"]["sigma_min_bps"])
    if request.service_binding["purpose"] == "training":
        return signal.in_zone
    return signal.category in UNIFORM
