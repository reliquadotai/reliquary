"""Read-only local prerequisites for an allocated runtime; never permission to launch one."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import re
import stat
import time
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from reliquary.rl_orders import supervisor as s
from reliquary.shared.strict_json import strict_json_loads

SCHEMA = "reliquary/training-preflight/v1"
UNVERIFIED = (
    "current_registry_authorization",
    "checkpoint_namespace_absence_and_atomic_claim",
    "exact_run_signer_authorization_and_mtls",
    "delegated_cgroup_creation_and_complete_owned_stop",
    "gpu_capacity_model_loading_and_numerical_correctness",
    "native_runtime_cuda_import_and_kernel_abi",
    "listener_isolation_and_operator_ingress",
    "durable_checkpoint_cache_and_owner_delivery",
    "nonpaying_execution_and_financial_isolation",
    "owner_billing_policy_approval",
    "inflight_work_overrun_beyond_admission_limits",
)


def _fresh_state(config):
    state = Path(config.state_dir)
    if any(path.is_symlink() for path in (state, *state.parents)):
        raise ValueError("state path is redirected")
    existing = next(path for path in (state, *state.parents) if path.exists())
    info = existing.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.geteuid() or info.st_mode & 0o077:
        raise ValueError("state needs an existing private owned parent")
    if state.exists():
        for child in state.iterdir():
            if child.name != "supervisor.lock" or not s._private_regular_file(child.lstat()):
                raise ValueError("runtime needs reconciliation")


def _credentials(config):
    values = config.credentials()
    required = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_ID", "R2_PUBLIC_ENDPOINT",
                "R2_PUBLIC_ACCESS_KEY_ID", "R2_PUBLIC_SECRET_ACCESS_KEY", "R2_PUBLIC_BUCKET")
    if not all(values.get(key, "").strip() for key in required):
        raise ValueError("storage credentials are incomplete")
    if not values.get("R2_ENDPOINT_URL") and not values.get("R2_ACCOUNT_ID"):
        raise ValueError("registry endpoint is missing")
    from urllib.parse import urlsplit

    for key in ("RELIQUARY_SIGNER_URL", "R2_PUBLIC_ENDPOINT"):
        url = urlsplit(values[key])
        if (url.scheme != "https" or not url.hostname or url.username or url.password
                or url.query or url.fragment):
            raise ValueError("fixed HTTPS endpoints are required")
        if key == "R2_PUBLIC_ENDPOINT" and (not url.hostname.endswith(".r2.cloudflarestorage.com")
                or url.port not in (None, 443) or url.path not in ("", "/")):
            raise ValueError("checkpoint endpoint must match final delivery")
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]{1,62}", values["R2_PUBLIC_BUCKET"]):
        raise ValueError("checkpoint bucket is invalid")
    for key in ("RELIQUARY_SIGNER_CA", "RELIQUARY_SIGNER_CERT", "RELIQUARY_SIGNER_KEY"):
        if not Path(values[key]).is_absolute() or not s._private_bytes(Path(values[key]), 1 << 20):
            raise ValueError("local signer material is missing")


def _order(config, entry, now):
    from reliquary.validator.rl_training_usage import BILLING_BASIS

    order = None if entry is None else entry.params.get("training_order")
    if (not isinstance(order, dict) or entry.task_id != config.task_id or entry.status != "active"
            or entry.admission != "open" or entry.profile_id != config.profile_id
            or order.get("order_sha256") != config.order_sha256
            or not isinstance(order.get("quote_id"), str)
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9-]{0,79}", order["quote_id"])
            or not isinstance(order.get("quote_sha256"), str)
            or not s._HEX.fullmatch(order["quote_sha256"])
            or order.get("billing_basis") != BILLING_BASIS):
        raise ValueError("local registry does not bind this exact active order")
    deadline = order.get("deadline_at")
    if (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
            or not math.isfinite(deadline) or deadline <= now):
        raise ValueError("order deadline is unavailable or expired")
    for name in ("max_steps", "max_tokens"):
        if type(order.get(name)) is not int or not 1 <= order[name] <= 2**53 - 1:
            raise ValueError("qualification needs explicit bounded steps and tokens")
    price = order.get("max_price_usd_per_mtoken")
    if isinstance(price, bool) or not isinstance(price, (int, float)) or not math.isfinite(price) or price <= 0:
        raise ValueError("qualification needs an explicit finite token ceiling")
    return {
        "max_steps": order["max_steps"], "max_tokens": order["max_tokens"],
        "max_price_usd_per_mtoken": price, "order_deadline_at": deadline,
        "effective_runtime_seconds": min(config.max_runtime_seconds, deadline - now),
        "start_seconds": config.start_seconds, "drain_seconds": config.drain_seconds,
        "stop_seconds": config.stop_seconds,
        "failure_cleanup_extra_stop_seconds": config.stop_seconds,
        "step_and_token_limits_close_admission": True,
        "nominal_process_budgets_seconds": min(config.max_runtime_seconds, deadline - now)
        + config.drain_seconds + config.stop_seconds,
        "nominal_budget_excludes": ["synchronous_http_deadline_overrun", "post_stop_checkpoint_finalization"],
    }


def _async_settings(config, entry):
    from reliquary.cli.main import _envs_need_grader
    from reliquary.protocol.profiles import entry_profile
    from reliquary.validator.rl_async_service import rl_async_settings

    profile = entry_profile(entry)
    settings = rl_async_settings(entry, float(entry.params["cap"]), run=config.run_id,
                                 max_tokens=lambda env: profile.environments[env].max_new_tokens,
                                 environ={"RELIQUARY_RL_PAY_OUTAGE_FILL": "off",
                                          "RELIQUARY_RL_BATCH_GROUPS": str(entry.params["rl_batch_groups"])})
    if (settings.auction is None or settings.auction.get("training_order") != entry.params["training_order"]
            or settings.pay_outage_fill or not settings.targets):
        raise ValueError("managed async auction is unavailable")
    return {"required_at_startup": _envs_need_grader(entry.env_split or ()), "availability_verified": False}


def _cuda_build():
    """Inspect installed metadata/source literals only; never import torch or initialize CUDA."""
    distribution = metadata.distribution("torch")
    source = Path(distribution.locate_file("torch/version.py"))
    with source.open("rb") as handle:
        raw = handle.read(65537)
    if len(raw) > 65536:
        raise ValueError("unrecognized runtime version metadata")
    versions = []
    for node in ast.parse(raw).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "cuda" for t in node.targets):
            versions.append(ast.literal_eval(node.value))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == "cuda":
            versions.append(ast.literal_eval(node.value))
    if (len(versions) != 1 or not isinstance(versions[0], str)
            or not re.fullmatch(r"\d+\.\d+", versions[0]) or "+cpu" in distribution.version.lower()):
        raise ValueError("this interpreter does not contain a CUDA torch build")
    return {"torch": distribution.version, "torch_cuda_build": versions[0],
            "transformers": metadata.version("transformers"), "cuda_initialized": False}


def preflight(config_file: str | Path, registry_file: str | Path, *, now=None) -> dict:
    """Read local files only. Never mkdir, lock, claim, bind ports, spawn, import CUDA or sign."""
    now = time.time() if now is None else now
    checks = []
    report = {"schema": SCHEMA, "status": "blocked", "paid_admission": False,
              "launch_authorized": False, "qualified": False, "checks": checks,
              "unverified": list(UNVERIFIED)}

    def check(name, operation):
        try:
            result = operation()
        except (OSError, ValueError, KeyError, TypeError, ImportError, SyntaxError):
            checks.append({"name": name, "status": "blocked"})
            return None
        checks.append({"name": name, "status": "passed"})
        return result

    config = check("private_config", lambda: s.SupervisorConfig.read(config_file))
    if config is None:
        return report
    report["binding"] = {"task_id": config.task_id, "order_sha256": config.order_sha256,
                         "profile_id": config.profile_id, "run_id": config.run_id,
                         "model_manifest_sha256": config.model_manifest_sha256,
                         "config_sha256": s._digest(asdict(config))}

    def registry():
        from reliquary.shared.task_registry import parse_registry

        raw = s._private_bytes(Path(registry_file), 1 << 20)
        strict_json_loads(raw)  # Duplicate keys/nonfinite values must not disappear during parsing.
        entry = parse_registry(raw, strict=True).get(config.task_id)
        limits = _order(config, entry, now)
        report["binding"]["registry_snapshot_sha256"] = hashlib.sha256(raw).hexdigest()
        report["limits"] = limits
        return entry

    entry = check("local_registry_order_and_limits", registry)
    if entry is not None:
        grader = check("managed_async_auction_settings", lambda: _async_settings(config, entry))
        if grader is not None:
            report["controller_grader"] = grader
        check("materialized_model_bytes", lambda: s.verify_model(config, entry))
    check("private_fresh_state", lambda: _fresh_state(config))
    check("local_storage_and_signer_material", lambda: _credentials(config))
    check("linux_cgroup_parent", lambda: s.LocalBackend.preflight(config))
    environment = check("cuda_runtime_metadata", _cuda_build)
    if environment is not None:
        report["environment"] = environment
    if all(item["status"] == "passed" for item in checks):
        report["status"] = "local_prerequisites_checked"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True, help="private LOCAL registry export")
    args = parser.parse_args()
    report = preflight(args.config, args.registry)
    print(json.dumps(report, sort_keys=True, allow_nan=False))
    raise SystemExit(0 if report["status"] == "local_prerequisites_checked" else 2)


if __name__ == "__main__":
    main()
