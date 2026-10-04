"""Read-only Affine receipts; signatures attest reports, not GPU execution."""

from __future__ import annotations

import hashlib
import json
import math
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def require(value, reason):
    if not value:
        raise ValueError(reason)


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def manifest_bindings(manifest, env_id=None):
    definitions = {row["env_id"]: row for row in manifest["environments"]}
    require(len(definitions) == len(manifest["environments"]), "duplicate_environment")
    selected = env_id or (next(iter(definitions)) if len(definitions) == 1 else None)
    require(selected in definitions, "requested_environment")
    definition = definitions[selected]
    return dict(bootstrapDigest=hashlib.sha256(canonical(manifest)).hexdigest(),
                checkpointDigest=manifest["checkpoint"]["id"],
                environmentDigest=hashlib.sha256(canonical(definition["spec"])).hexdigest(),
                harnessDigest=hashlib.sha256(canonical(definition["harness"])).hexdigest(),
                sourceBundleDigest=hashlib.sha256(canonical(manifest["source_bundle"])).hexdigest())


def _sampling_policy(manifest):
    if "sampling_contract" not in manifest and "sampling_source_hash" not in manifest:
        return None
    contract = manifest.get("sampling_contract")
    require(isinstance(contract, dict) and set(contract) == {
        "version", "generation", "verification", "max_attempts", "randomness"}, "sampling_contract_fields")
    require(contract["version"] == "forced-inverse-cdf-replay-v1"
            and contract["generation"] == "uncached-eager-inverse-cdf"
            and contract["verification"] == "exact-token-replay"
            and type(contract["max_attempts"]) is int and 2 <= contract["max_attempts"] <= 128,
            "sampling_contract_policy")
    require(all(isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value)
                for value in (contract["randomness"], manifest.get("sampling_source_hash"))),
            "sampling_contract_binding")
    # Public randomness changes each epoch; sampler semantics and source must not.
    return {**{name: value for name, value in contract.items() if name != "randomness"},
            "source_hash": manifest["sampling_source_hash"]}


def _require_sampling_report(manifest, audit):
    if _sampling_policy(manifest) is None:
        return
    contract = manifest["sampling_contract"]
    context = dict(contract=contract, epoch=manifest["epoch"], checkpoint=manifest["checkpoint"]["id"])
    binding = hashlib.sha256(canonical(context)).hexdigest()
    require(audit.get("sampling_assurance") == dict(
        sampling_required=True, scope="fully-audited-rollouts-only", version=contract["version"],
        binding_sha256=binding, verification=contract["verification"], historical_execution_proven=False),
        "audit_sampling_assurance")
    for batch in audit["accepted"]:
        for rollout in batch["rollouts"]:
            attempt = rollout.get("seed")
            require(type(attempt) is int and 0 <= attempt < contract["max_attempts"], "rollout_sampling_attempt")
            require(rollout.get("sampling") == dict(version=contract["version"], binding_sha256=binding,
                                                   attempt=attempt), "rollout_sampling_binding")


def verify_bundle(bundle, signed, expected_bindings=None):
    """Validate native envelopes with the approved upstream signature verifier.

    Byte checks remain separate executor observations. No chain payment is inferred.
    """
    require(bundle.get("schema") == "affine-evidence-bundle/v1", "bundle_schema")
    epoch, miner, expected = (bundle[k] for k in ("epoch_id", "miner_id", "submission_sha256"))
    require(all(re.fullmatch("[0-9a-f]{64}", v or "") for v in (miner, expected, bundle["authority"])), "bundle_identity")
    envelopes = bundle["envelopes"]
    documents = {name: signed(canonical(value), bundle["authority"]) for name, value in envelopes.items()}
    manifest = documents["manifest"]
    require(manifest["epoch"] == epoch and manifest["transport_policy"] == "direct-r2-v1", "manifest_binding")
    require(manifest.get("source_bundle", {}).get("sha256") == bundle["source_sha256"], "source_binding")
    checkpoint = manifest["checkpoint"]
    require(hashlib.sha256(canonical(checkpoint["files"])).hexdigest() == checkpoint["id"], "input_checkpoint_map")
    definitions = {row["env_id"]: row for row in manifest["environments"]}
    selected = bundle.get("env_id") or (next(iter(definitions)) if len(definitions) == 1 else None)
    require(selected in definitions, "requested_environment")
    definition = definitions[selected]
    requested_indices = bundle.get("requested_indices")
    if requested_indices is not None:
        require(isinstance(requested_indices, list) and bool(requested_indices)
                and all(type(i) is int for i in requested_indices)
                and len(requested_indices) == len(set(requested_indices))
                and set(requested_indices) <= set(definition["indices"]), "requested_indices")
    bindings = manifest_bindings(manifest, selected)
    if expected_bindings is not None:
        require(expected_bindings == bindings, "request_profile_binding")
    result = dict(schema="affine-evidence-result/v1", epoch_id=epoch, stage="awaiting_freeze",
                  authoritative_acceptance=False, accepted_batches=0, consumed_batches=0,
                  points=0, payable=manifest.get("payable", False), paid=False,
                  inference_recomputed=False, model_execution_verified=False,
                  checkpoint_bytes_verified=False, handover_verified=False,
                  training_verified=False, qualified_successor=False, bindings=bindings,
                  source_sha256=bundle["source_sha256"], input_checkpoint=checkpoint["id"])
    if "scores" not in documents:
        return result
    scores, challenge = documents["scores"], documents["audit_challenge"]
    require(scores["epoch_id"] == epoch and scores["checkpoint"] == checkpoint["id"], "score_binding")
    require(scores.get("payable") is manifest.get("payable"), "payable_binding")
    require(scores["finalized_at"] >= manifest["deadline"] and challenge["generated_after_freeze_at"] >= manifest["deadline"], "early_freeze")
    require(challenge["receipts"] == scores["receipts"], "frozen_receipt_binding")
    receipt = scores["receipts"].get(miner)
    if receipt is None:
        result["stage"] = "not_submitted"
        return result
    require(receipt["sha256"] == expected and manifest["start"] <= receipt["received_at"] < manifest["deadline"], "submission_binding")
    audit = documents["audit"]
    require(audit["epoch"] == epoch and audit["submission_sha256"] == expected, "audit_binding")
    require("audit_seed" not in audit or audit["audit_seed"] == challenge["seed"], "audit_seed_binding")
    _require_sampling_report(manifest, audit)
    accepted = audit["accepted"]
    pair_hashes, batch_keys = {}, set()
    for batch in accepted:
        key = (batch["env_id"], batch["index"])
        require(key not in batch_keys and batch["epoch"] == epoch and batch["checkpoint"] == checkpoint["id"], "accepted_batch_binding")
        batch_keys.add(key)
        definition = definitions.get(key[0])
        require(definition is not None and type(key[1]) is int and key[1] in definition["indices"] and batch.get("sample_index", key[1]) == key[1], "accepted_task_binding")
        require(key[0] == selected and (requested_indices is None or key[1] in requested_indices), "accepted_request_binding")
        require(batch.get("environment_version") == definition["spec"]["version"], "accepted_environment_version")
        outcomes = [row for row in audit["outcomes"] if row.get("env_id") == key[0] and row.get("index") == key[1]]
        require(len(outcomes) == 1 and outcomes[0].get("valid") is True and outcomes[0].get("fully_audited") is True, "accepted_full_audit")
        positive = [r for r in batch["rollouts"] if r.get("classification") == "positive"]
        negative = [r for r in batch["rollouts"] if r.get("classification") == "negative"]
        require(len(positive) == manifest["K"] == 1 and len(negative) == manifest["L"] == 1, "accepted_pair_count")
        for rollout in positive + negative:
            require(rollout.get("env_id") == key[0] and rollout["index"] == key[1], "accepted_rollout_binding")
        pair_hashes[key] = tuple(hashlib.sha256(canonical(r)).hexdigest() for r in (positive[0], negative[0]))
    points = scores["points"].get(miner, 0)
    require(type(points) is int and 0 <= points <= len(accepted), "invalid_points")
    weights = scores["weights"]
    require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in weights.values()), "invalid_weights")
    require(math.isclose(sum(weights.values()), 1) if scores["total"] else not weights, "unnormalized_weights")
    result.update(stage="accepted" if accepted else "rejected", authoritative_acceptance=bool(accepted),
                  accepted_batches=len(accepted), points=points,
                  provisional_score=bool(scores.get("provisional", False) or scores.get("duplicate_coverage") == "incomplete"))
    if "training" not in documents or documents["training"].get("weights_changed") is not True:
        return result
    training = documents["training"]
    require(training["source_epoch"] == epoch and training.get("input_checkpoint") == checkpoint["id"], "training_input_binding")
    require(type(training["steps"]) is int and training["steps"] > 0 and training["checkpoint"] != checkpoint["id"], "training_update_binding")
    consumed = set()
    covered = manifest.get("training_policy") == "bf16-full-adamw-covered-fixed-reference-v3"
    if covered:
        require(training.get("training_policy") == manifest["training_policy"]
                and training.get("full_model_finetune") is True and training["steps"] <= 32,
                "covered_training_policy")
        require(isinstance(challenge.get("seed"), str) and re.fullmatch("[0-9a-f]{64}", challenge["seed"]),
                "covered_training_seed")
        require(training.get("training_coverage") == dict(
            version="frozen-verified-pairs-v1", epoch=epoch, checkpoint=checkpoint["id"], seed=challenge["seed"],
            receipts_sha256=hashlib.sha256(canonical(scores["receipts"])).hexdigest(),
            generated_after_freeze_at=challenge["generated_after_freeze_at"]), "covered_training_context")
        require(isinstance(training.get("updates"), list) and len(training["updates"]) == training["steps"],
                "covered_training_steps")
    for step, update in enumerate(training.get("updates", []), 1):
        attributions = [update]
        if covered:
            require(isinstance(update, dict) and type(update.get("optimizer_step")) is int and update["optimizer_step"] == step
                    and type(update.get("steps")) is int and update["steps"] == 1
                    and update.get("training_policy") == manifest["training_policy"]
                    and update.get("full_model_finetune") is True, "covered_optimizer_step")
            attributions = update.get("pairs")
            require(isinstance(attributions, list) and bool(attributions)
                    and type(update.get("gradient_pairs")) is int and update["gradient_pairs"] == len(attributions),
                    "covered_gradient_pairs")
        for pair in attributions:
            if covered:
                require(isinstance(pair, dict) and type(pair.get("optimizer_step")) is int and pair["optimizer_step"] == step
                        and type(pair.get("index")) is int
                        and pair.get("attribution_revision") == "verified-pair-v1", "covered_pair_step")
            key = (pair.get("env_id"), pair.get("index"))
            if key in pair_hashes and pair_hashes[key] == (pair.get("positive_rollout_sha256"), pair.get("negative_rollout_sha256")):
                consumed.add(key)
    result.update(stage="training_reported", output_checkpoint=training["checkpoint"],
                  consumed_batches=len(consumed), training_steps=training["steps"])
    if "checkpoint_descriptor" not in documents:
        return result
    output = documents["checkpoint_descriptor"]
    require(output["id"] == training["checkpoint"] == hashlib.sha256(canonical(output["files"])).hexdigest(), "output_checkpoint_map")
    old_weights = {k: v for k, v in checkpoint["files"].items() if k.endswith(".safetensors")}
    new_weights = {k: v for k, v in output["files"].items() if k.endswith(".safetensors")}
    require(old_weights and new_weights and old_weights != new_weights, "unchanged_weights")
    checks = bundle.get("byte_checks", {})
    byte_verified = checks.get("frozen_sha256") == expected and checks.get("checkpoint_files") == output["files"] and checks.get("source_sha256") == bundle["source_sha256"]
    result.update(stage="checkpoint_published", checkpoint_bytes_verified=byte_verified,
                  training_verified=bool(byte_verified and consumed))
    if "next_manifest" in documents:
        successor = documents["next_manifest"]
        require(successor["epoch"] != epoch and successor["start"] >= manifest["deadline"], "successor_epoch_binding")
        require(successor["checkpoint"]["id"] == output["id"] and successor["checkpoint"]["files"] == output["files"], "successor_checkpoint_binding")
        result.update(stage="handover_reported", handover_verified=bool(byte_verified and consumed))
        if byte_verified and consumed:
            result["stage"] = "cycle_verified"
            # A completed epoch can qualify its successor's protocol continuity;
            # it does not establish model quality or a changed runtime's fitness.
            try:
                next_bindings = manifest_bindings(successor, selected)
                sampling_compatible = _sampling_policy(manifest) == _sampling_policy(successor)
            except (KeyError, TypeError, ValueError):
                next_bindings = None
            policies = ("runtime_profile", "model_runtime_revision", "numerical_policy", "backend_profile",
                        "model_id", "artifact_policy", "audit_policy", "training_policy",
                        "environment_revision", "harness_source_hash", "tokenizer_binding",
                        "K", "L", "max_batches", "transport_policy")
            if next_bindings is not None and sampling_compatible \
                    and successor.get("source_bundle", {}).get("sha256") == bundle["source_sha256"] \
                    and next_bindings["environmentDigest"] == bindings["environmentDigest"] \
                    and next_bindings["harnessDigest"] == bindings["harnessDigest"] \
                    and all(successor.get(name) == manifest.get(name) for name in policies):
                result.update(qualified_successor=True, next_bindings=next_bindings)
    return result


def _worker(config, request):
    from .affine import _directory, validate_config, verify_checkout

    validate_config(config)
    verify_checkout(config)
    _directory(Path(config.state_dir))
    request.update(upstream_checkout=str(config.upstream_checkout), authority=config.authority,
                   current_url=config.current_url, state_dir=str(config.state_dir), env_id=config.env_id,
                   requested_indices=list(config.indices) if config.indices is not None else None)
    try:
        # A new empty cache prefix avoids loading ignored cached bytecode. -B
        # disables writes, while ordinary import resolution reads tracked source.
        with tempfile.TemporaryDirectory(prefix="affine-inspection-") as fresh_cache:
            child = subprocess.run([str(config.python), "-I", "-X", "pycache_prefix=" + fresh_cache,
                                   "-B", str(Path(__file__).resolve()), "--child"],
                                   input=canonical(request), capture_output=True,
                                   timeout=config.process_timeout_seconds)
    except subprocess.TimeoutExpired:
        raise RuntimeError("evidence_check_timeout") from None
    if child.returncode:
        raise RuntimeError("evidence_worker_failed")
    try:
        result = json.loads(child.stdout)
    except (ValueError, UnicodeError):
        raise RuntimeError("evidence_worker_invalid_response") from None
    if result.get("error"):
        raise RuntimeError(result["error"])
    return result


def inspect_bootstrap(config, *, allow_closed=False):
    """Return a PRIVATE authenticated snapshot; callers must redact its contents."""
    return _worker(config, dict(operation="bootstrap", key_file=str(config.key_file) if config.key_file else None,
                                cap_file=str(config.cap_file) if config.cap_file else None,
                                allow_closed=allow_closed))


def inspect_delegation(config):
    """Return PRIVATE native decrypted capability data for a local client key."""
    require(config.key_file is not None and config.cap_file is None, "delegation_local_key_required")
    return _worker(config, dict(operation="delegate", key_file=str(config.key_file)))


def bootstrap_readiness(config, snapshot):
    """Describe the authenticated challenge without publishing native credentials."""
    import time

    manifest = snapshot["manifest"]
    selected = config.env_id or manifest["environments"][0]["env_id"]
    definition = next(row for row in manifest["environments"] if row["env_id"] == selected)
    authorized = definition["indices"]
    require(isinstance(authorized, list) and 0 < len(authorized) <= 10000
            and all(type(index) is int and index >= 0 for index in authorized)
            and len(set(authorized)) == len(authorized), "bootstrap_task_pool")
    maximum = manifest.get("max_batches", 4)
    require(type(maximum) is int and 0 < maximum <= 10000, "bootstrap_batch_limit")
    sampling = _sampling_policy(manifest)
    attempts = sampling["max_attempts"] if sampling else 128
    budget = snapshot["artifact_budget"]
    return dict(schema="affine-readiness/v1", bootstrap_verified=True,
                epoch_open=manifest["deadline"] > time.time(), deadline_epoch=manifest["deadline"],
                credential_mode="delegated_capability" if config.cap_file else "local_key",
                bindings=snapshot["bindings"], hardware_qualified=False, paid=False,
                uploaded=None, accepted=None, trained=None,
                constraints=dict(env_id=selected, authorized_indices=definition["indices"],
                                 requested_indices=list(config.indices) if config.indices is not None else None,
                                 K=manifest["K"], L=manifest["L"], max_batches=maximum,
                                 effective_max_batches=min(config.max_batches, maximum,
                                                           len(config.indices or definition["indices"])),
                                 signed_max_attempts=sampling["max_attempts"] if sampling else None,
                                 search_budget=min(config.search_budget, attempts),
                                 artifact_budget=dict(compressed_bytes=budget["compressed_bytes"],
                                                      raw_bytes=budget["raw_bytes"],
                                                      tensor_rows_per_array=budget["tensor_rows"]),
                                 harness=definition["harness"], audit_policy=manifest.get("audit_policy", {"mode": "full"}),
                                 numerical_policy=manifest.get("numerical_policy"),
                                 backend_profile=manifest.get("backend_profile")))


def inspect_evidence(config, epoch_id, miner_id, submission_sha256=None, expected_bindings=None):
    """Inspect owner-private native history without importing its ML runtime."""
    from .affine import validate_config
    validate_config(config)
    require(isinstance(epoch_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,200}", epoch_id) and ".." not in epoch_id, "epoch_identifier")
    require(re.fullmatch("[0-9a-f]{64}", miner_id or ""), "miner_identifier")
    if submission_sha256 is None:
        local = config.state_dir / f"{epoch_id}-{miner_id}.zip"
        require(local.is_file() and not local.is_symlink(), "local_submission_required")
        submission_sha256 = digest_file(local)
    require(re.fullmatch("[0-9a-f]{64}", submission_sha256 or ""), "submission_digest")
    return _worker(config, dict(epoch_id=epoch_id, miner_id=miner_id, submission_sha256=submission_sha256,
                                expected_bindings=expected_bindings))


def _inspect(request):
    """Only called inside the isolated, explicitly pinned upstream interpreter."""
    import tempfile
    import zipfile

    sys.path.insert(0, request["upstream_checkout"])
    from subnet.source_bootstrap import JSON_LIMIT, COMPRESSED_LIMIT, canonical as native_json, download, r2_url, signed
    from subnet.artifact_budget import for_manifest
    import requests

    authority = request["authority"]
    root = Path(request["state_dir"]) / "evidence"
    require(not root.is_symlink(), "evidence_directory_symlink")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    root.chmod(0o700)
    envelopes = {}

    def document(url):
        body = download(r2_url(url), JSON_LIMIT)
        signed(body, authority)
        return json.loads(body)

    def stream(url, limit, destination=None):
        digest, size = hashlib.sha256(), 0
        with requests.get(r2_url(url), stream=True, timeout=180, allow_redirects=False,
                          headers={"Accept-Encoding": "identity"}) as response:
            require(not 300 <= response.status_code < 400, "artifact_redirect")
            response.raise_for_status()
            require(response.headers.get("Content-Encoding", "identity") == "identity", "artifact_encoding")
            length = response.headers.get("Content-Length")
            require(length is None or length.isdecimal() and int(length) <= limit, "artifact_size")
            for chunk in response.iter_content(1024 * 1024):
                size += len(chunk)
                require(size <= limit, "artifact_size")
                digest.update(chunk)
                if destination is not None:
                    destination.write(chunk)
            require(length is None or size == int(length), "artifact_truncated")
        return digest.hexdigest(), size

    current_pointer_envelope = document(request["current_url"])
    current = signed(native_json(current_pointer_envelope), authority)
    require(current.get("transport_policy") == "direct-r2-v1", "discovery_transport")
    current_envelope = document(current["manifest_url"])
    current_manifest = signed(native_json(current_envelope), authority)
    require(current_manifest["epoch"] == current["epoch"], "discovery_epoch")
    if request.get("operation") in ("bootstrap", "delegate"):
        import time
        deadline = current_manifest.get("deadline")
        require(current_manifest.get("transport_policy") == "direct-r2-v1"
                and type(deadline) in (int, float) and math.isfinite(deadline), "bootstrap_deadline")
        require(request.get("allow_closed") or deadline > time.time(), "bootstrap_closed")
        if request.get("key_file"):
            from subnet.storage import Identity
            key_path = Path(request["key_file"])
            require(key_path.stat().st_size <= 128, "local_identity_key_size")
            seed = key_path.read_text().strip()
            require(re.fullmatch("[0-9a-fA-F]{64}", seed), "local_identity_key_format")
            identity = Identity(bytes.fromhex(seed))
            miner = identity.id
            capability = None
        else:
            cap_path = Path(request["cap_file"])
            require(cap_path.stat().st_size <= 65536, "delegated_capability_size")
            capability = json.loads(cap_path.read_text())
            require(isinstance(capability, dict) and capability.get("epoch") == current_manifest["epoch"], "delegated_epoch")
            miner = capability["identity"]
        require(re.fullmatch("[0-9a-f]{64}", miner or "") and miner in current_manifest["capabilities"], "bootstrap_identity")
        if request.get("operation") == "delegate":
            require(request.get("key_file"), "delegation_local_key_required")
            decrypted = identity.decrypt(current_manifest["capabilities"][miner])
            require(isinstance(decrypted, dict), "delegated_capability_fields")
            capability = {name: decrypted.get(name) for name in ("transport", "put_url", "headers", "deadline")}
            capability.update(epoch=current_manifest["epoch"], identity=miner)
            require(len(canonical(capability)) <= 65536, "delegated_capability_size")
        if capability is not None:
            require(capability.get("transport") == "direct-r2-v1"
                    and type(capability.get("deadline")) is int and capability["deadline"] == deadline
                    and capability.get("headers") == {"Content-Type": "application/octet-stream"},
                    "delegated_transport")
            from urllib.parse import urlsplit, unquote
            route = r2_url(capability.get("put_url"))
            require(unquote(urlsplit(route).path).endswith(
                "/private/" + current_manifest["epoch"] + "/staging/" + str(miner) + ".zip"),
                "delegated_upload_object")
        bindings = manifest_bindings(current_manifest, request.get("env_id"))
        selected = request.get("env_id") or current_manifest["environments"][0]["env_id"]
        definition = next(row for row in current_manifest["environments"] if row["env_id"] == selected)
        indices = request.get("requested_indices")
        require(indices is None or set(indices) <= set(definition["indices"]), "bootstrap_task_scope")
        snapshot = dict(schema="affine-bootstrap-snapshot/v1", manifest=current_manifest, miner_id=miner,
                        bindings=bindings, current_envelope=current_pointer_envelope,
                        manifest_envelope=current_envelope, artifact_budget=for_manifest(current_manifest),
                        paid=False, hardware_qualified=False)
        if request.get("operation") == "delegate":
            snapshot["capability"] = capability
        return snapshot
    history_envelope = document(current["history_url"])
    history = signed(native_json(history_envelope), authority)
    require(history.get("authority") == authority and history.get("version") == 1, "history_authority")
    rows = [r for r in history["epochs"] if r["epoch_id"] == request["epoch_id"]]
    require(len(rows) <= 1, "duplicate_history_epoch")
    if not rows:
        return dict(schema="affine-evidence-result/v1", stage="awaiting_history", epoch_id=request["epoch_id"],
                    authoritative_acceptance=False, paid=False, inference_recomputed=False,
                    model_execution_verified=False, handover_verified=False, qualified_successor=False)
    row = rows[0]
    envelopes["history"] = history_envelope
    envelopes["manifest"] = document(row["objects"]["manifest"])
    manifest = signed(native_json(envelopes["manifest"]), authority)
    source = row["source_bundle"]
    require(source.get("binding") == "epoch-signed" and source["sha256"] == manifest["source_bundle"]["sha256"] and source["size"] == manifest["source_bundle"]["size"], "historical_source_binding")
    source_hash, source_size = stream(source["read_url"], COMPRESSED_LIMIT)
    require(source_hash == source["sha256"] and source_size == source["size"], "source_bytes")
    envelopes["scores"] = document(row["objects"]["scores"])
    envelopes["audit_challenge"] = document(row["objects"]["audit-challenge"])
    scores = signed(native_json(envelopes["scores"]), authority)
    miner = request["miner_id"]
    checks = dict(source_sha256=source_hash)
    frozen_batches = None
    if miner in scores["receipts"]:
        route = row["frozen"][miner]
        require(route["sha256"] == scores["receipts"][miner]["sha256"] == request["submission_sha256"], "frozen_route_binding")
        budget = for_manifest(manifest)
        with tempfile.TemporaryFile(dir=root) as artifact:
            digest, size = stream(route["url"], budget["compressed_bytes"], artifact)
            require(digest == route["sha256"] and (route.get("size") is None or size == route["size"]), "frozen_bytes")
            artifact.seek(0)
            with zipfile.ZipFile(artifact) as archive:
                names = archive.namelist()
                require(len(names) == len(set(names)) and len(names) <= 4096 and all("/" not in n and ".." not in n for n in names), "frozen_archive_names")
                require(sum(i.file_size for i in archive.infolist()) <= budget["raw_bytes"] and archive.getinfo("manifest.json").file_size <= 2_000_000, "frozen_archive_budget")
                records = json.loads(archive.read("manifest.json"))
                require(isinstance(records, list) and len(records) <= manifest["max_batches"], "frozen_batch_budget")
                frozen_batches = [record["batch"] for record in records]
        checks["frozen_sha256"] = digest
        envelopes["audit"] = document(row["audits"][miner])
        audit = signed(native_json(envelopes["audit"]), authority)
        if manifest.get("sampling_contract") is not None:
            from subnet.forced_sampling import require_report
            require_report(manifest, audit)
        require(all(any(batch == candidate for candidate in frozen_batches) for batch in audit["accepted"]), "frozen_accepted_binding")
    try:
        envelopes["training"] = document(row["objects"]["training"])
    except requests.HTTPError as error:
        if error.response.status_code != 404:
            raise
    output = row.get("trained_checkpoint")
    if output:
        envelopes["checkpoint_descriptor"] = document(output["descriptor_url"])
        descriptor = signed(native_json(envelopes["checkpoint_descriptor"]), authority)
        require(output["id"] == descriptor["id"] and output["files"] == descriptor["files"], "history_checkpoint_binding")
        require(set(output["read_urls"]) == set(output["files"]), "checkpoint_routes")
        observed = {}
        for name, expected in output["files"].items():
            require(Path(name).name == name and name not in (".", ".."), "checkpoint_filename")
            observed[name], _ = stream(output["read_urls"][name], 20_000_000_000)
            require(observed[name] == expected, "checkpoint_bytes")
        checks["checkpoint_files"] = observed
    following = []
    position = next(i for i, candidate in enumerate(history["epochs"]) if candidate["epoch_id"] == request["epoch_id"])
    for candidate in history["epochs"][position + 1:position + 2]:
        envelope = document(candidate["objects"]["manifest"])
        value = signed(native_json(envelope), authority)
        if value["start"] >= manifest["deadline"]:
            following.append((value["start"], envelope))
    if current_manifest["start"] >= manifest["deadline"] and current_manifest["epoch"] != manifest["epoch"]:
        following.append((current_manifest["start"], current_envelope))
    if following:
        envelopes["next_manifest"] = min(following, key=lambda row: row[0])[1]
    bundle = dict(schema="affine-evidence-bundle/v1", authority=authority, epoch_id=request["epoch_id"],
                  miner_id=miner, submission_sha256=request["submission_sha256"], source_sha256=source_hash,
                  env_id=request.get("env_id"), requested_indices=request.get("requested_indices"),
                  envelopes=envelopes, byte_checks=checks)
    result = verify_bundle(bundle, signed, request.get("expected_bindings"))
    data = canonical(bundle)
    require(len(data) <= 1024 * 1024, "signed_bundle_budget")
    path = root / (hashlib.sha256(data).hexdigest() + ".json")
    if path.exists():
        require(not path.is_symlink() and path.read_bytes() == data, "existing_bundle_integrity")
    else:
        import os
        with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as output:
            output.write(data)
    result.update(bundle_path=str(path), bundle_sha256=hashlib.sha256(data).hexdigest(),
                  native_artifact_digests={name: hashlib.sha256(canonical(value)).hexdigest()
                                          for name, value in envelopes.items()},
                  assurance="authority-signed reports with local source/frozen/checkpoint byte checks")
    return result


if __name__ == "__main__" and sys.argv[1:] == ["--child"]:
    try:
        print(json.dumps(_inspect(json.load(sys.stdin))))
    except Exception as error:
        # Native HTTP errors can contain private capability URLs; never relay them.
        reason = str(error) if re.fullmatch("[a-z_]{1,64}", str(error)) else type(error).__name__
        print(json.dumps({"error": "evidence_check_failed:" + reason}))
