"""Issue owned, nonpayable native diagnostic jobs for pinned public checkpoints.

The original evaluator owns all science. This operator entry point only binds a
compatible checkpoint, fresh bounded lifetime, and a distinct signing authority.
It does not grant public Affine authority or qualify a checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from types import SimpleNamespace

from .affine import _private_file
from .affine_epoch import canonical
from .affine_native_evaluation import _admit, REVIEWED_REVISION
from .affine_scoring import require, read_json, write_json

POLICY_ID = "mimo-diagnostic4-v1"
SUITE = [dict(env_id="affine_math", indices=[6903, 3689, 47, 3166],
              name="diagnostic4", repeats=1, seed=20261002)]
MAX_CHECKPOINT_BYTES = 20 * 1024**3
MAX_METADATA_BYTES = 32 * 1024**2
HEX = re.compile(r"[0-9a-f]{64}")


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def _safe_name(name):
    return (isinstance(name, str) and bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,199}", name))
            and ".." not in name and name.endswith((".json", ".safetensors", ".txt", ".model", ".jinja", ".tiktoken")))


def _lfs_sha(item):
    lfs = item.lfs
    value = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
    require(isinstance(value, str) and HEX.fullmatch(value), "safetensors require HF SHA256 metadata")
    return value


def discover_checkpoint(repository, revision, baseline, baseline_config, *, api=None, download=None):
    """Read small metadata; the existing materializer later verifies every weight byte."""
    require(isinstance(repository, str) and re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}/[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", repository)
        and ".." not in repository and "--" not in repository and not repository.endswith(".git")
        and isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
        "public HF repository and immutable full revision required")
    if api is None or download is None:
        from huggingface_hub import HfApi, hf_hub_download
        api = api or HfApi(token=False)
        download = download or hf_hub_download
    info = api.model_info(repository, revision=revision, files_metadata=True, timeout=30)
    require(info.sha == revision and info.private is False and info.gated is False,
            "exact public ungated HF revision required")
    siblings = {item.rfilename: item for item in info.siblings or []}
    require(len(siblings) == len(info.siblings or []), "duplicate HF filenames")
    inventory, metadata_bytes = {}, 0

    def small(name):
        nonlocal metadata_bytes
        require(_safe_name(name) and name in siblings, "required checkpoint metadata is missing")
        size = siblings[name].size
        require(type(size) is int and 0 < size <= MAX_METADATA_BYTES, "bounded checkpoint metadata required")
        metadata_bytes += size
        require(metadata_bytes <= 64 * 1024**2, "checkpoint metadata exceeds byte ceiling")
        path = Path(download(repository, name, revision=revision, token=False))
        require(path.is_file() and path.stat().st_size == size, "HF metadata size differs")
        data = path.read_bytes()
        inventory[name] = dict(sha256=hashlib.sha256(data).hexdigest(), bytes=size)
        return data

    config = json.loads(small("config.json"))
    harmless = {"_name_or_path", "transformers_version"}
    geometry = lambda value: {k: v for k, v in value.items() if k not in harmless}
    require(isinstance(config, dict) and geometry(config) == geometry(baseline_config)
            and config.get("architectures") == ["Qwen3_5ForConditionalGeneration"]
            and not config.get("quantization_config") and not config.get("auto_map"),
            "only full checkpoints with the approved MiMo geometry are supported")
    # Preserve tokenizer, special tokens, processors and the signed chat template.
    for name, expected in sorted(baseline["files"].items()):
        if name == "config.json" or name.endswith(".safetensors") or name == "model.safetensors.index.json":
            continue
        small(name)
        require(inventory[name]["sha256"] == expected, "checkpoint tokenizer or processor differs from approved runtime")
    if "model.safetensors.index.json" in siblings:
        index = json.loads(small("model.safetensors.index.json"))
        mapping = index.get("weight_map") if isinstance(index, dict) else None
        require(isinstance(mapping, dict) and mapping and all(isinstance(k, str) for k in mapping),
                "full safetensors weight map required")
        weights = set(mapping.values())
    else:
        weights = {"model.safetensors"}
    require(1 <= len(weights) <= 64 and all(_safe_name(n) and n.endswith(".safetensors") for n in weights),
            "bounded top-level safetensors checkpoint required")
    for name in sorted(weights):
        require(name in siblings, "safetensors shard missing")
        item = siblings[name]
        require(type(item.size) is int and item.size > 0, "safetensors byte size required")
        inventory[name] = dict(sha256=_lfs_sha(item), bytes=item.size)
    total = sum(item["bytes"] for item in inventory.values())
    require(0 < total <= MAX_CHECKPOINT_BYTES, "checkpoint exceeds pilot byte ceiling")
    files = {name: row["sha256"] for name, row in inventory.items()}
    return dict(checkpoint=dict(id=digest(files), files=files), inventory=inventory,
                repository=repository, revision=revision, weightBytes=sum(inventory[n]["bytes"] for n in weights),
                totalBytes=total, architecture=config["architectures"][0])


def bounded_deadline(now, session_deadline, provider_deadline, max_seconds):
    require(type(now) is int and type(session_deadline) is int and type(provider_deadline) is int,
            "finite window required")
    expires = min(session_deadline, provider_deadline - 600)
    require(1425 <= expires - now <= 86400 and max_seconds == 900,
            "insufficient native execution and finalization window")
    return expires


def issue(*, recipe_path, recipe_sha256, observation_path, baseline_config_path,
          authority_seed_path, receipt_public_key, source, request, output, now=None):
    """Prepare once. An existing output is retained for reconciliation, never overwritten."""
    now = int(time.time()) if now is None else now
    require(type(now) is int, "integer issuance time required")
    for path in (recipe_path, observation_path, baseline_config_path, authority_seed_path):
        _private_file(Path(path))
    recipe_bytes = Path(recipe_path).read_bytes()
    require(HEX.fullmatch(recipe_sha256) and hashlib.sha256(recipe_bytes).hexdigest() == recipe_sha256,
            "approved recipe hash differs")
    previous = json.loads(recipe_bytes)
    old_job, old_manifest, _ = _admit(previous, previous["signer"])
    require(old_manifest.get("payable") is False and old_manifest.get("evaluation_sets") == SUITE
            and old_manifest["environments"][0]["harness"]["max_output_tokens"] == 8192
            and not any(k in old_manifest for k in ("mimo_release_policy", "mimo_checkpoint_authorization", "task_assets")),
            "approved nonpayable four-case diagnostic recipe required")
    baseline_bytes = Path(baseline_config_path).read_bytes()
    require(hashlib.sha256(baseline_bytes).hexdigest() == old_manifest["checkpoint"]["files"]["config.json"],
            "approved base configuration bytes differ")
    candidate = discover_checkpoint(request["repository"], request["revision"], old_manifest["checkpoint"],
                                    json.loads(baseline_bytes))
    require(request.get("policyId") == POLICY_ID and request.get("policyDigest") == hashlib.sha256(POLICY_ID.encode()).hexdigest(),
            "supported diagnostic policy required")
    require(isinstance(request.get("requestId"), str) and re.fullmatch(r"[A-Za-z0-9_-]{1,100}", request["requestId"])
            and isinstance(request.get("ownerUserId"), str) and request["ownerUserId"], "owned request required")
    expires = bounded_deadline(now, request.get("sessionDeadline"), request.get("providerDeadline"),
                               request.get("maxSeconds"))
    observation = read_json(observation_path, 65536)
    require(observation.get("qualification") is False and observation.get("publicAuthorityDelegation") is False
            and observation.get("checkpoint") == old_manifest["checkpoint"]["id"]
            and digest(observation) == old_manifest["sampling_contract"]["calibration"]["report_sha256"],
            "explicit original observational policy required")
    observation = copy.deepcopy(observation)
    observation["checkpoint"] = candidate["checkpoint"]["id"]
    calibration = copy.deepcopy(old_manifest["sampling_contract"]["calibration"])
    calibration.update(checkpoint=candidate["checkpoint"]["id"], report_sha256=digest(observation))
    require(all(calibration[k] == 0 for k in ("cdf_abs_error", "logprob_atol", "toploc_exp_mismatches",
                                            "toploc_mant_err_mean", "toploc_mant_err_median")),
            "only the original zero-tolerance observational policy is supported")
    from nacl.signing import SigningKey
    seed = Path(authority_seed_path).read_bytes()
    require(len(seed) == 32 and isinstance(receipt_public_key, str) and HEX.fullmatch(receipt_public_key),
            "operator authority and separate receipt public key required")
    key = SigningKey(seed)
    identity = SimpleNamespace(id=key.verify_key.encode().hex(), key=key)
    require(identity.id != receipt_public_key, "authority and receipt signers must be distinct")
    source = Path(source).resolve(strict=True)
    observed = {}
    for path in sorted((source / "subnet").rglob("*.py")):
        require(path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(source),
                "ordinary approved source required")
        observed[path.relative_to(source).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    require(observed == old_job["source_files"], "approved original source closure differs")
    for name, module in tuple(sys.modules.items()):
        if name == "subnet" or name.startswith("subnet."):
            filename = getattr(module, "__file__", None)
            require(filename is None or Path(filename).resolve().is_relative_to(source),
                    "original source must run in its own fresh process")
    sys.path.insert(0, str(source))
    try:
        from subnet.mimo_manifest import build
        from subnet.mimo_release_policy import sign
        from subnet.mimo_worker import source_inventory, validate
        require(source_inventory(source) == old_job["source_files"], "approved original source closure differs")
        run_id = "internal-evaluation-" + request["requestId"]
        m = old_manifest
        manifest = build(run_id=run_id, epoch="nonpayable-" + request["requestId"],
            checkpoint=candidate["checkpoint"], source_bundle=m["source_bundle"], environments=m["environments"],
            runtime_spec=m["mimo_runtime"], vocab_size=m["group_transport_policy"]["vocab_size"],
            backend_profile=m["backend_profile"], calibration=calibration, start=now,
            duration=min(3600, expires-now), group_size=m["group_contract"]["group_size"],
            max_batches=m["max_batches"], max_output_tokens=8192,
            randomness=m["sampling_contract"]["randomness"], evaluation_sets=SUITE)
        manifest["deadline"] = expires
        ignored = {"run_id", "epoch", "start", "deadline", "checkpoint", "sampling_contract"}
        require({k:v for k,v in manifest.items() if k not in ignored} ==
                {k:v for k,v in m.items() if k not in ignored}, "frozen native science differs")
        require({k:v for k,v in manifest["sampling_contract"].items() if k != "calibration"} ==
                {k:v for k,v in m["sampling_contract"].items() if k != "calibration"}, "frozen native sampling differs")
        job = dict(schema=1, worker_policy=old_job["worker_policy"], role="evaluate",
            job_id="internal-" + request["requestId"], created_at=now, expires_at=expires,
            manifest=sign(identity, manifest), source_files=old_job["source_files"],
            runtime_versions=old_job["runtime_versions"],
            evaluation=dict(version=old_job["evaluation"]["version"], run_id=run_id, set_name="diagnostic4"))
        envelope = sign(identity, job)
        validate(envelope, identity.id, now, allow_public=False)
        _admit(envelope, identity.id, now=now)
    finally:
        sys.path.remove(str(source))
    root = Path(request["runtimeRoot"])
    require(root.is_absolute() and ".." not in root.parts, "absolute isolated runtime root required")
    output = Path(output)
    output.mkdir(mode=0o700)
    intake = dict(schema="affine-native-evaluation-intake/v1", assurance="native-mimo-evaluation",
        authority=identity.id, jobEnvelope=str(root / "fresh-job/job-envelope.private.json"),
        sourceDirectory=str(Path(request["sourceDirectory"])), sourceRevision=REVIEWED_REVISION,
        checkpointDirectory=str(root / "checkpoint"),
        limits=dict(maxSeconds=900, finalizationReserveSeconds=120, maxLogBytes=16*1024**2,
                    maxOutputBytes=12_000_000, maxWorkspaceBytes=80*1024**3),
        review=dict(sourceApproved=True, checkpointApproved=True, heldoutApproved=True))
    require(Path(intake["sourceDirectory"]).is_absolute(), "absolute approved source path required")
    provenance = dict(schema="owned-native-job-provenance/v1", requestId=request["requestId"],
        ownerUserId=request["ownerUserId"], repository=candidate["repository"], revision=candidate["revision"],
        checkpointId=candidate["checkpoint"]["id"], totalBytes=candidate["totalBytes"],
        jobEnvelopeSha256=digest(envelope), createdAt=now, nativeJobExpiresAt=expires,
        receiptPublicKey=receipt_public_key, policyId=POLICY_ID, affineAuthorization=False,
        samplingCalibrationQualified=False, freshNumericalComparisonQualified=False, customerAccepted=False)
    for name, value in (("job-envelope.private.json", envelope), ("intake.private.json", intake),
                        ("observational-policy.private.json", observation), ("provenance.private.json", provenance)):
        write_json(output / name, value)
    return provenance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "observation", "baseline-config", "authority-seed", "source", "request", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--recipe-sha256", required=True)
    parser.add_argument("--receipt-public-key", required=True)
    args = parser.parse_args(argv)
    _private_file(args.request)
    result = issue(recipe_path=args.recipe, recipe_sha256=args.recipe_sha256,
        observation_path=args.observation, baseline_config_path=args.baseline_config,
        authority_seed_path=args.authority_seed, receipt_public_key=args.receipt_public_key,
        source=args.source, request=read_json(args.request, 65536), output=args.output)
    print(json.dumps({"prepared":True, "checkpointId":result["checkpointId"],
                      "nativeJobExpiresAt":result["nativeJobExpiresAt"], "qualified":False}))


if __name__ == "__main__":
    main()
