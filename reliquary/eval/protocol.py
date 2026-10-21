"""Content identities shared by set publication, execution and grading."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

SET_PROTOCOL = "reliquary/evaluation-protocol/v1"
REQUEST_PROTOCOL = "reliquary/evaluation-request/v1"
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
HARNESSES = {"single-turn": ("single_turn", 1), "telecom-episode-v1": ("episode", 40),
             "signed-bash-v1": ("signed_episode", 40)}

# Declared source policy, not a measured capacity or task qualification.
SIGNED_SOURCE_POLICIES = {
    ("reliquary-swe==0.3.0", "eval"): "9456020aa9c05c193a3f3e1b9ec38db2646bcbf0d1e8cec5664b432c239a5640",
    ("reliquary-terminal==0.3.1", "tmax"): "8e5607fd25268c704becc4fb0bcdc48a983e35ef94530f90aef40c44c1194065",
    ("reliquary-terminal==0.3.1", "tmax_sft"): "8e5607fd25268c704becc4fb0bcdc48a983e35ef94530f90aef40c44c1194065",
    ("reliquary-terminal==0.3.1", "tmax_rl"): "8e5607fd25268c704becc4fb0bcdc48a983e35ef94530f90aef40c44c1194065",
}


def canonical(document) -> bytes:
    return json.dumps(document, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def digest(document) -> str:
    return hashlib.sha256(canonical(document)).hexdigest()


def _same(a, b) -> bool:
    if any(isinstance(value, float) and not math.isfinite(value) for value in (a, b)):
        return False
    if isinstance(a, bool) or isinstance(b, bool):
        return type(a) is type(b) and a == b
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(_same(v, b[k]) for k, v in a.items())
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    return a == b


def _envelope(document: dict) -> dict:
    raw = document.get("protocol_json")
    if not isinstance(raw, str):
        raise ValueError("protocol_json producer bytes are required")

    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate protocol key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("non-finite protocol number")

    parsed = json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)
    if not isinstance(parsed, dict) or not _same(parsed, document.get("protocol")) \
            or hashlib.sha256(raw.encode("utf-8")).hexdigest() != document.get("protocol_sha256"):
        raise ValueError("protocol digest/body mismatch")
    return parsed


def grader_sha256() -> str:
    root = Path(__file__).parent
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("grading.py", "verifiers_source.py", "metrics.py", "protocol.py",
                                "../corpus/export.py", "../environment/registry.py",
                                "../environment/agentic/external.py", "../validator/admission.py",
                                "../environment/stdio_program.py", "agentic.py", "signed_agentic.py",
                                "../environment/agentic/runner.py", "../environment/agentic/types.py")})


def harness_sha256(harness: str = "single-turn") -> str:
    root = Path(__file__).parent
    extra = {"single-turn": (), "telecom-episode-v1": ("agentic.py", "../environment/agentic/runner.py",
                                                     "../environment/agentic/types.py"),
             "signed-bash-v1": ("signed_agentic.py", "../sandbox/evaluation_engagements.py")}[harness]
    files = {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
             for name in ("runner.py", "model_process.py", "platform_client.py", "protocol.py", *extra)}
    if harness == "signed-bash-v1":
        from reliquary.eval.verifiers_source import installed_artifacts
        files["signed_sandbox_and_verifiers"] = installed_artifacts("reliquary-sandbox")
    return digest(files)


def renderer_sha256() -> str:
    from importlib.metadata import distribution

    installed = distribution("renderers")
    if installed.version != "0.1.11":
        raise ValueError("signed episodes require the pinned renderer distribution")
    files = {str(path): hashlib.sha256(Path(path.locate()).read_bytes()).hexdigest()
             for path in installed.files or () if path.suffix == ".py" or str(path).endswith("direct_url.json")}
    if not files:
        raise ValueError("renderer artifact identity is unavailable")
    return digest(files)


def telecom_renderer_sha256() -> str:
    root = Path(__file__).parents[1] / "environment/agentic"
    return digest({name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                   for name in ("chatml.py", "renderer.py", "renderers.py")})


def public_taskset(taskset: dict | None) -> dict | None:
    if taskset is None:
        return None
    result = {key: taskset[key] for key in ("id", "package", "package_version", "verifiers_version",
                                           "artifact_sha256") if key in taskset}
    result["args_sha256"] = (digest(taskset["args"]) if "args" in taskset
                              else taskset["args_sha256"])
    return result


def _episode_identity(episode) -> bool:
    return (isinstance(episode, dict) and set(episode) == {
        "renderer_id", "renderer_sha256", "tools_sha256", "runtime_image_sha256"}
        and isinstance(episode["renderer_id"], str) and 0 < len(episode["renderer_id"]) <= 512
        and all(isinstance(episode[key], str) and SHA256.fullmatch(episode[key])
                for key in ("renderer_sha256", "tools_sha256"))
        and (episode["runtime_image_sha256"] is None or isinstance(episode["runtime_image_sha256"], str)
             and SHA256.fullmatch(episode["runtime_image_sha256"])))


def validate_execution(document: dict, bindings: dict, *, request: dict | None = None) -> None:
    """Check the frozen harness at execution, never at a later grading read."""
    if not isinstance(document, dict) or document.keys() != bindings.keys():
        raise ValueError("execution requires each frozen set protocol")
    for set_id, envelope in document.items():
        body = _envelope(envelope)
        if body.get("schema") != SET_PROTOCOL or body.get("set", {}).get("set_id") != set_id \
                or envelope["protocol_sha256"] != bindings[set_id]:
            raise ValueError("execution set protocol binding changed")
        harness = body.get("harness", {})
        identity = harness.get("id")
        turns = harness.get("max_turns")
        if (identity not in HARNESSES or type(turns) is not int
                or not 1 <= turns <= HARNESSES[identity][1]
                or (identity != "signed-bash-v1" and turns != HARNESSES[identity][1])
                or harness != {"id": identity, "max_turns": turns, "sha256": harness_sha256(identity)}):
            raise ValueError("set protocol harness drift")
        if request is not None and (request["harness"] != identity
                                    or request["max_turns"] > harness["max_turns"]):
            raise ValueError("request and frozen harness differ")
        if identity != "single-turn" and not _episode_identity(body.get("source", {}).get("episode")):
            raise ValueError("invalid episode renderer/tool/runtime identity")
        if identity == "signed-bash-v1" and body["source"]["episode"]["renderer_sha256"] != renderer_sha256():
            raise ValueError("set protocol renderer drift")
        if identity == "telecom-episode-v1" and body["source"]["episode"]["renderer_sha256"] != telecom_renderer_sha256():
            raise ValueError("set protocol renderer drift")


def seal_set(card: dict, *, reviewed: dict | None = None) -> dict:
    """Bind measured files and implementation to optional reviewed data rights/pins.

    Missing source/rights evidence stays null; a sealed digest is identity, not
    a qualification or an assertion that a source is held out.
    """
    reviewed = dict(reviewed or {})
    if set(reviewed) - {"data", "licenses", "disjointness"}:
        raise ValueError("protocol accepts data, licenses and disjointness only")
    data, licenses = reviewed.get("data"), reviewed.get("licenses", [])
    if data is not None and (not isinstance(data, dict) or not data
                             or not all(isinstance(k, str) and isinstance(v, str)
                                        and SHA256.fullmatch(v) for k, v in data.items())):
        raise ValueError("protocol data must map artifact names to SHA-256 digests")
    if not isinstance(licenses, list) or any(
            not isinstance(row, dict) or set(row) != {"id", "sha256"}
            or not isinstance(row["id"], str) or not row["id"]
            or not isinstance(row["sha256"], str) or not SHA256.fullmatch(row["sha256"])
            for row in licenses):
        raise ValueError("protocol licenses require id and exact license-body sha256")
    disjointness = reviewed.get("disjointness", card.get("disjointness"))
    if "disjointness" in reviewed and (
            not isinstance(disjointness, dict)
            or set(disjointness) != {"disjoint", "evidence_sha256"}
            or type(disjointness.get("disjoint")) is not bool
            or not isinstance(disjointness.get("evidence_sha256"), str)
            or not SHA256.fullmatch(disjointness["evidence_sha256"])):
        raise ValueError("reviewed disjointness needs a boolean and evidence_sha256")
    if "disjointness" in reviewed:
        card = {**card, "disjointness": disjointness}
    interaction = card.get("interaction", "single_turn")
    harness, turns = next(((key, limit) for key, (mode, limit) in HARNESSES.items()
                           if mode == interaction), (None, None))
    if harness is None:
        raise ValueError("unsupported set interaction")
    turns = card.get("harness_max_turns", turns)
    if type(turns) is not int or not 1 <= turns <= HARNESSES[harness][1]:
        raise ValueError("set harness exceeds its supported turn bound")
    episode = card.get("episode")
    if interaction != "single_turn" and not _episode_identity(episode):
        raise ValueError("episode source requires its exact renderer/tool/runtime identity")
    protocol = {
        "schema": SET_PROTOCOL,
        "set": {k: card[k] for k in ("set_id", "count", "prompts_sha256", "grading_sha256")},
        "source": {"id": card["source"], "kind": card.get("source_kind", "catalog"),
                   "split": card["split"], "taskset": public_taskset(card.get("taskset")),
                   "environment_manifest_sha256": card.get("environment_manifest_sha256"),
                   "data": data, "licenses": licenses, "grader_sha256": grader_sha256(),
                   **({"episode": episode} if episode is not None else {})},
        "disjointness": disjointness,
        "harness": {"id": harness, "max_turns": turns, "sha256": harness_sha256(harness)},
    }
    raw = canonical(protocol)
    return {**card, "protocol": protocol, "protocol_json": raw.decode("utf-8"),
            "protocol_sha256": hashlib.sha256(raw).hexdigest()}


def validate_set(card: dict, *, current_grader: bool = False) -> str | None:
    """Legacy cards remain readable, but never acquire a protocol retrospectively."""
    body, sha = card.get("protocol"), card.get("protocol_sha256")
    if body is None and sha is None:
        return None
    body = _envelope(card)
    if body.get("schema") != SET_PROTOCOL:
        raise ValueError("set protocol schema mismatch")
    if body.get("set") != {k: card[k] for k in
                           ("set_id", "count", "prompts_sha256", "grading_sha256")}:
        raise ValueError("set protocol files changed")
    source = body.get("source", {})
    if (source.get("id") != card["source"] or source.get("split") != card["split"]
            or source.get("kind") != card.get("source_kind", "catalog")
            or source.get("environment_manifest_sha256") != card.get("environment_manifest_sha256")
            or source.get("episode") != card.get("episode")
            or source.get("taskset") != public_taskset(card.get("taskset"))
            or body.get("disjointness") != card.get("disjointness")):
        raise ValueError("set protocol source changed")
    if current_grader and source.get("grader_sha256") != grader_sha256():
        raise ValueError("set protocol grader drift")
    if card.get("interaction") == "signed_episode":
        private = card.get("signed_episode", {})
        policy = SIGNED_SOURCE_POLICIES.get((private.get("package"), card["split"]))
        if (policy is None or private.get("source_policy_sha256") != policy
                or (source.get("data") or {}).get("environment.toml") != policy
                or body.get("harness", {}).get("max_turns") != card.get("harness_max_turns")
                or not 1 <= card.get("harness_max_turns", 0) <= 40):
            raise ValueError("signed source policy changed")
    return sha


def validate_request(document: dict, *, bindings: dict | None = None) -> str | None:
    body, sha = document.get("protocol"), document.get("protocol_sha256")
    if body is None and sha is None and not document.get("protocol_bindings"):
        return None
    body = _envelope(document)
    if body.get("schema") != REQUEST_PROTOCOL:
        raise ValueError("request protocol schema mismatch")
    if set(body) != {"schema", "sets", "protocol_bindings", "sampling", "thinking",
                     "max_new_tokens", "harness", "max_turns", "max_episode_tokens",
                     "episode_timeout_seconds", "interaction"}:
        raise ValueError("unknown or missing request protocol field")
    if not isinstance(body.get("protocol_bindings"), dict) or not body["protocol_bindings"] \
            or any(not isinstance(value, str) or not SHA256.fullmatch(value)
                   for value in body["protocol_bindings"].values()):
        raise ValueError("request requires frozen set protocol hashes")
    if body.get("protocol_bindings") != document.get("protocol_bindings"):
        raise ValueError("request protocol bindings changed")
    if bindings is not None and body.get("protocol_bindings") != bindings:
        raise ValueError("request does not bind the actual frozen set protocols")
    if "model" in body or "model_access" in body:
        raise ValueError("evaluation protocol must be model-independent")
    for key, value in (("sampling", document.get("sampling")),
                       ("thinking", document.get("thinking")),
                       ("max_new_tokens", document.get("max_new_tokens"))):
        if body.get(key) != value:
            raise ValueError(f"request protocol {key} changed")
    harness = HARNESSES.get(body.get("harness"))
    if (harness is None or body.get("interaction") != harness[0]
            or type(body.get("max_turns")) is not int or not 1 <= body["max_turns"] <= harness[1]
            or type(body.get("max_episode_tokens")) is not int
            or not 1 <= body["max_episode_tokens"] <= 65536
            or type(body.get("episode_timeout_seconds")) is not int
            or not 1 <= body["episode_timeout_seconds"] <= 3600):
        raise ValueError("unsupported evaluation interaction or harness")
    if type(body.get("max_new_tokens")) is not int or not 1 <= body["max_new_tokens"] < body["max_episode_tokens"]:
        raise ValueError("a model turn must leave room for its frozen transcript")
    if body["harness"] == "telecom-episode-v1" and body["thinking"] is not True:
        raise ValueError("the telecom harness requires its thinking renderer")
    return sha
