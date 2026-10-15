"""The checkpoint index: object layout, signature and download planning.

The trainer writes anchors and deltas to the ``reliquary-public`` bucket and an
unsigned ``index.json`` last. The validator serves it on ``GET /rl/checkpoint``
signed with its hotkey, with presigned URLs. Miners verify the signature and plan
their download here (docs/rl-async-design.md §4.8).

Index shape::

    {"format": "reliquary-ckpt-index/1", "run": str,
     "latest": {"version": int, "sha256": str},
     "anchors": [{"version", "sha256", "files": [{"key", "size", "sha256"}],
                  "model_files": [{"name", "key", "size", "sha256"}]}],      # optional
     "deltas": [{"from", "to", "key", "size", "sha256", "target_sha256"}]}

A delta from..to is valid from any version in [from, to) (overwrite semantics,
see ``reliquary.shared.ckpt_delta``), so per-step and cumulative deltas are one
kind of edge.

An anchor also carries the model's non-weight files (``model_files``: the HF
config, generation config, tokenizer and chat template), so a miner that starts
from the bucket alone can load and serve the checkpoint. Their names are plain
file names (``model_file_name_ok``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from typing import Any, Callable, Mapping

INDEX_FORMAT = "reliquary-ckpt-index/1"
INDEX_KEY = "index.json"
SIGNING_DOMAIN = b"reliquary-ckpt-index/1\n"


def run_prefix(run: str) -> str:
    if not run or "/" in run or run in {".", ".."}:
        raise ValueError("run must be a single path component")
    return f"rl/{run}"


def anchor_key(run: str, version: int, filename: str) -> str:
    return f"{run_prefix(run)}/anchors/{int(version):08d}/{filename}"


def delta_key(run: str, from_version: int, to_version: int) -> str:
    return f"{run_prefix(run)}/deltas/{int(from_version):08d}-{int(to_version):08d}.rqd"


def index_key(run: str) -> str:
    return f"{run_prefix(run)}/{INDEX_KEY}"


def observations_root(run: str) -> str:
    """The async validator's scan log (``validator/rl_observations``), beside the run's checkpoints."""
    return f"{run_prefix(run)}/observations"


def index_keys(index: Mapping[str, Any]) -> list[str]:
    keys = [f["key"] for a in index.get("anchors", []) for f in a["files"]]
    keys += [f["key"] for a in index.get("anchors", []) for f in a.get("model_files", [])]
    return keys + [d["key"] for d in index.get("deltas", [])]


# The HF loader's non-weight files an anchor carries: configs, tokenizer, chat template.
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".safetensors.index.json", ".bin.index.json")
MAX_MODEL_FILE_BYTES = 64 * 1024 * 1024


def model_file_name_ok(name: str) -> bool:
    """A plain file name (no directory, not hidden) that is not a weight file."""
    return (isinstance(name, str) and 0 < len(name) <= 128 and "/" not in name and "\\" not in name
            and not name.startswith(".") and name not in {".", ".."} and not name.endswith(WEIGHT_SUFFIXES))


def model_file_key(run: str, version: int, name: str) -> str:
    if not model_file_name_ok(name):
        raise ValueError(f"{name!r} is not a model file name")
    return f"{run_prefix(run)}/anchors/{int(version):08d}/model/{name}"


def latest_model_files(index: Mapping[str, Any]) -> list[dict]:
    """The model files of the newest anchor that lists some (empty when none does)."""
    for anchor in sorted(index.get("anchors", []), key=lambda a: int(a["version"]), reverse=True):
        if anchor.get("model_files"):
            return list(anchor["model_files"])
    return []


# ------------------------------------------------------------------- signature

def index_signing_payload(index: Mapping[str, Any]) -> bytes:
    """Canonical bytes the validator hotkey signs (URLs are never signed)."""
    return SIGNING_DOMAIN + json.dumps(
        index, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode()


def sign_index(index: Mapping[str, Any], sign_fn: Callable[[bytes], bytes]) -> str:
    return bytes(sign_fn(index_signing_payload(index))).hex()


def _bittensor_verify(hotkey: str, payload: bytes, signature: bytes) -> bool:
    import bittensor as bt

    return bool(bt.Keypair(ss58_address=hotkey).verify(data=payload, signature=signature))


def verify_index_signature(
    index: Mapping[str, Any],
    signature_hex: str,
    hotkey: str,
    *,
    verify_fn: Callable[[str, bytes, bytes], bool] | None = None,
) -> bool:
    try:
        signature = bytes.fromhex(signature_hex)
    except (TypeError, ValueError):
        return False
    return (verify_fn or _bittensor_verify)(hotkey, index_signing_payload(index), signature)


# -------------------------------------------------------------------- planning

@dataclass
class DownloadPlan:
    target_version: int
    target_sha256: str
    anchor: dict | None = None
    deltas: list[dict] = field(default_factory=list)

    @property
    def bytes(self) -> int:
        anchor = sum(f["size"] for f in self.anchor["files"]) if self.anchor else 0
        return anchor + sum(d["size"] for d in self.deltas)

    @property
    def empty(self) -> bool:
        return self.anchor is None and not self.deltas


class NoDownloadPath(RuntimeError):
    """The index offers no way to reach its latest version."""


def plan_download(index: Mapping[str, Any], current_version: int | None) -> DownloadPlan:
    """Cheapest route (in bytes) to the latest version.

    Starts from the miner's own version (free) or from any anchor (its size),
    then follows deltas; a delta from..to applies from any version in [from, to).
    ``current_version=None`` forces an anchor, which is the fallback after a hash
    mismatch.
    """
    latest = int(index["latest"]["version"])
    sha = str(index["latest"]["sha256"])
    if current_version is not None and current_version >= latest:
        return DownloadPlan(latest, sha)
    # best[version] = (bytes, steps, anchor, deltas)
    best: dict[int, tuple[int, int, dict | None, list[dict]]] = {}

    def relax(version: int, cost: int, anchor: dict | None, deltas: list[dict]) -> None:
        old = best.get(version)
        if old is None or (cost, len(deltas)) < (old[0], old[1]):
            best[version] = (cost, len(deltas), anchor, deltas)

    if current_version is not None:
        relax(int(current_version), 0, None, [])
    for anchor in index.get("anchors", []):
        if int(anchor["version"]) <= latest:
            relax(int(anchor["version"]), sum(f["size"] for f in anchor["files"]), anchor, [])
    deltas = sorted(index.get("deltas", []), key=lambda d: (d["to"], d["from"]))
    nodes = sorted(set(best) | {int(d["to"]) for d in deltas})
    for version in nodes:
        if version not in best or version >= latest:
            continue
        cost, _, anchor, path = best[version]
        for d in deltas:
            if int(d["from"]) <= version < int(d["to"]) <= latest:
                relax(int(d["to"]), cost + int(d["size"]), anchor, [*path, d])
    if latest not in best:
        raise NoDownloadPath(f"no route to version {latest} from {current_version}")
    _, _, anchor, path = best[latest]
    return DownloadPlan(latest, sha, anchor, path)
