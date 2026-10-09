"""What a legacy RL task, a single-turn v2 order and a corpus job put on the wire and on disk, as digests.

``probe()`` runs the same inputs through the reliquary tree that is first on ``sys.path`` and returns JSON-able
values. ``tests/unit/data/legacy_wire_1c6930f4.json`` holds what it returned for the tree BEFORE signed episodes (the
phase 2 base 1c6930f4); ``test_episode_inertness`` compares today's tree with it.

It only calls APIs both trees have, with inputs from test helpers that are byte-identical in both trees
(``test_training_payload_codec``, ``test_batch_submission_schema``, ``test_episode_protocol``,
``service_v2_fixtures``, ``test_service_runtime_v2``, ``proof_worker_support``).

To regenerate (never from today's tree)::

    mkdir old && git archive 1c6930f4 reliquary tests | tar -x -C old
    cd old && <repo>/.venv/bin/python -I -c "import sys, json; sys.path.insert(0, '.'); \\
        import importlib.util as u; s = u.spec_from_file_location('probe', '<repo>/tests/unit/legacy_wire_probe.py'); \\
        m = u.module_from_spec(s); s.loader.exec_module(m); print(json.dumps(m.probe(), indent=1, sort_keys=True))"
"""
from __future__ import annotations

import copy
import hashlib
import io
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from typing import Any


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json_sha(value: Any) -> str:
    return _sha(json.dumps(value, sort_keys=True, separators=(",", ":"), default=repr).encode())


def _plain(value: Any) -> Any:
    if isinstance(value, SimpleNamespace):
        return {k: _plain(v) for k, v in sorted(vars(value).items())}
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


# ------------------------------------------------------------------ training payload

def _decoded(blob: bytes) -> dict:
    from reliquary.shared.training_payload import decode_training_payload

    decoded = decode_training_payload(blob)
    return {"schema_version": decoded.schema_version, "batches": _plain(decoded.batches())}


def payload() -> dict:
    import numpy as np

    from tests.unit.test_training_payload_codec import _payload_bytes

    blob = _payload_bytes()
    with np.load(io.BytesIO(blob), allow_pickle=False) as npz:
        header = json.loads(bytes(npz["header"]))
        arrays = {name: _sha(npz[name].tobytes()) for name in sorted(npz.files) if name != "header"}
    historical = (Path(__file__).parents[1] / "fixtures/payloads/historical-episode-schema3.npz").read_bytes()
    return {
        "single_turn_header": header,
        "single_turn_arrays": arrays,
        "single_turn_decoded": _json_sha(_decoded(blob)),
        "historical_episode_v1_decoded": _json_sha(_decoded(historical)),
    }


# ------------------------------------------------------------------ the submission wire

def _rollout_meta_refused(meta: dict) -> bool:
    from pydantic import ValidationError

    from reliquary.protocol.submission import RolloutMetadata

    try:
        RolloutMetadata.model_validate(meta)
    except ValidationError:
        return True
    return False


def submission() -> dict:
    from reliquary.constants import MAX_NEW_TOKENS_PROTOCOL_CAP
    from reliquary.protocol.submission import BatchSubmissionRequest, RolloutMetadata
    from tests.unit.test_batch_submission_schema import _valid_rollouts
    from tests.unit.test_episode_protocol import _episode_commit

    request = BatchSubmissionRequest(miner_hotkey="hk" * 24, prompt_idx=42, window_start=1000,
                                     merkle_root="00" * 32, rollouts=_valid_rollouts(k=4),
                                     checkpoint_hash="sha256:test")
    single = dict(request.rollouts[0].commit["rollout"])
    episode = _episode_commit()["rollout"]
    over = {**single, "completion_length": MAX_NEW_TOKENS_PROTOCOL_CAP + 1}
    episode_over = {**episode, "completion_length": MAX_NEW_TOKENS_PROTOCOL_CAP + 1}
    return {
        "request_json": _sha(request.model_dump_json().encode()),
        "request_dump": _json_sha(request.model_dump()),
        "single_turn_meta": RolloutMetadata.model_validate(single).model_dump_json(),
        "episode_v1_meta": _sha(RolloutMetadata.model_validate(episode).model_dump_json().encode()),
        "over_cap_refused": [_rollout_meta_refused(over), _rollout_meta_refused(episode_over)],
        "cap_accepted": _rollout_meta_refused({**single, "completion_length": MAX_NEW_TOKENS_PROTOCOL_CAP}),
    }


# ------------------------------------------------------------------ what a commit signature is checked against

def signature_messages() -> dict:
    """For each legacy / single-turn v2 commit shape: whether ``verify_commit_signature`` checks it, and the sha of
    the message it checks the signature against (a fake keypair records it)."""
    from reliquary.protocol import signatures
    from reliquary.protocol.seed_pool import PROOF_VERSION as POOL_PROOF
    from reliquary.protocol.service_submission import PROOF_VERSION as SERVICE_PROOF
    from tests.unit.test_episode_protocol import _episode_commit

    seen: list[bytes] = []

    class _Keypair:
        def __init__(self, ss58_address):
            self.address = ss58_address

        def verify(self, data, signature):
            seen.append(bytes(data))
            return True

    base = {"tokens": [1, 2, 3, 4], "commitments": [{"sketch": 1}] * 4, "signature": "ab" * 32,
            "beacon": {"randomness": "cd" * 16}, "model": {"name": "m", "layer_index": 3},
            "rollout": {"prompt_length": 2, "completion_length": 2}}
    from reliquary.protocol.seed_pool import ROLLOUT_SCHEMA as POOL_ROLLOUT_SCHEMA
    from reliquary.protocol.service_submission import ServiceBinding

    pool = {"schema": POOL_ROLLOUT_SCHEMA, "pool_sha256": "e" * 64, "seed_index": 1, "rollout_index": 0}
    binding = ServiceBinding("f" * 64, "training").rollout_binding(0)
    episode = _episode_commit()["rollout"]["episode"]
    shapes = {
        "grail_single_turn": dict(base, proof_version=signatures.GRAIL_PROOF_VERSION),
        "grail_episode_v1": dict(base, proof_version=signatures.GRAIL_EPISODE_PROOF_VERSION,
                                 rollout={**base["rollout"], "episode": episode}),
        "grail_episode_missing": dict(base, proof_version=signatures.GRAIL_EPISODE_PROOF_VERSION),
        "grail_with_pool": dict(base, proof_version=signatures.GRAIL_PROOF_VERSION,
                                rollout={**base["rollout"], "seed_pool": pool}),
        "public_pool": dict(base, proof_version=POOL_PROOF, rollout={**base["rollout"], "seed_pool": pool}),
        "public_pool_service": dict(base, proof_version=POOL_PROOF,
                                    rollout={**base["rollout"], "seed_pool": pool, "service_binding": binding}),
        "public_pool_service_no_pool": dict(base, proof_version=POOL_PROOF,
                                            rollout={**base["rollout"], "service_binding": binding}),
        "public_pool_episode_v1": dict(base, proof_version=POOL_PROOF,
                                       rollout={**base["rollout"], "seed_pool": pool, "service_binding": binding,
                                                "episode": episode}),
        "service": dict(base, proof_version=SERVICE_PROOF, rollout={**base["rollout"], "service_binding": binding}),
        "service_episode_v1": dict(base, proof_version=SERVICE_PROOF,
                                   rollout={**base["rollout"], "service_binding": binding, "episode": episode}),
        "unknown_version": dict(base, proof_version="nope/v0"),
    }
    real = signatures.bt
    signatures.bt = SimpleNamespace(Keypair=_Keypair)
    try:
        out = {}
        for name, commit in shapes.items():
            seen.clear()
            verdict = signatures.verify_commit_signature(copy.deepcopy(commit), "5Fhot")
            out[name] = [verdict, _sha(seen[0]) if seen else None]
        return out
    finally:
        signatures.bt = real


# ------------------------------------------------------------------ the remote proof wire

def proof_wire() -> dict:
    from reliquary.protocol.profiles import TOPLOC_DEPLOYED_DEFAULTS
    from reliquary.validator import batcher
    from reliquary.validator.remote_proof_protocol import ProofInput, ProofValues, canonical_bytes
    from tests.unit.proof_worker_support import proof_result_handler

    result = proof_result_handler({"calls": 0}, {"tokens": list(range(8))}, "ab", None)
    toploc = copy.copy(result)
    toploc.toploc_checked, toploc.toploc_passed, toploc.toploc_reason = True, False, "exp_mismatch"
    commit = {"tokens": [5, 6, 7, 8], "commitments": [{"sketch": 0}] * 4,
              "rollout": {"prompt_length": 2, "completion_length": 2}, "toploc_proofs": ["/9kAAQ=="]}
    profile = SimpleNamespace(proofs=(TOPLOC_DEPLOYED_DEFAULTS,))
    prove = getattr(batcher, "_proof_commit", batcher._with_toploc_spec)
    sent = {}
    for name, active in (("with_toploc", profile), ("without_toploc", SimpleNamespace(proofs=()))):
        try:
            proof_commit = prove(copy.deepcopy(commit), active)
        except Exception as exc:          # a profile shape this tree does not read: say so, never guess
            sent[name] = f"error:{type(exc).__name__}"
            continue
        payload = ProofInput(tokens=proof_commit["tokens"], commitments=proof_commit["commitments"],
                             rollout=proof_commit["rollout"], randomness="ab", seed_u_values=None,
                             toploc_proofs=proof_commit.get("toploc_proofs"),
                             toploc_spec=proof_commit.get("toploc_spec"))
        sent[name] = [_sha(canonical_bytes(proof_commit)), _sha(canonical_bytes(payload.model_dump()))]
    return {
        "proof_values": canonical_bytes(ProofValues.from_kernel(result).model_dump()).decode(),
        "proof_values_toploc": canonical_bytes(ProofValues.from_kernel(toploc).model_dump()).decode(),
        "proof_input": sent,
    }


# ------------------------------------------------------------------ a single-turn v2 order

def _without_ids(value: dict) -> dict:
    return {k: v for k, v in value.items() if k not in ("id", "observation_id")}


def single_turn_v2() -> dict:
    from reliquary.protocol.release_contract import canonical_json_bytes
    from tests.unit import test_service_runtime_v2 as rt_tests
    from tests.unit.service_v2_fixtures import contract_v2

    contract = contract_v2()
    with tempfile.TemporaryDirectory() as tmp:
        rt = rt_tests.runtime(Path(tmp))
        try:
            announcement = rt.announcement(window=1, randomness=rt_tests.WINDOW_BEACON)
            explored = rt_tests.explore(rt)
            trained = rt_tests.train(rt, prompt=9)
            # Observation ids are salted per runtime (random): everything else of each event is compared.
            events = [_without_ids(e) for _, e in rt.events(limit=10_000)]
            envelope = rt.envelope(1)
        finally:
            rt.close()
    return {
        "contract_sha256": contract.sha256,
        "contract_bytes": _sha(canonical_json_bytes(contract.to_dict())),
        "announcement": _sha(canonical_json_bytes(announcement)),
        "supported_capabilities": announcement["supported_capabilities"],
        "explored": _json_sha(_without_ids(explored)),
        "trained": _json_sha(_without_ids(trained)),
        "events": _json_sha(events),
        "envelope": _json_sha(envelope),
    }


# ------------------------------------------------------------------ registered environments

def environments() -> dict:
    from reliquary.environment.registry import ENVIRONMENT_SPECS, environment_manifest_sha256, get_environment_spec

    return {
        "manifest_sha256": environment_manifest_sha256(),
        "per_env": {name: _json_sha(get_environment_spec(name).consensus_manifest()) for name in sorted(ENVIRONMENT_SPECS)},
        "modes": {name: spec.interaction_mode for name, spec in sorted(ENVIRONMENT_SPECS.items())},
    }


# ------------------------------------------------------------------ the corpus sandbox sessions

def corpus_sessions() -> dict:
    from reliquary.corpus.signed_reasons import corpus_engagement
    from reliquary.infrastructure import sandbox_store
    from reliquary.sandbox.sessions import SessionRecord

    record = SessionRecord(session_id="s-1", hotkey="5Hot", request_id="r-1", engagement_sha256="a" * 64,
                           kind="corpus", engagement=corpus_engagement("job-7", 3), env="swe", split="train",
                           index=3, checkpoint="c" * 40, job_id="job-7", prompt_index=3, machine_id="m-1",
                           issued_at=1_800_000_000, expires_at=1_800_003_600, token_sha256="b" * 64)
    document = record.to_document()
    return {
        "engagement": corpus_engagement("job-7", 3),
        "session_key": sandbox_store.session_key("s-1", 1_800_003_600),
        "session_prefix": sandbox_store.SESSION_PREFIX,
        "document": json.dumps(document, sort_keys=True),
        "round_trip": SessionRecord.from_document(document).to_document() == document,
    }


def probe() -> dict:
    return {
        "payload": payload(),
        "submission": submission(),
        "signature_messages": signature_messages(),
        "proof_wire": proof_wire(),
        "single_turn_v2": single_turn_v2(),
        "environments": environments(),
        "corpus_sessions": corpus_sessions(),
    }
