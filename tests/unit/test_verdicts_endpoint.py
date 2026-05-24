"""GET /verdicts/{hotkey} surfaces the real per-submission verdicts the
``/submit`` response cannot return synchronously under the production
queue-worker path.

Before this feature, the /submit response was always ``accepted=True
reason=submitted`` under the production worker — even for submissions
the worker later rejected with GRAIL_FAIL / WRONG_RANDOMNESS / etc.
Miners had to wait minutes for the R2 archive upload to learn the real
outcome. The endpoint here closes that gap to a few seconds.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from reliquary.constants import (
    CHALLENGE_K,
    M_ROLLOUTS,
    MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW,
)
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RejectReason,
    RolloutSubmission,
    WindowState,
)
from reliquary.validator.batcher import GrpoWindowBatcher
from reliquary.validator.server import ValidatorServer, VERDICT_CAP_PER_HOTKEY


class _FakeEnv:
    name = "fake"
    def __len__(self): return 1000
    def get_problem(self, idx):
        return {"prompt": f"p{idx}", "ground_truth": "", "id": f"p{idx}"}
    def compute_reward(self, p, c):
        return 1.0 if "CORRECT" in c else 0.0


class _ModelStub:
    class config:
        vocab_size = 10000
        max_position_embeddings = 4096


def _always_true_proof(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


def _make_commit(success: bool = False, total_reward: float = 0.0) -> dict:
    prompt_length = 4
    seq_len = CHALLENGE_K + prompt_length
    tokens = list(range(seq_len))
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v5",
        "model": {"name": "test-model", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": prompt_length,
            "completion_length": completion_length,
            "success": success,
            "total_reward": total_reward,
            "advantage": 0.0,
            "token_logprobs": [0.0] * seq_len,
        },
    }


def _batcher(window_start: int = 500) -> GrpoWindowBatcher:
    b = GrpoWindowBatcher(
        window_start=window_start,
        env=_FakeEnv(),
        model=_ModelStub(),
        verify_commitment_proofs_fn=_always_true_proof,
        verify_signature_fn=lambda c, h: True,
        completion_text_fn=lambda r: "CORRECT" if r.reward > 0.5 else "wrong",
        # v2.3: requests in this suite default to ``drand_round=0`` (the
        # legacy/sentinel value). The drand-round timing gate would reject
        # them all as STALE_ROUND before the rest of the pipeline ever
        # runs, hiding what these tests actually exercise. Disable here;
        # the gate is independently covered by test_grpo_window_batcher.
        drand_round_check_enabled=False,
    )
    b.current_checkpoint_hash = "sha256:test"
    b.randomness = "cd" * 16  # match _make_commit's beacon
    return b


def _request(
    prompt_idx: int = 42,
    window_start: int = 500,
    hotkey: str = "hk",
    merkle_root: str | None = None,
    k_success: int = 4,
) -> BatchSubmissionRequest:
    rollouts = []
    for i in range(M_ROLLOUTS):
        success = i < k_success
        reward = 1.0 if success else 0.0
        commit = _make_commit(success=success, total_reward=reward)
        rollouts.append(
            RolloutSubmission(
                tokens=commit["tokens"], reward=reward, commit=commit,
            )
        )
    # Default to a deterministic merkle-root per (hotkey, prompt) so tests
    # can find the verdict for the specific submission they fired.
    if merkle_root is None:
        merkle_root = f"{prompt_idx:032x}{0:032x}"
    return BatchSubmissionRequest(
        miner_hotkey=hotkey,
        prompt_idx=prompt_idx,
        window_start=window_start,
        merkle_root=merkle_root,
        rollouts=rollouts,
        checkpoint_hash="sha256:test",
    )


def _make_server_open() -> tuple[ValidatorServer, TestClient]:
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.OPEN)
    return server, TestClient(server.app)


# --- happy path: sync verdict surfaces immediately -------------------------


def test_accepted_submission_appears_in_verdicts() -> None:
    """Under TestClient (no worker), /submit runs the batcher inline and
    the real ACCEPTED verdict is known before the response goes out. The
    /verdicts endpoint must reflect that verdict immediately."""
    server, client = _make_server_open()
    req = _request(prompt_idx=42, hotkey="hkA")

    resp = client.post("/submit", json=req.model_dump(mode="json"))
    assert resp.status_code == 200
    assert resp.json()["accepted"] is True
    assert resp.json()["reason"] == "accepted"

    v = client.get("/verdicts/hkA").json()
    assert len(v["verdicts"]) == 1
    entry = v["verdicts"][0]
    assert entry["merkle_root"] == req.merkle_root
    assert entry["accepted"] is True
    assert entry["reason"] == "accepted"
    assert entry["window_n"] == 500
    assert entry["ts"] > 0


def test_rate_limited_submission_appears_in_verdicts() -> None:
    """A miner blasting past the per-hotkey rate cap gets RATE_LIMITED
    responses synchronously. Each of those rejections must land in the
    verdicts ring so the miner can see WHY their burst is getting eaten."""
    server, client = _make_server_open()
    # Fire one over the cap.
    for i in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW + 1):
        req = _request(prompt_idx=i, hotkey="hkB")
        client.post("/submit", json=req.model_dump(mode="json"))

    v = client.get("/verdicts/hkB").json()["verdicts"]
    rate_limited = [e for e in v if e["reason"] == "rate_limited"]
    assert len(rate_limited) == 1, "exactly one rate_limited verdict expected"


def test_window_not_active_appears_in_verdicts() -> None:
    """When the validator is in TRAINING / PUBLISHING / READY (anything
    other than OPEN), submissions get WINDOW_NOT_ACTIVE — and the miner
    sees that verdict via /verdicts."""
    server = ValidatorServer()
    server.set_active_batcher(_batcher(window_start=500))
    server.set_current_state(WindowState.TRAINING)
    client = TestClient(server.app)

    req = _request(hotkey="hkC")
    client.post("/submit", json=req.model_dump(mode="json"))

    v = client.get("/verdicts/hkC").json()["verdicts"]
    assert len(v) == 1
    assert v[0]["reason"] == "window_not_active"
    assert v[0]["accepted"] is False


# --- batch-filled cutoff (PR #22 path) -------------------------------------


def test_batch_filled_appears_in_verdicts() -> None:
    """Once the batcher is sealed, the HTTP cutoff rejects further
    submissions as BATCH_FILLED. Those rejections must also be visible
    via /verdicts."""
    server, client = _make_server_open()
    # Force the batcher to look sealed.
    server.active_batcher._seal_flag.set()

    req = _request(hotkey="hkD")
    client.post("/submit", json=req.model_dump(mode="json"))

    v = client.get("/verdicts/hkD").json()["verdicts"]
    assert any(e["reason"] == "batch_filled" for e in v)


# --- isolation: one hotkey's verdicts don't leak into another's -----------


def test_verdicts_are_per_hotkey_only() -> None:
    """Submitting from hkX must not show up in /verdicts/hkY. Per-hotkey
    ring buffer; no cross-pollination."""
    server, client = _make_server_open()
    client.post("/submit", json=_request(prompt_idx=1, hotkey="hkX").model_dump(mode="json"))
    client.post("/submit", json=_request(prompt_idx=2, hotkey="hkY").model_dump(mode="json"))

    x_verdicts = client.get("/verdicts/hkX").json()["verdicts"]
    y_verdicts = client.get("/verdicts/hkY").json()["verdicts"]
    assert len(x_verdicts) == 1
    assert len(y_verdicts) == 1
    assert x_verdicts[0]["merkle_root"] != y_verdicts[0]["merkle_root"]


def test_verdicts_empty_for_unseen_hotkey() -> None:
    """Querying a hotkey we've never seen returns an empty list (not 404).
    Avoids leaking 'this hotkey exists vs not' via status code."""
    server, _ = _make_server_open()
    client = TestClient(server.app)
    r = client.get("/verdicts/nobody")
    assert r.status_code == 200
    assert r.json() == {"verdicts": []}


# --- since= incremental polling --------------------------------------------


def test_since_filter_returns_only_newer_verdicts() -> None:
    """A miner polling /verdicts should pass ``since=<last seen ts>`` to
    avoid re-downloading verdicts they already have. The endpoint
    contract: strict ``>``, so the same ts is excluded."""
    server, client = _make_server_open()
    client.post("/submit", json=_request(prompt_idx=1, hotkey="hkE").model_dump(mode="json"))
    first_ts = client.get("/verdicts/hkE").json()["verdicts"][0]["ts"]

    # Small sleep then a second submission so timestamps differ.
    import time
    time.sleep(0.005)
    client.post("/submit", json=_request(prompt_idx=2, hotkey="hkE").model_dump(mode="json"))

    later = client.get(f"/verdicts/hkE?since={first_ts}").json()["verdicts"]
    assert len(later) == 1
    assert later[0]["merkle_root"] == _request(prompt_idx=2).merkle_root


def test_since_in_future_returns_empty() -> None:
    """A miner passing a ``since`` in the future (clock skew, debugging)
    gets an empty list — not an error."""
    server, client = _make_server_open()
    client.post("/submit", json=_request(prompt_idx=1, hotkey="hkF").model_dump(mode="json"))
    r = client.get("/verdicts/hkF?since=99999999999")
    assert r.status_code == 200
    assert r.json()["verdicts"] == []


# --- ring buffer bounds ----------------------------------------------------


def test_ring_buffer_caps_at_cap_per_hotkey() -> None:
    """Spamming submissions from one hotkey can't grow the ring buffer
    without bound. After ``VERDICT_CAP_PER_HOTKEY`` writes the oldest
    entries roll off."""
    server, client = _make_server_open()
    # Submit just enough to exceed cap via the rate_limit fast-path
    # (cheap — no GRAIL forward pass even in the inline TestClient).
    for i in range(VERDICT_CAP_PER_HOTKEY + 5):
        req = _request(prompt_idx=i % 1000, hotkey="hkG")
        client.post("/submit", json=req.model_dump(mode="json"))

    v = client.get("/verdicts/hkG").json()["verdicts"]
    assert len(v) <= VERDICT_CAP_PER_HOTKEY


# --- payload shape -----------------------------------------------------------


def test_verdict_payload_shape() -> None:
    """Every entry must carry the contract fields. Bound by the Pydantic
    model so the response can be parsed by miner-side schemas without
    field-by-field defensive handling."""
    server, client = _make_server_open()
    req = _request(prompt_idx=99, hotkey="hkH")
    client.post("/submit", json=req.model_dump(mode="json"))

    body = client.get("/verdicts/hkH").json()
    assert set(body.keys()) == {"verdicts"}
    entry = body["verdicts"][0]
    assert {"merkle_root", "window_n", "accepted", "reason", "ts"}.issubset(
        entry.keys()
    )
    assert isinstance(entry["merkle_root"], str)
    assert len(entry["merkle_root"]) == 64   # protocol pattern enforces
    assert isinstance(entry["accepted"], bool)
    assert isinstance(entry["reason"], str)
    assert isinstance(entry["ts"], float)


# --- ordering --------------------------------------------------------------


def test_verdicts_ordered_by_ts_ascending() -> None:
    """The miner streams verdicts in order. Pinning the ordering contract
    here so a future refactor (e.g. switching to a heap) doesn't silently
    break that."""
    server, client = _make_server_open()
    for i in range(3):
        req = _request(prompt_idx=i, hotkey="hkI")
        client.post("/submit", json=req.model_dump(mode="json"))
        import time
        time.sleep(0.003)

    v = client.get("/verdicts/hkI").json()["verdicts"]
    ts_seq = [e["ts"] for e in v]
    assert ts_seq == sorted(ts_seq)


# --- direct API surface ----------------------------------------------------


def test_record_verdict_method_directly() -> None:
    """The public ``record_verdict`` method is the recording interface
    used by every code path that produces a verdict. Test it directly
    so the contract is pinned independently of which /submit path
    happened to flow into it."""
    server = ValidatorServer()

    server.record_verdict(
        "hkJ", "ab" * 32, accepted=True, reason=RejectReason.ACCEPTED,
        window_n=42,
    )
    server.record_verdict(
        "hkJ", "cd" * 32, accepted=False, reason=RejectReason.GRAIL_FAIL,
        window_n=43,
    )
    client = TestClient(server.app)
    v = client.get("/verdicts/hkJ").json()["verdicts"]
    assert len(v) == 2
    assert v[0]["reason"] == "accepted"
    assert v[1]["reason"] == "grail_fail"


def test_record_verdict_accepts_str_reason_for_late_drops() -> None:
    """Late drops use the worker_dropped pseudo-reason which is a string
    rather than an enum value. ``record_verdict`` must accept either."""
    server = ValidatorServer()
    server.record_verdict(
        "hkK", "ee" * 32, accepted=False,
        reason=RejectReason.WORKER_DROPPED,
        window_n=42,
    )
    client = TestClient(server.app)
    v = client.get("/verdicts/hkK").json()["verdicts"]
    assert v[0]["reason"] == "worker_dropped"
