"""Cheap rejects pre-queue on the /submit HTTP handler.

Every reject reason that depends only on O(1) batcher state must be
returned synchronously by the HTTP handler, BEFORE the request hits the
async worker queue. Without this, a STALE_ROUND or WRONG_CHECKPOINT
submission has to wait in line behind ~5–25 s GRAIL forward passes of
honest submissions ahead of it in the queue — minutes of latency on what
should be a microsecond rejection.

These tests pin the contract: each reject reason returns synchronously
on /submit, and the submit_queue is NOT populated (the worker never sees
the request).
"""

from unittest.mock import MagicMock

from fastapi.testclient import TestClient

from reliquary.constants import MAX_SUBMISSIONS_PER_PROMPT
from reliquary.protocol.submission import (
    BatchSubmissionResponse, RejectReason, WindowState,
)
from reliquary.validator.server import ValidatorServer


def _submission(prompt_idx: int = 42, checkpoint_hash: str = "sha256:current",
                window_start: int = 500, drand_round: int = 0,
                hotkey: str = "hkA") -> dict:
    commit = {
        "tokens": list(range(36)),
        "commitments": [{"sketch": 0} for _ in range(36)],
        "proof_version": "v5",
        "model": {"name": "test", "layer_index": 6},
        "signature": "ab" * 32,
        "beacon": {"randomness": "cd" * 16},
        "rollout": {
            "prompt_length": 4, "completion_length": 32,
            "success": True, "total_reward": 1.0, "advantage": 0.0,
            "token_logprobs": [0.0] * 36,
        },
    }
    return {
        "miner_hotkey": hotkey,
        "prompt_idx": prompt_idx,
        "window_start": window_start,
        "merkle_root": "00" * 32,
        "rollouts": [{"tokens": list(range(36)), "reward": 1.0, "commit": commit}] * 8,
        "checkpoint_hash": checkpoint_hash,
        "drand_round": drand_round,
    }


def _setup(*,
           current_checkpoint_hash: str = "sha256:current",
           cooldown_prompts: list[int] | None = None,
           env_len: int = 1000,
           drand_round_check_enabled: bool = False,
           validate_round_returns: RejectReason | None = None,
           prompt_count: int = 0) -> tuple[ValidatorServer, MagicMock]:
    """Build a server + mocked batcher in OPEN state with the given knobs."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    batcher = MagicMock()
    batcher.window_start = 500
    batcher.current_checkpoint_hash = current_checkpoint_hash
    batcher.cooldown_prompts_snapshot = cooldown_prompts or []
    batcher.env = MagicMock()
    batcher.env.__len__.return_value = env_len
    batcher.is_sealed.return_value = False
    batcher.drand_round_check_enabled = drand_round_check_enabled
    batcher.validate_drand_round.return_value = validate_round_returns
    batcher.prompt_submission_count.return_value = prompt_count
    # The TestClient runs /submit synchronously (no worker), so the happy
    # path calls batcher.accept_submission directly. Return an ACCEPTED
    # so happy-path tests can distinguish "pre-queue reject" from "passed
    # the cheap checks and the (mocked) worker logic ACCEPTED".
    batcher.accept_submission.return_value = BatchSubmissionResponse(
        accepted=True, reason=RejectReason.ACCEPTED,
    )
    s.set_active_batcher(batcher)
    return s, batcher


def _assert_pre_queue_reject(s: ValidatorServer, payload: dict,
                              expected: RejectReason) -> None:
    """Common assertion: /submit returns expected reason, queue stays empty."""
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] is False, body
    assert body["reason"] == expected.value
    # The worker queue must NOT have been populated.
    assert s._submit_queue.qsize() == 0


def test_wrong_checkpoint_rejected_pre_queue():
    s, _ = _setup(current_checkpoint_hash="sha256:current")
    payload = _submission(checkpoint_hash="sha256:stale")
    _assert_pre_queue_reject(s, payload, RejectReason.WRONG_CHECKPOINT)


def test_empty_current_checkpoint_skips_gate():
    """Empty server-side checkpoint hash is the bootstrap sentinel; any
    miner-claimed checkpoint passes through."""
    s, _ = _setup(current_checkpoint_hash="")
    payload = _submission(checkpoint_hash="sha256:whatever")
    # No reject reason for checkpoint mismatch; submission queues.
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    assert r.status_code == 200
    assert r.json()["reason"] == RejectReason.ACCEPTED.value


def test_bad_prompt_idx_rejected_pre_queue():
    s, _ = _setup(env_len=100)
    payload = _submission(prompt_idx=500)  # 500 >= env_len=100
    _assert_pre_queue_reject(s, payload, RejectReason.BAD_PROMPT_IDX)


def test_prompt_in_cooldown_rejected_pre_queue():
    s, _ = _setup(cooldown_prompts=[42, 99])
    payload = _submission(prompt_idx=42)
    _assert_pre_queue_reject(s, payload, RejectReason.PROMPT_IN_COOLDOWN)


def test_stale_round_rejected_pre_queue():
    s, _ = _setup(
        drand_round_check_enabled=True,
        validate_round_returns=RejectReason.STALE_ROUND,
    )
    payload = _submission(drand_round=42)
    _assert_pre_queue_reject(s, payload, RejectReason.STALE_ROUND)


def test_future_round_rejected_pre_queue():
    s, _ = _setup(
        drand_round_check_enabled=True,
        validate_round_returns=RejectReason.FUTURE_ROUND,
    )
    payload = _submission(drand_round=9_999_999)
    _assert_pre_queue_reject(s, payload, RejectReason.FUTURE_ROUND)


def test_drand_check_disabled_skips_gate():
    """When the batcher has the gate off (legacy fixtures), validate_drand_round
    is NOT called and the submission queues normally."""
    s, batcher = _setup(drand_round_check_enabled=False)
    payload = _submission(drand_round=0)
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    assert r.json()["reason"] == RejectReason.ACCEPTED.value
    batcher.validate_drand_round.assert_not_called()


def test_prompt_full_rejected_pre_queue():
    s, _ = _setup(prompt_count=MAX_SUBMISSIONS_PER_PROMPT)
    payload = _submission(prompt_idx=42)
    _assert_pre_queue_reject(s, payload, RejectReason.PROMPT_FULL)


def test_prompt_full_below_cap_passes():
    """K_p < MAX is the common case; submission must queue normally."""
    s, _ = _setup(prompt_count=MAX_SUBMISSIONS_PER_PROMPT - 1)
    payload = _submission(prompt_idx=42)
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    assert r.json()["reason"] == RejectReason.ACCEPTED.value


def test_pre_queue_rejects_recorded_as_verdicts():
    """Each pre-queue reject must show up in the per-hotkey verdict ring
    buffer with the right reason — same as the worker-path rejects do."""
    s, _ = _setup(current_checkpoint_hash="sha256:current")
    payload = _submission(hotkey="hkV", checkpoint_hash="sha256:stale")
    with TestClient(s.app) as client:
        client.post("/submit", json=payload)
    verdicts = list(s._verdicts.get("hkV", []))
    assert len(verdicts) == 1
    assert verdicts[0]["accepted"] is False
    assert verdicts[0]["reason"] == RejectReason.WRONG_CHECKPOINT.value


def test_cheap_reject_does_not_burn_rate_limit_budget():
    """Cheap rejects DO consume the per-hotkey counter (rate_limit increments
    happen before the cheap rejects, intentionally — a spammer flooding bad
    submissions still trips the rate limit). Document the contract here so
    future refactors don't accidentally re-order it."""
    from reliquary.constants import MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW
    s, _ = _setup(current_checkpoint_hash="sha256:current")
    payload = _submission(hotkey="hkR", checkpoint_hash="sha256:stale")
    with TestClient(s.app) as client:
        for _ in range(MAX_SUBMISSIONS_PER_HOTKEY_PER_WINDOW):
            client.post("/submit", json=payload)
        # The (N+1)th post hits RATE_LIMITED before the checkpoint check.
        r = client.post("/submit", json=payload)
    assert r.json()["reason"] == RejectReason.RATE_LIMITED.value
