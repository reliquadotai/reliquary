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

from copy import deepcopy
from types import SimpleNamespace
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


def _submission_with_completion_tokens(completion_tokens: list[int]) -> dict:
    payload = _submission()
    prompt_tokens = list(range(4))
    tokens = prompt_tokens + list(completion_tokens)
    rollouts = []
    for rollout in payload["rollouts"]:
        commit = deepcopy(rollout["commit"])
        commit["tokens"] = list(tokens)
        commit["commitments"] = [{"sketch": 0} for _ in tokens]
        commit["rollout"]["prompt_length"] = len(prompt_tokens)
        commit["rollout"]["completion_length"] = len(completion_tokens)
        commit["rollout"]["token_logprobs"] = [0.0] * len(tokens)
        rollouts.append({
            "tokens": list(tokens),
            "reward": rollout["reward"],
            "commit": commit,
        })
    payload["rollouts"] = rollouts
    return payload


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
    # MagicMock attribute access auto-creates truthy mocks; pin the seal
    # extension's trigger round attribute to None so the BATCH_FILLED
    # gate at the cheap-reject layer doesn't fire for tests that don't
    # exercise the seal extension.
    batcher._seal_trigger_round = None
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


class _ProofAdmissionFullBatcher:
    window_start = 500
    current_checkpoint_hash = "sha256:current"
    cooldown_prompts_snapshot: list[int] = []
    valid_count = 7
    _seal_trigger_round = 123
    drand_round_check_enabled = False
    proof_admission_count = 32
    post_trigger_proof_admission_count = 8

    class _Env:
        def __len__(self):
            return 1000

    def __init__(self):
        self.env = self._Env()

    def is_sealed(self) -> bool:
        return False

    def prompt_submission_count(self, prompt_idx: int) -> int:
        return 0

    def try_reserve_proof_admission(self, request):
        return False, "proof_admission_window_full"

    def accept_submission(self, request, *, telemetry=None):  # pragma: no cover
        raise AssertionError("proof-admission reject must happen pre-queue")


class _PreflightAdmissionBatcher:
    window_start = 500
    current_checkpoint_hash = "sha256:current"
    cooldown_prompts_snapshot: list[int] = []
    valid_count = 0
    _seal_trigger_round = None
    drand_round_check_enabled = False
    proof_admission_count = 0
    post_trigger_proof_admission_count = 0
    bootstrap = False

    class _Env:
        def __len__(self):
            return 1000

    def __init__(self, eos_token_id: int = 99):
        self.env = self._Env()
        self.tokenizer = SimpleNamespace(eos_token_id=eos_token_id)
        self.model = SimpleNamespace(
            config=SimpleNamespace(
                vocab_size=200_000,
                max_position_embeddings=20_000,
            ),
            generation_config=SimpleNamespace(eos_token_id=eos_token_id),
        )

    def is_sealed(self) -> bool:
        return False

    def prompt_submission_count(self, prompt_idx: int) -> int:
        return 0

    def try_reserve_proof_admission(self, request):
        self.proof_admission_count += 1
        return True, None

    def accept_submission(self, request, *, telemetry=None):
        return BatchSubmissionResponse(
            accepted=True, reason=RejectReason.ACCEPTED,
        )


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


def test_proof_admission_full_rejected_pre_queue():
    """When the global proof budget is exhausted, /submit must reject before
    queueing or running the batcher's expensive accept path.
    """
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    s.set_active_batcher(_ProofAdmissionFullBatcher())
    s._worker_task = object()
    payload = _submission(drand_round=123)

    _assert_pre_queue_reject(s, payload, RejectReason.BATCH_FILLED)
    verdicts = list(s._verdicts.get("hkA", []))
    assert verdicts
    assert verdicts[-1]["reject_stage"] == "proof_admission"


def test_non_cap_non_eos_rejected_before_proof_admission():
    """Short completions that never emit EOS cannot pass termination, so they
    must not reserve one of the scarce proof slots first."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    batcher = _PreflightAdmissionBatcher(eos_token_id=99)
    s.set_active_batcher(batcher)
    payload = _submission_with_completion_tokens(list(range(4, 36)))

    _assert_pre_queue_reject(s, payload, RejectReason.BAD_TERMINATION)
    assert batcher.proof_admission_count == 0
    verdicts = list(s._verdicts.get("hkA", []))
    assert verdicts
    assert verdicts[-1]["reject_stage"] == "termination_preflight"


def test_eos_padding_rejected_before_proof_admission():
    """Repeated or interior EOS tokens are structural padding and should be
    rejected before they can enter the GRAIL queue."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    batcher = _PreflightAdmissionBatcher(eos_token_id=99)
    s.set_active_batcher(batcher)
    completion_tokens = list(range(4, 35)) + [99]
    completion_tokens[10] = 99
    payload = _submission_with_completion_tokens(completion_tokens)

    _assert_pre_queue_reject(s, payload, RejectReason.BAD_TERMINATION)
    assert batcher.proof_admission_count == 0


def test_final_eos_reaches_proof_admission():
    """The preflight is structural only: a normal final-EOS completion still
    goes through proof admission and the existing verifier decides p_stop."""
    s = ValidatorServer()
    s.set_current_state(WindowState.OPEN)
    batcher = _PreflightAdmissionBatcher(eos_token_id=99)
    s.set_active_batcher(batcher)
    payload = _submission_with_completion_tokens(list(range(4, 35)) + [99])

    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)

    body = r.json()
    assert body["accepted"] is True, body
    assert body["reason"] == RejectReason.ACCEPTED.value
    assert batcher.proof_admission_count == 1


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


def test_reject_order_matches_accept_locked():
    """When a submission trips multiple cheap checks at once, the handler
    must return the SAME reason the worker's _accept_locked would have
    returned. Order pinned: WRONG_CHECKPOINT > drand_round >
    BAD_PROMPT_IDX > PROMPT_IN_COOLDOWN > PROMPT_FULL."""
    # WRONG_CHECKPOINT trips first even if drand_round + cooldown also fail.
    s, _ = _setup(
        current_checkpoint_hash="sha256:current",
        cooldown_prompts=[42],
        drand_round_check_enabled=True,
        validate_round_returns=RejectReason.STALE_ROUND,
    )
    payload = _submission(prompt_idx=42, checkpoint_hash="sha256:stale", drand_round=99)
    _assert_pre_queue_reject(s, payload, RejectReason.WRONG_CHECKPOINT)

    # When checkpoint passes, drand_round trips next.
    s, _ = _setup(
        current_checkpoint_hash="sha256:current",
        cooldown_prompts=[42],
        drand_round_check_enabled=True,
        validate_round_returns=RejectReason.STALE_ROUND,
    )
    payload = _submission(prompt_idx=42, checkpoint_hash="sha256:current", drand_round=99)
    _assert_pre_queue_reject(s, payload, RejectReason.STALE_ROUND)

    # When checkpoint + drand pass, BAD_PROMPT_IDX trips before cooldown.
    s, _ = _setup(
        env_len=100,
        cooldown_prompts=[500],
    )
    payload = _submission(prompt_idx=500)  # both > env_len AND in cooldown
    _assert_pre_queue_reject(s, payload, RejectReason.BAD_PROMPT_IDX)


def test_cheap_reject_logs_warning():
    """Each cheap reject emits a WARNING log line matching the worker's
    format. Without this, operators lose the ``grep stale_round`` flow
    they use to identify non-conformant miners after a v2.3 deploy."""
    import logging
    s, _ = _setup(current_checkpoint_hash="sha256:current")
    payload = _submission(hotkey="hkL", checkpoint_hash="sha256:stale")

    # Capture WARNING logs from the server module.
    server_logger = logging.getLogger("reliquary.validator.server")
    records = []
    handler = logging.Handler()
    handler.setLevel(logging.WARNING)
    handler.emit = records.append
    server_logger.addHandler(handler)
    try:
        with TestClient(s.app) as client:
            client.post("/submit", json=payload)
    finally:
        server_logger.removeHandler(handler)

    reject_lines = [r for r in records if "rejected prompt" in r.getMessage()
                    and "wrong_checkpoint" in r.getMessage()]
    assert reject_lines, "WARNING log missing for cheap WRONG_CHECKPOINT reject"


def test_cheap_reject_log_includes_drand_round():
    """The WARNING log line for a cheap reject must include
    ``drand_round=N`` so operators can grep the per-drand-round
    distribution of rejections (= verify whether miners are burst-firing
    in the first drand round or spreading their POSTs)."""
    import logging
    s, _ = _setup(current_checkpoint_hash="sha256:current")
    payload = _submission(
        hotkey="hkD", checkpoint_hash="sha256:stale", drand_round=28700123,
    )

    server_logger = logging.getLogger("reliquary.validator.server")
    records = []
    handler = logging.Handler()
    handler.setLevel(logging.WARNING)
    handler.emit = records.append
    server_logger.addHandler(handler)
    try:
        with TestClient(s.app) as client:
            client.post("/submit", json=payload)
    finally:
        server_logger.removeHandler(handler)

    reject_lines = [
        r for r in records
        if "rejected prompt" in r.getMessage()
        and "drand_round=28700123" in r.getMessage()
    ]
    assert reject_lines, (
        "cheap-reject log must surface the submission's drand_round; "
        f"got messages: {[r.getMessage() for r in records]}"
    )


def test_validate_drand_round_called_with_arrival_timestamp():
    """The HTTP cheap-reject must forward the middleware-stamped
    ``t_arrival`` into ``batcher.validate_drand_round``. Without this,
    the drand check uses ``time.time()`` at handler-execution time and
    becomes vulnerable to event-loop stalls — the prod failure mode the
    arrival-time stamping was added to fix.
    """
    import time

    s, batcher = _setup(
        current_checkpoint_hash="sha256:current",
        drand_round_check_enabled=True,
        validate_round_returns=None,  # treat round as OK so we reach acceptance
    )
    payload = _submission()
    t_before = time.time()
    with TestClient(s.app) as client:
        client.post("/submit", json=payload)
    t_after = time.time()

    assert batcher.validate_drand_round.called, (
        "drand check should have run on the cheap-reject path"
    )
    _args, kwargs = batcher.validate_drand_round.call_args
    assert "t_arrival" in kwargs, (
        "/submit must pass t_arrival kwarg (middleware-stamped wall clock) "
        "into validate_drand_round — the bug this regression test pins is "
        "the handler reading time.time() too late, after a stall"
    )
    t_arrival = kwargs["t_arrival"]
    # The stamp must land between t_before and t_after — i.e. the
    # middleware ran in real time on this request, not some cached value.
    assert t_before <= t_arrival <= t_after, (
        f"t_arrival={t_arrival} outside [{t_before}, {t_after}] — "
        "middleware is stamping the wrong instant"
    )


def test_stalled_handler_does_not_reject_round_inside_arrival_window():
    """Simulate the v2.3 prod failure mode: the asyncio loop stalls for
    >30 s after the middleware ran, so by the time ``batcher.validate_drand_round``
    executes the wall clock is many drand rounds ahead of the timestamp
    the middleware recorded.

    With arrival-time stamping, the check uses the middleware timestamp,
    not the (stalled) wall clock. The submission lands inside its round
    and must be accepted, even though a wall-clock-based check would
    reject it as STALE_ROUND.
    """
    import time

    s, batcher = _setup(
        current_checkpoint_hash="sha256:current",
        drand_round_check_enabled=True,
    )

    # The mocked validate_drand_round inspects t_arrival to decide.
    # Real validate_drand_round behaviour: with t_arrival inside the
    # accepted window, returns None (accept); without t_arrival or with
    # a later one, returns STALE_ROUND.
    def _round_check(drand_round, *, t_arrival=None):
        if t_arrival is None:
            return RejectReason.STALE_ROUND  # would happen w/o the fix
        # arrival-stamped: accepted regardless of how late the handler is
        return None
    batcher.validate_drand_round.side_effect = _round_check

    payload = _submission()
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)

    body = r.json()
    assert body["accepted"] is True, (
        f"arrival-stamped submission should pass cheap-reject; got {body}"
    )
    assert body["reason"] != RejectReason.STALE_ROUND.value


def test_seal_extension_http_rejects_late_drand_pre_queue():
    """When the batcher has captured a trigger drand round, HTTP
    cheap-reject must reject any submission with
    ``drand_round > trigger_round`` as BATCH_FILLED without queuing.
    This is the v2.3 seal-extension gate: trigger-round stragglers are
    still accepted (they feed the boundary fair-split), but later-drand
    submissions are too late and don't deserve a worker dequeue."""
    s, batcher = _setup(current_checkpoint_hash="sha256:current")
    # Simulate the batcher having recorded a trigger drand round.
    batcher._seal_trigger_round = 100
    # A submission with drand_round = 101 — later than trigger — must
    # be rejected at the HTTP cheap-reject layer.
    payload = _submission(drand_round=101)
    _assert_pre_queue_reject(s, payload, RejectReason.BATCH_FILLED)


def test_seal_extension_http_accepts_trigger_round_post_trigger():
    """The complement to the previous test: after trigger is recorded,
    submissions WITHIN the trigger drand round must still be accepted
    by HTTP cheap-reject. This is what lets the boundary fair-split
    accumulate > B candidates."""
    s, batcher = _setup(current_checkpoint_hash="sha256:current")
    batcher._seal_trigger_round = 100
    # drand_round == trigger_round → passes the seal-extension gate.
    # (The drand_check is disabled in _setup default, so the request
    # doesn't get rejected for being older than current.)
    payload = _submission(drand_round=100)
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    body = r.json()
    assert body["accepted"] is True, (
        f"trigger-round submission post-trigger must pass cheap-reject; got {body}"
    )
    # Specifically not BATCH_FILLED — that's the false-positive we're
    # guarding against.
    assert body.get("reason") != RejectReason.BATCH_FILLED.value


def test_seal_extension_http_no_change_when_trigger_not_set():
    """Pre-trigger (the common case during the bulk of a window),
    ``batcher._seal_trigger_round is None`` and the new HTTP gate is
    a no-op. Any drand_round value goes through (subject to the other
    cheap-reject gates)."""
    s, batcher = _setup(current_checkpoint_hash="sha256:current")
    # Default _setup leaves _seal_trigger_round = None.
    assert batcher._seal_trigger_round is None
    payload = _submission(drand_round=12345)  # arbitrary, way "later"
    with TestClient(s.app) as client:
        r = client.post("/submit", json=payload)
    body = r.json()
    assert body["accepted"] is True, (
        f"no-trigger submission must pass cheap-reject; got {body}"
    )


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
