"""_archive_window includes prompt + rollout content on R2."""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.validator.batcher import ValidSubmission
from reliquary.protocol.submission import RolloutSubmission


@dataclass
class _FakeEnv:
    @property
    def name(self): return "fake"
    def __len__(self): return 100
    def get_problem(self, i):
        return {"prompt": f"question {i}", "ground_truth": f"answer {i}", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _rollout(r=1.0, eos=False):
    tokens = [1, 2, 3, 4, 5]
    if eos:
        tokens = tokens + [99]  # 99 = fake eos
    return RolloutSubmission(
        tokens=tokens,
        reward=r,
        commit={
            "tokens": tokens, "proof_version": "v7",
            "rollout": {
                "prompt_length": 2,
                "completion_length": len(tokens) - 2,
                "token_logprobs": [],
            },
        },
        env_name="openmathinstruct",
    )


def _valid_submission(prompt_idx, k=4, hotkey="hk", eos_first=False):
    import math
    rollouts = [
        _rollout(r=1.0 if i < k else 0.0, eos=(eos_first and i == 0))
        for i in range(8)
    ]
    p = k / 8
    sigma = math.sqrt(p * (1 - p))
    return ValidSubmission(
        hotkey=hotkey,
        prompt_idx=prompt_idx,
        merkle_root_bytes=b"\xab" * 32,
        sigma=sigma,
        rollouts=rollouts,
        completion_texts=[f"text_{i}" for i in range(8)],
        sketch_diff_max=412,
        lp_dev_max=0.00037,
        dist_q10_min=0.74,
        all_token_auth_shadow_findings=5,
        all_token_auth_shadow_min_prob=6.0e-7,
        all_token_auth_shadow_positive_findings=2,
        all_token_auth_shadow_positive_min_prob=8.0e-6,
        code_semantic_auth_findings=3,
        code_semantic_auth_min_prob=0.0002,
        code_semantic_auth_positive_findings=1,
        code_semantic_auth_positive_min_prob=0.0004,
        claimed_checkpoint_hash="sha256:fake",
    )


@pytest.mark.asyncio
async def test_archive_includes_prompt_and_rollout_content():
    from reliquary.validator.service import ValidationService

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )

    batcher = MagicMock()
    batcher.window_start = 42
    batcher.randomness = "0xdeadbeef"
    batcher.window_opened_at = 100.0
    batcher.rewarded_but_not_selected_by_hotkey = {"hk_runner": 1}
    from reliquary.validator.batcher import RejectedSubmission
    batcher.reject_counts = {"out_of_zone": 3, "logprob_mismatch": 1}
    batcher.rejected_submissions = [
        RejectedSubmission(
            hotkey="hk_evict", prompt_idx=4, reason="out_of_zone",
        ),
        RejectedSubmission(
            hotkey="hk_grail_cheater", prompt_idx=5, reason="grail_fail",
            # sketch_diff_max intentionally None — set by _reject() in prod.
        ),
    ]

    batch = [
        _valid_submission(prompt_idx=7, k=4, hotkey="hk1", eos_first=True),
        _valid_submission(prompt_idx=13, k=5, hotkey="hk2"),
    ]
    batch[0].arrived_at = 102.5  # 2.5 s after window open
    batch[1].arrived_at = 107.0  # 7.0 s after window open

    runner = _valid_submission(prompt_idx=99, k=4, hotkey="hk_runner")
    runner.arrived_at = 110.0
    runner.rollout_hashes = [bytes([i]) * 32 for i in range(8)]
    batcher.valid_submissions.return_value = list(batch) + [runner]
    batcher.selection_metadata_by_id = {
        id(batch[0]): {"selected_for_batch": True, "rewarded": True},
        id(batch[1]): {"selected_for_batch": True, "rewarded": True},
        id(runner): {
            "selected_for_batch": False,
            "rewarded": True,
            "selection_reason": "same_trigger_round_lost_canonical_ordering",
        },
    }

    captured = {}

    class _StubQueue:
        def enqueue(self, window_start, data):
            captured["window_start"] = window_start
            captured["data"] = data

    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(),
    ):
        await svc._archive_window(batcher, batch)

    assert captured["window_start"] == 42
    archive = captured["data"]
    assert archive["window_start"] == 42
    assert archive["environment"] == "fake"
    assert len(archive["batch"]) == 2

    import math
    entry0 = archive["batch"][0]
    assert entry0["prompt_idx"] == 7
    assert entry0["prompt"] == "question 7"
    assert entry0["ground_truth"] == "answer 7"
    expected_sigma = math.sqrt((4 / 8) * (1 - 4 / 8))  # Bernoulli(p=0.5) → 0.5
    assert abs(entry0["sigma"] - expected_sigma) < 1e-9
    assert len(entry0["rollouts"]) == 8
    assert entry0["rollouts"][0]["tokens"] == [1, 2, 3, 4, 5, 99]  # first is eos-terminated
    assert entry0["rollouts"][1]["tokens"] == [1, 2, 3, 4, 5]
    assert entry0["rollouts"][0]["completion_text"] == "text_0"
    assert entry0["rollouts"][0]["reward"] == 1.0

    # cooldown rebuild backward-compat: still has window_start and batch[*].prompt_idx
    assert {"window_start", "batch"}.issubset(archive.keys())
    assert all("prompt_idx" in e for e in archive["batch"])

    # response_time: seconds between window-open and submission-accepted.
    assert entry0["response_time"] == pytest.approx(2.5)
    assert archive["batch"][1]["response_time"] == pytest.approx(7.0)

    # filter telemetry passed through verbatim.
    assert entry0["sketch_diff_max"] == 412
    assert entry0["lp_dev_max"] == pytest.approx(0.00037)
    assert entry0["dist_q10_min"] == pytest.approx(0.74)
    assert entry0["all_token_auth_shadow_findings"] == 5
    assert entry0["all_token_auth_shadow_min_prob"] == pytest.approx(6.0e-7)
    assert entry0["all_token_auth_shadow_positive_findings"] == 2
    assert entry0["all_token_auth_shadow_positive_min_prob"] == pytest.approx(8.0e-6)
    assert entry0["code_semantic_auth_findings"] == 3
    assert entry0["code_semantic_auth_min_prob"] == pytest.approx(0.0002)
    assert entry0["code_semantic_auth_positive_findings"] == 1
    assert entry0["code_semantic_auth_positive_min_prob"] == pytest.approx(0.0004)

    # forensic fields.
    assert entry0["merkle_root"] == "ab" * 32
    assert entry0["selection_digest"] == "ab" * 32
    assert entry0["claimed_checkpoint_hash"] == "sha256:fake"

    # eos detection: rollout 0 of entry 0 ends with eos_token_id=99 → True.
    assert entry0["rollouts"][0]["eos_terminated"] is True
    assert entry0["rollouts"][1]["eos_terminated"] is False
    assert entry0["rollouts"][0]["completion_length"] == 4  # 6 tokens − 2 prompt
    assert entry0["rollouts"][1]["completion_length"] == 3

    # runners_up: validated submissions that didn't make the batch — metadata only.
    assert "runners_up" in archive
    assert len(archive["runners_up"]) == 1
    ru = archive["runners_up"][0]
    assert ru["hotkey"] == "hk_runner"
    assert ru["prompt_idx"] == 99
    assert ru["response_time"] == pytest.approx(10.0)
    assert ru["all_token_auth_shadow_positive_findings"] == 2
    assert ru["all_token_auth_shadow_positive_min_prob"] == pytest.approx(8.0e-6)
    assert ru["code_semantic_auth_positive_findings"] == 1
    assert ru["code_semantic_auth_positive_min_prob"] == pytest.approx(0.0004)
    assert ru["selection_digest"] == "ab" * 32
    assert "rollouts" not in ru and "prompt" not in ru  # metadata only
    assert ru["rollout_hashes"] == [h.hex() for h in runner.rollout_hashes]
    assert archive["rewarded_but_not_selected_by_hotkey"] == {"hk_runner": 1}

    # reject_summary persisted from batcher.
    assert archive["reject_summary"] == {"out_of_zone": 3, "logprob_mismatch": 1}

    # rejected[] persisted from batcher.rejected_submissions.
    # Test the archive contract WITHOUT pinning the full dataclass field set —
    # adding a new RejectedSubmission field shouldn't break this test, but a
    # missing identity field or an anti-tuning regression must.
    assert "rejected" in archive
    assert len(archive["rejected"]) == 2

    # Required public fields must be present on every entry.
    REQUIRED_KEYS = {"hotkey", "prompt_idx", "reason"}
    for entry in archive["rejected"]:
        assert REQUIRED_KEYS.issubset(entry.keys()), (
            f"archive entry missing required keys: {REQUIRED_KEYS - entry.keys()}"
        )

    evict, grail = archive["rejected"]
    assert (evict["hotkey"], evict["prompt_idx"], evict["reason"]) == (
        "hk_evict", 4, "out_of_zone",
    )
    assert (grail["hotkey"], grail["prompt_idx"], grail["reason"]) == (
        "hk_grail_cheater", 5, "grail_fail",
    )

    # Anti-tuning invariant: GRAIL_FAIL must NOT surface sketch_diff_max in
    # the public archive — even if the dataclass gains new diagnostic fields,
    # this specific value MUST stay scrubbed.
    assert grail["sketch_diff_max"] is None


@pytest.mark.asyncio
async def test_archive_includes_per_rollout_hash():
    """Each rollout in the archive's batch entry carries a hex SHA256 hash."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from reliquary.validator.service import ValidationService

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )

    # Two rollouts with distinct tokens to verify per-rollout hashing.
    r0_tokens = [1, 2, 3, 4]
    r1_tokens = [5, 6, 7, 8]
    valid_sub = _valid_submission(prompt_idx=42)
    valid_sub.rollouts = [
        RolloutSubmission(
            tokens=r0_tokens, reward=1.0,
            commit={"tokens": r0_tokens, "proof_version": "v7",
                    "rollout": {"prompt_length": 2, "completion_length": 2,
                                "token_logprobs": []}},
            env_name="openmathinstruct",
        ),
        RolloutSubmission(
            tokens=r1_tokens, reward=0.0,
            commit={"tokens": r1_tokens, "proof_version": "v7",
                    "rollout": {"prompt_length": 2, "completion_length": 2,
                                "token_logprobs": []}},
            env_name="openmathinstruct",
        ),
    ]
    valid_sub.completion_texts = ["a", "b"]

    from reliquary.validator.dedup import compute_rollout_hash
    valid_sub.rollout_hashes = [
        compute_rollout_hash(r0_tokens),
        compute_rollout_hash(r1_tokens),
    ]

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []
        def valid_submissions(self): return [valid_sub]

    captured = {}

    def _capture_enqueue(window, archive):
        captured["archive"] = archive

    class _StubQueue:
        def enqueue(self, w, a):
            _capture_enqueue(w, a)

    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(),
    ):
        await svc._archive_window(_FakeBatcher(), [valid_sub])

    archive = captured["archive"]
    entry = archive["batch"][0]
    assert len(entry["rollouts"]) == 2
    assert entry["rollouts"][0]["hash"] == compute_rollout_hash(r0_tokens).hex()
    assert entry["rollouts"][1]["hash"] == compute_rollout_hash(r1_tokens).hex()


@pytest.mark.asyncio
async def test_archive_includes_late_drops_and_clears_counter():
    """First archive snapshot captures recorded late drops and resets the
    counter; a subsequent archive with no events emits an empty dict."""
    from unittest.mock import MagicMock, patch
    from reliquary.validator.service import ValidationService

    fake_tok = MagicMock()
    fake_tok.eos_token_id = 99
    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=fake_tok,
        env=_FakeEnv(), netuid=99,
    )
    svc.record_late_drop("hkA", "window_not_active")
    svc.record_late_drop("hkA", "window_not_active")
    svc.record_late_drop("hkB", "worker_dropped")

    valid_sub = _valid_submission(prompt_idx=42)

    class _FakeBatcher:
        window_start = 500
        randomness = "abcd"
        window_opened_at = 0.0
        reject_counts: dict = {}
        rejected_submissions: list = []
        def valid_submissions(self): return [valid_sub]

    captured = []

    class _StubQueue:
        def enqueue(self, w, a):
            captured.append(a)

    with patch(
        "reliquary.infrastructure.archive_queue.get_archive_queue",
        return_value=_StubQueue(),
    ):
        await svc._archive_window(_FakeBatcher(), [valid_sub])
        # Populated archive carries the snapshot; counter is now reset.
        assert captured[-1]["late_drops"] == {
            "hkA": {"window_not_active": 2},
            "hkB": {"worker_dropped": 1},
        }
        assert svc._late_drops == {}

        # Second archive run with no new events must emit an empty dict.
        await svc._archive_window(_FakeBatcher(), [valid_sub])
        assert captured[-1]["late_drops"] == {}
