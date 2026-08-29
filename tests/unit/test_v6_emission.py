"""v6 pays by token; the archive must carry what the payment divides."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reliquary.validator.admission import AdmissionContext
from reliquary.validator.batch_selection import select_batch_and_distribute
from reliquary.validator.batcher import PendingSubmission
from reliquary.validator.cooldown import CooldownMap

from tests.unit.test_archive_window_content import (
    _FakeEnv,
    _FakeWallet,
    _valid_submission,
)


def test_v6_rewards_are_proportional_to_eos_tokens(monkeypatch):
    """Two accepted groups, nine times the tokens, nine times the share."""
    import reliquary.validator.batch_selection as selection
    monkeypatch.setattr(selection, "FILL_CLOSED_ENABLED", True)

    batch, rewards = select_batch_and_distribute(
        [
            _valid_submission(
                prompt_idx=1, hotkey="short", eos_first=True, eos_tokens=1_000
            ),
            _valid_submission(
                prompt_idx=2, hotkey="long", eos_first=True, eos_tokens=9_000
            ),
        ],
        b=2,
        cooldown_map=CooldownMap(cooldown_windows=50),
        current_window=42,
        pool=1.0,
    )

    assert len(batch) == 2
    assert abs(rewards["short"] - 0.1) < 1e-9
    assert abs(rewards["long"] - 0.9) < 1e-9


def test_the_auction_path_is_untouched_when_the_gate_is_off():
    """v4 and v5 keep the flat slot share, byte for byte."""
    _batch, rewards = select_batch_and_distribute(
        [
            _valid_submission(
                prompt_idx=1, hotkey="short", eos_first=True, eos_tokens=1_000
            ),
            _valid_submission(
                prompt_idx=2, hotkey="long", eos_first=True, eos_tokens=9_000
            ),
        ],
        b=2,
        cooldown_map=CooldownMap(cooldown_windows=50),
        current_window=42,
        pool=1.0,
    )

    assert rewards["short"] == rewards["long"]


def _context(eos_token_ids=(99,), environment="openmathinstruct"):
    return AdmissionContext(
        randomness="cd" * 16,
        environment=environment,
        vocab_size=None,
        max_sequence_length=4096,
        eos_token_ids=eos_token_ids,
        canonical_force_ids=(),
        think_close_ids=(),
        bootstrap=False,
        enforce_envelope_signature=False,
        enforce_legacy_merkle=False,
    )


def test_eos_tokens_are_counted_at_admission_and_exclude_cap_hits(monkeypatch):
    """One EOS-terminated rollout of N tokens, one cap-hit rollout: only the
    EOS-terminated one pays, and the count lands on PendingSubmission."""
    import reliquary.validator.admission as admission

    context = _context()
    # Cap-hit classification needs prompt_length + completion_length to
    # reach the protocol cap without an EOS token present.
    monkeypatch.setattr(admission, "max_new_tokens_for_environment", lambda env: 8)

    eos_terminated = SimpleNamespace(
        commit={
            "tokens": [1, 2, 3, 4, 5, 99],
            "rollout": {"prompt_length": 1, "completion_length": 5},
        }
    )
    cap_hit = SimpleNamespace(
        commit={
            "tokens": [1] + [7] * 8,
            "rollout": {"prompt_length": 1, "completion_length": 8},
        }
    )
    request = SimpleNamespace(rollouts=[eos_terminated, cap_hit])

    eos_tokens = admission.count_eos_completion_tokens(request, context)
    assert eos_tokens == 5

    pending = PendingSubmission(
        hotkey="hk",
        prompt_idx=1,
        request=request,
        rewards=[1.0, 0.0],
        drand_round=0,
        merkle_root=b"\x00" * 32,
        selection_digest=b"\x00" * 32,
        eos_tokens=eos_tokens,
    )
    assert pending.eos_tokens == 5


async def _archive_one_v6_window():
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
    batcher.window_opened_wall_ts = 1_000.0
    batcher.difficulty_auction_enabled = True
    batcher.force_seal_reason = None
    batcher.rewarded_but_not_selected_by_hotkey = {}
    batcher.reward_alignment = {}
    batcher.logical_group_reservation_count = 0
    batcher.logical_group_duplicate_rejects = 0
    batcher.grader_failures = {}
    batcher.reject_counts = {}
    batcher.rejected_submissions = []
    batcher.selection_metadata_by_id = {}
    batcher.difficulty_auction_metadata_by_id = {}
    batcher.difficulty_auction_shadow = {}

    batch = [
        _valid_submission(prompt_idx=7, hotkey="hk1", eos_tokens=42),
        _valid_submission(prompt_idx=13, hotkey="hk2", eos_tokens=99),
    ]
    batcher.valid_submissions.return_value = list(batch)

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

    return captured["data"]


@pytest.mark.asyncio
async def test_the_archive_records_eos_tokens_per_accepted_group(monkeypatch):
    """The weight-only replay divides by tokens, so the archive must carry
    them or two validators cannot converge on the same weights."""
    import reliquary.validator.service as service
    monkeypatch.setattr(service, "FILL_CLOSED_ENABLED", True)

    archive = await _archive_one_v6_window()

    for entry in archive["batch"]:
        assert isinstance(entry["eos_tokens"], int)
