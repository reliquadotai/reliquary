"""v6 pays by token; the archive must carry what the payment divides."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reliquary.constants import BFT_ANSWER_BUDGET, BFT_THINKING_BUDGET
from reliquary.validator.admission import AdmissionContext
from reliquary.validator.batch_selection import select_batch_and_distribute
from reliquary.validator.batcher import PendingSubmission
from reliquary.validator.cooldown import CooldownMap

from tests.unit.test_archive_window_content import (
    _FakeEnv,
    _FakeWallet,
    _valid_submission,
)


def test_the_seal_path_never_pays_per_token_under_v6(monkeypatch):
    """R20: v6 payment does not live here.

    The spec removes the auction under v6 and the seal path IS the auction,
    so ``select_batch_and_distribute`` keeps its v4/v5 arithmetic whatever
    the gate says. The token split runs in ``FillClosedBatchAssembler``,
    where the assembled batches -- not the window's slots -- are known.
    """
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
    assert rewards["short"] == rewards["long"] == 0.5


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


def test_selection_telemetry_names_who_actually_paid(monkeypatch):
    """The Minor: ``rewarded``/``reward_amount`` are a SLOT share.

    Under v6 nothing on this path is paid, so reporting a slot share with
    no further qualification tells an operator a number that was never
    credited. Name the payer instead.
    """
    import reliquary.validator.batch_selection as selection

    submissions = [
        _valid_submission(prompt_idx=1, hotkey="short", eos_first=True),
        _valid_submission(prompt_idx=2, hotkey="long", eos_first=True),
    ]
    kwargs = dict(
        b=2,
        cooldown_map=CooldownMap(cooldown_windows=50),
        current_window=42,
        pool=1.0,
    )

    meta = selection.explain_batch_selection(submissions, **kwargs)
    assert {row["payment_source"] for row in meta.values()} == {"slot_share"}

    monkeypatch.setattr(selection, "FILL_CLOSED_ENABLED", True)
    meta = selection.explain_batch_selection(submissions, **kwargs)
    assert {row["payment_source"] for row in meta.values()} == {
        "fill_closed_token_split"
    }


def test_auction_telemetry_names_who_actually_paid(monkeypatch):
    """Same Minor, on the arm production actually runs.

    ``DIFFICULTY_AUCTION_ENFORCE=1`` means ``_finalize_auction_winners``, not
    ``explain_batch_selection``, is what writes ``selection_metadata_by_id``
    in production -- so it is the reporter that drifts under v6.
    """
    from reliquary.validator.batcher import GrpoWindowBatcher
    import reliquary.validator.batcher as batcher_module

    def _finalize():
        stub = SimpleNamespace(
            _valid=[SimpleNamespace(
                hotkey="hk", prompt_idx=1, prompt_content_sha256="c" * 64,
            )],
            difficulty_auction_metadata_by_id={},
            auction_candidates=[],
            reward_alignment={},
            selection_metadata_by_id={},
            rewarded_but_not_selected_by_hotkey={},
            rewards_by_hotkey={},
        )
        GrpoWindowBatcher._finalize_auction_winners(stub, pool=1.0)
        return list(stub.selection_metadata_by_id.values())

    assert {row["payment_source"] for row in _finalize()} == {"slot_share"}

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    assert {row["payment_source"] for row in _finalize()} == {
        "fill_closed_token_split"
    }


def _context(
    eos_token_ids=(99,), environment="openmathinstruct", think_close_ids=(),
):
    return AdmissionContext(
        randomness="cd" * 16,
        environment=environment,
        vocab_size=None,
        max_sequence_length=4096,
        eos_token_ids=eos_token_ids,
        canonical_force_ids=(),
        think_close_ids=think_close_ids,
        bootstrap=False,
        enforce_envelope_signature=False,
        enforce_legacy_merkle=False,
    )


def test_eos_tokens_are_counted_at_admission_and_land_on_the_submission(
    monkeypatch,
):
    """One EOS-terminated rollout of N tokens: it pays N, and the count is
    carried on PendingSubmission rather than recomputed downstream."""
    import reliquary.validator.admission as admission
    monkeypatch.setattr(admission, "FILL_CLOSED_ENABLED", True)

    context = _context()
    eos_terminated = SimpleNamespace(
        commit={
            "tokens": [1, 2, 3, 4, 5, 99],
            "rollout": {"prompt_length": 1, "completion_length": 5},
        }
    )
    request = SimpleNamespace(rollouts=[eos_terminated])

    eos_tokens = admission.count_eos_completion_tokens(request, context)
    assert eos_tokens == 5

    pending = PendingSubmission(
        hotkey="hk",
        prompt_idx=1,
        request=request,
        rewards=[1.0],
        drand_round=0,
        merkle_root=b"\x00" * 32,
        selection_digest=b"\x00" * 32,
        eos_tokens=eos_tokens,
    )
    assert pending.eos_tokens == 5


def _natural_cap_rollout():
    """An ACCEPTED cap shape that never emitted an EOS token.

    ``_classify_termination`` returns ``"ok"`` for this (see
    ``_natural_cap_termination``): openmathinstruct, unforced, a completion
    exactly BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET long that fills the
    token stream, with a think-close token inside phase one. It is
    admitted and graded -- it simply never chose to stop.
    """
    completion = [7] + [3] * (BFT_THINKING_BUDGET + BFT_ANSWER_BUDGET - 1)
    return SimpleNamespace(
        commit={
            "tokens": [1] + completion,
            "rollout": {
                "prompt_length": 1,
                "completion_length": len(completion),
            },
        }
    )


def test_an_accepted_cap_shape_without_eos_pays_nothing(monkeypatch):
    """The single property per-token payment rests on.

    A rollout that ran to the budget without ever emitting EOS is still
    ACCEPTED (``_classify_termination`` -> ``"ok"``), so it is not the
    ``"truncated"`` reject shape. If it were paid for its length, padding
    to the cap would earn the maximum per group and the strictly negative
    margin on padding -- the reason the flat slot share could be removed
    at all -- would invert.
    """
    import reliquary.validator.admission as admission
    monkeypatch.setattr(admission, "FILL_CLOSED_ENABLED", True)

    context = _context(think_close_ids=(7,))
    cap_shape = _natural_cap_rollout()
    # The shape is ACCEPTED, not rejected -- that is what makes it decisive.
    assert admission._classify_termination(cap_shape, context) == "ok"

    request = SimpleNamespace(rollouts=[cap_shape])
    assert admission.count_eos_completion_tokens(request, context) == 0


def test_a_declared_completion_length_cannot_inflate_the_payment(monkeypatch):
    """Payment counts the validator's OWN slice, never a miner's number.

    ``completion_length`` arrives in the miner's commit. Nothing on the
    general path binds it to ``len(tokens)``, so a short EOS-terminated
    rollout can declare a huge one; under a proportional split that buys
    the pool.
    """
    import reliquary.validator.admission as admission
    monkeypatch.setattr(admission, "FILL_CLOSED_ENABLED", True)

    context = _context()
    real_completion = [2] * 499 + [99]
    inflated = SimpleNamespace(
        commit={
            "tokens": [1] + real_completion,
            "rollout": {
                "prompt_length": 1,
                # Ten times the truth.
                "completion_length": 10 * len(real_completion),
            },
        }
    )
    request = SimpleNamespace(rollouts=[inflated])

    # Still admitted: the slice ends on EOS whatever the declaration says.
    assert admission._classify_termination(inflated, context) == "ok"
    assert admission.count_eos_completion_tokens(request, context) == 500


def test_counting_is_gated_off_outside_v6(monkeypatch):
    """v4/v5 pay a flat slot share, so they must not pay a third
    ``_classify_termination`` pass per submission -- and their archives
    carry ``eos_tokens=0``."""
    import reliquary.validator.admission as admission
    monkeypatch.setattr(admission, "FILL_CLOSED_ENABLED", False)

    context = _context()
    eos_terminated = SimpleNamespace(
        commit={
            "tokens": [1, 2, 3, 4, 5, 99],
            "rollout": {"prompt_length": 1, "completion_length": 5},
        }
    )
    calls = []
    real = admission._classify_termination
    monkeypatch.setattr(
        admission, "_classify_termination",
        lambda rollout, ctx: (calls.append(rollout), real(rollout, ctx))[1],
    )

    request = SimpleNamespace(rollouts=[eos_terminated])
    assert admission.count_eos_completion_tokens(request, context) == 0
    assert calls == []


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
