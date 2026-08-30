"""v6 pays by token; the archive must carry what the payment divides."""
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from reliquary.constants import (
    BFT_ANSWER_BUDGET,
    BFT_THINKING_BUDGET,
    FILL_CLOSED_EMISSIONS_PER_WINDOW,
)
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


# --- R20: payment lives in the assembler, not the seal path ------------

ENV_ORDER = ["openmathinstruct", "opencodeinstruct"]


def _paying_group(env, hotkey, eos_tokens, prompt_idx):
    from tests.unit.test_training_payload_codec import _group, _roll

    group = _group([_roll(1.0, 4, env=env)], prompt_idx=prompt_idx)
    group.hotkey = hotkey
    group.eos_tokens = eos_tokens
    return group


def _assembler(monkeypatch, window=42, window_pool=1.0):
    from reliquary.validator.fill_closed_batch_assembler import (
        FillClosedBatchAssembler,
    )
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    return FillClosedBatchAssembler(
        window_start=window,
        env_order=ENV_ORDER,
        enqueue_fn=lambda key, data: None,
        tombstone_fn=lambda key, data: None,
        window_pool=window_pool,
    )


def _one_batch(assembler, math_groups, code_groups, window=42):
    """Force exactly one assembled batch out of partial chunks."""
    assembler.accept("openmathinstruct", math_groups, window, "rev")
    assembler.accept("opencodeinstruct", code_groups, window, "rev")
    assembler.close()


def test_the_assembler_splits_a_batch_pool_by_eos_tokens(monkeypatch):
    """R20 + R15: nine times the tokens, nine times the share."""
    assembler = _assembler(monkeypatch)
    _one_batch(
        assembler,
        [
            _paying_group("openmathinstruct", "short", 1_000, 1),
            _paying_group("openmathinstruct", "long", 9_000, 2),
        ],
        [_paying_group("opencodeinstruct", "coder", 5_000, 3)],
    )

    rewards = assembler.reward_map()
    # One batch draws window_pool / envs / emissions-per-window per env.
    env_batch_pool = 1.0 / len(ENV_ORDER) / FILL_CLOSED_EMISSIONS_PER_WINDOW
    assert abs(rewards["short"] - 0.1 * env_batch_pool) < 1e-12
    assert abs(rewards["long"] - 0.9 * env_batch_pool) < 1e-12


def test_one_environments_token_mass_cannot_eat_the_others_share(monkeypatch):
    """Each environment keeps its own pool, exactly as the seal path's
    ``pool_per_env`` does -- otherwise a long-completion environment
    takes a short one's emission through raw token mass."""
    assembler = _assembler(monkeypatch)
    _one_batch(
        assembler,
        [_paying_group("openmathinstruct", "mather", 1_000, 1)],
        [_paying_group("opencodeinstruct", "coder", 1_000_000, 2)],
    )

    rewards = assembler.reward_map()
    assert abs(rewards["mather"] - rewards["coder"]) < 1e-12


def test_a_group_with_no_eos_tokens_is_paid_nothing(monkeypatch):
    """A group whose every rollout hit the cap without EOS pays zero --
    the admission-side property, carried through to the split."""
    assembler = _assembler(monkeypatch)
    _one_batch(
        assembler,
        [
            _paying_group("openmathinstruct", "padder", 0, 1),
            _paying_group("openmathinstruct", "finisher", 500, 2),
        ],
        [_paying_group("opencodeinstruct", "coder", 500, 3)],
    )

    rewards = assembler.reward_map()
    assert "padder" not in rewards
    env_batch_pool = 1.0 / len(ENV_ORDER) / FILL_CLOSED_EMISSIONS_PER_WINDOW
    assert abs(rewards["finisher"] - env_batch_pool) < 1e-12


def test_one_batch_draws_exactly_its_even_share_of_the_window(monkeypatch):
    """The divisor: one window's pool spread evenly over its batches, so
    the totals match a once-per-window split."""
    assembler = _assembler(monkeypatch)
    _one_batch(
        assembler,
        [_paying_group("openmathinstruct", "mather", 700, 1)],
        [_paying_group("opencodeinstruct", "coder", 700, 2)],
    )

    total = sum(assembler.reward_map().values())
    assert abs(total - 1.0 / FILL_CLOSED_EMISSIONS_PER_WINDOW) < 1e-12


def test_a_quarantined_batch_still_pays(monkeypatch):
    """Quarantine protects model state, not emission -- the seal path
    says so in as many words (service.py: "Rewards and archives remain
    per-window; this gate only protects model state"). A miner whose
    proven group lands in a batch some OTHER miner poisoned must not
    lose its pay for it."""
    import reliquary.validator.fill_closed_batch_assembler as module

    assembler = _assembler(monkeypatch)
    monkeypatch.setattr(
        module, "assess_training_batch",
        lambda batch, reject_counts: SimpleNamespace(
            quarantined=True, reasons=["poisoned"],
            to_archive=lambda: {"quarantined": True},
        ),
    )
    _one_batch(
        assembler,
        [_paying_group("openmathinstruct", "mather", 700, 1)],
        [_paying_group("opencodeinstruct", "coder", 700, 2)],
    )

    assert assembler.reward_map()["mather"] > 0.0


async def _archive_one_v6_window(assembler=None):
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

    if assembler is not None:
        svc._fill_closed_assemblers[assembler.window_start] = assembler

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


@pytest.mark.asyncio
async def test_the_archive_pays_from_the_assembler_not_the_auction(monkeypatch):
    """R20: under v6 the seal path pays nothing, so the authoritative
    per-hotkey emission in the archive -- what a weight-only validator
    replays -- must be the assembler's token split."""
    import reliquary.validator.service as service
    monkeypatch.setattr(service, "FILL_CLOSED_ENABLED", True)

    assembler = _assembler(monkeypatch, window=42)
    _one_batch(
        assembler,
        [_paying_group("openmathinstruct", "mather", 700, 1)],
        [_paying_group("opencodeinstruct", "coder", 2_100, 2)],
    )

    archive = await _archive_one_v6_window(assembler=assembler)

    assert archive["rewards_by_hotkey"] == assembler.reward_map()
    assert set(archive["rewards_by_hotkey"]) == {"mather", "coder"}


@pytest.mark.asyncio
async def test_the_archive_closes_the_window_it_is_archiving(monkeypatch):
    """The remainder batch is emitted by ``close()``. In serial mode
    ``close()`` otherwise runs at the NEXT window's open -- after this
    archive was already written -- so its pay would never reach the
    archive that a weight-only validator replays."""
    import reliquary.validator.service as service
    monkeypatch.setattr(service, "FILL_CLOSED_ENABLED", True)

    assembler = _assembler(monkeypatch, window=42)
    # Chunks handed over, but nothing closed: the remainder is still held.
    assembler.accept(
        "openmathinstruct",
        [_paying_group("openmathinstruct", "mather", 700, 1)],
        42, "rev",
    )
    assembler.accept(
        "opencodeinstruct",
        [_paying_group("opencodeinstruct", "coder", 700, 2)],
        42, "rev",
    )
    assert assembler.reward_map() == {}

    archive = await _archive_one_v6_window(assembler=assembler)

    assert set(archive["rewards_by_hotkey"]) == {"mather", "coder"}


# --- I1: the direct admission path must carry eos_tokens too ----------


def _eos_terminated_request(prompt_idx=11, hotkey="miner"):
    """A ``_request`` whose every rollout ends on the tokenizer's EOS id
    (99, see ``_make_batcher``'s default tokenizer)."""
    from tests.unit.test_grpo_window_batcher import _request

    request = _request(prompt_idx=prompt_idx, hotkey=hotkey)
    for rollout in request.rollouts:
        tokens = list(rollout.commit["tokens"]) + [99]
        rollout.tokens = tokens
        rollout.commit["tokens"] = tokens
        meta = rollout.commit["rollout"]
        meta["completion_length"] = int(meta["completion_length"]) + 1
        rollout.commit["commitments"] = [{"sketch": 0} for _ in tokens]
        meta["token_logprobs"] = [0.0] * len(tokens)
    return request


def _arm_direct_path(monkeypatch):
    import reliquary.validator.admission as admission_module
    import reliquary.validator.batcher as batcher_module

    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)
    monkeypatch.setattr(admission_module, "FILL_CLOSED_ENABLED", True)


def test_a_group_through_the_direct_path_is_paid(monkeypatch):
    """I1: ``_accept_locked`` -- the compatibility path for callers that do
    not go through the process-isolated admission worker -- built its
    ``PendingSubmission`` without ``eos_tokens``, so every group admitted
    through it defaulted to 0 and the token split paid it nothing."""
    from tests.unit.test_grpo_window_batcher import _make_batcher

    _arm_direct_path(monkeypatch)
    batcher = _make_batcher()
    batcher.difficulty_auction_enabled = True

    response = batcher.accept_submission(_eos_terminated_request())

    assert response.accepted is True
    assert batcher.pending_submissions()[-1].eos_tokens > 0


def test_the_direct_path_counts_nothing_when_the_gate_is_off():
    """R21: v4/v5 must not spend a third ``_classify_termination`` pass per
    submission, and their archives carry 0."""
    from tests.unit.test_grpo_window_batcher import _make_batcher

    batcher = _make_batcher()
    batcher.difficulty_auction_enabled = True

    batcher.accept_submission(_eos_terminated_request())

    assert batcher.pending_submissions()[-1].eos_tokens == 0


def test_the_direct_path_pays_only_genuine_eos_terminations(monkeypatch):
    """The same restriction the prepared path enforces: a rollout that ran
    to the cap without EOS contributes no paid tokens."""
    from tests.unit.test_grpo_window_batcher import _make_batcher, _request

    _arm_direct_path(monkeypatch)
    batcher = _make_batcher()
    batcher.difficulty_auction_enabled = True

    batcher.accept_submission(_request(prompt_idx=12, hotkey="capper"))

    assert batcher.pending_submissions()[-1].eos_tokens == 0


# --- I4 (R24): the archive's batch is what was PAID -------------------


def _paid_valid_submission(prompt_idx, hotkey, eos_tokens):
    return _valid_submission(
        prompt_idx=prompt_idx, hotkey=hotkey, eos_first=True,
        eos_tokens=eos_tokens,
    )


def _single_env_assembler(monkeypatch, window=42, b_batch=None):
    from reliquary.validator import fill_closed_batch_assembler as module
    from reliquary.validator.fill_closed_batch_assembler import (
        FillClosedBatchAssembler,
    )
    import reliquary.infrastructure.training_payload_queue as queue_module

    monkeypatch.setattr(queue_module, "FILL_CLOSED_ENABLED", True)
    if b_batch is not None:
        # Shrink a full batch so one window assembles several of them
        # without fabricating 16 groups apiece.
        monkeypatch.setattr(module, "B_BATCH", b_batch)
    return FillClosedBatchAssembler(
        window_start=window,
        env_order=["fake"],
        enqueue_fn=lambda key, data: None,
        tombstone_fn=lambda key, data: None,
        window_pool=1.0,
    )


@pytest.mark.asyncio
async def test_the_archive_batch_is_the_assembler_paid_set(monkeypatch):
    """R24: with the seal path selecting nothing under v6, the archive's
    ``batch`` must come from the assembler -- the auction's winners are a
    different set (usually empty) and a weight-only validator replaying the
    reward map needs the groups the map was computed over."""
    import reliquary.validator.service as service
    monkeypatch.setattr(service, "FILL_CLOSED_ENABLED", True)

    assembler = _single_env_assembler(monkeypatch)
    assembler.accept(
        "fake",
        [
            _paid_valid_submission(101, "paid1", 1_000),
            _paid_valid_submission(102, "paid2", 3_000),
        ],
        42, "rev",
    )

    archive = await _archive_one_v6_window(assembler=assembler)

    archived = {(e["hotkey"], e["prompt_idx"]) for e in archive["batch"]}
    assert archived == {("paid1", 101), ("paid2", 102)}
    assert set(archive["rewards_by_hotkey"]) == {"paid1", "paid2"}


@pytest.mark.asyncio
async def test_the_archived_batch_replays_the_reward_map_per_batch(monkeypatch):
    """R28: a weight-only validator divides the pool over the archive's
    ``eos_tokens`` -- but payment is per assembled BATCH, so the replay
    has to divide per batch too. The archive carries ``batch_index`` per
    entry for exactly that; without it the replay is only right for a
    single-batch window, which no real window is."""
    import reliquary.validator.service as service
    monkeypatch.setattr(service, "FILL_CLOSED_ENABLED", True)

    assembler = _single_env_assembler(monkeypatch, b_batch=3)
    # Batch 0 fills on arrival at B_BATCH=3 and is written immediately.
    assembler.accept(
        "fake",
        [
            _paid_valid_submission(101, "a", 1_000),
            _paid_valid_submission(102, "b", 3_000),
            _paid_valid_submission(103, "d", 4_000),
        ],
        42, "rev",
    )
    # Batch 1 is the partial remainder close() forces out at archive.
    # "b" straddles the two batches -- the case a flat replay gets wrong.
    assembler.accept(
        "fake",
        [
            _paid_valid_submission(104, "b", 1_000),
            _paid_valid_submission(105, "c", 1_000),
        ],
        42, "rev",
    )

    archive = await _archive_one_v6_window(assembler=assembler)

    entries = archive["batch"]
    assert {int(e["batch_index"]) for e in entries} == {0, 1}
    live = archive["rewards_by_hotkey"]

    env_batch_pool = 1.0 / 1 / FILL_CLOSED_EMISSIONS_PER_WINDOW
    by_batch: dict[int, list] = {}
    for entry in entries:
        by_batch.setdefault(int(entry["batch_index"]), []).append(entry)
    replayed: dict[str, float] = {}
    for batch in by_batch.values():
        batch_tokens = sum(int(e["eos_tokens"]) for e in batch)
        for entry in batch:
            share = env_batch_pool * int(entry["eos_tokens"]) / batch_tokens
            replayed[entry["hotkey"]] = (
                replayed.get(entry["hotkey"], 0.0) + share
            )

    assert set(replayed) == set(live) == {"a", "b", "c", "d"}
    for hotkey, share in replayed.items():
        assert abs(share - live[hotkey]) < 1e-12


@pytest.mark.asyncio
async def test_a_flat_replay_of_a_multi_batch_window_is_wrong(monkeypatch):
    """The counter-example R28 rests on: dividing the whole window's pool
    over the whole window's tokens does NOT reproduce the live map once a
    hotkey is paid in two batches. Pinned so a future reader cannot mistake
    the single-batch case for the general one."""
    import reliquary.validator.service as service
    monkeypatch.setattr(service, "FILL_CLOSED_ENABLED", True)

    assembler = _single_env_assembler(monkeypatch, b_batch=3)
    assembler.accept(
        "fake",
        [
            _paid_valid_submission(101, "a", 1_000),
            _paid_valid_submission(102, "b", 3_000),
            _paid_valid_submission(103, "d", 4_000),
        ],
        42, "rev",
    )
    assembler.accept(
        "fake",
        [
            _paid_valid_submission(104, "b", 1_000),
            _paid_valid_submission(105, "c", 1_000),
        ],
        42, "rev",
    )

    archive = await _archive_one_v6_window(assembler=assembler)

    entries = archive["batch"]
    live = archive["rewards_by_hotkey"]
    window_pool = sum(live.values())
    window_tokens = sum(int(e["eos_tokens"]) for e in entries)
    flat: dict[str, float] = {}
    for entry in entries:
        share = window_pool * int(entry["eos_tokens"]) / window_tokens
        flat[entry["hotkey"]] = flat.get(entry["hotkey"], 0.0) + share

    assert abs(flat["b"] - live["b"]) > 1e-6


# --- Minor: a duplicate payload digest is refused at precommit --------


def _precommit_batcher(monkeypatch=None, **overrides):
    from tests.unit.test_grpo_window_batcher import _make_batcher

    batcher = _make_batcher(**overrides)
    batcher.difficulty_auction_enabled = True
    batcher._operator_for_hotkey = lambda hotkey: f"op-{hotkey}"
    return batcher


def _precommit(batcher, receipt_id, digest, hotkey="miner"):
    return batcher.try_register_upload_precommit(
        receipt_id,
        hotkey,
        t_arrival_wall=batcher.window_opened_wall_ts,
        payload_bytes=1234,
        payload_sha256=digest,
    )


def test_a_duplicate_payload_digest_is_refused_before_upload(monkeypatch):
    """Under the flat slot share a resubmitted group won a duplicate slot at
    worst; under per-token payment it collects the SAME tokens twice. The
    precommit already carries the digest, so the refusal costs nothing and
    lands before any payload moves."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _precommit_batcher()
    digest = "ab" * 32

    assert _precommit(batcher, "r1", digest)[0] is True
    accepted, reason, deadline = _precommit(batcher, "r2", digest)

    assert accepted is False
    assert reason == "precommit_duplicate_payload"
    assert deadline is None


def test_a_distinct_payload_digest_is_still_accepted(monkeypatch):
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _precommit_batcher()

    assert _precommit(batcher, "r1", "ab" * 32)[0] is True
    assert _precommit(batcher, "r2", "cd" * 32)[0] is True


def test_the_digest_guard_is_inert_when_the_gate_is_off():
    """v4/v5 keep their behaviour byte for byte: the flat slot share made a
    resubmission cost a slot, not a second payment."""
    batcher = _precommit_batcher()
    digest = "ab" * 32

    assert _precommit(batcher, "r1", digest)[0] is True
    assert _precommit(batcher, "r2", digest)[0] is True


# --- R29: an expired precommit gives its payload digest back ----------


def test_an_expired_precommit_releases_its_payload_digest(monkeypatch):
    """The digest is burned at precommit-ACCEPT, before the body has moved.
    A miner whose upload fails and whose receipt then expires must be able
    to retry the same body: otherwise it is locked out for the rest of the
    window, the exact shape of the no-reveal circuit regression that made
    100% of ``rate_limited`` rejects honest operators."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    clock = [1_000.0]
    batcher = _precommit_batcher(time_fn=lambda: clock[0])
    digest = "ab" * 32

    accepted, _reason, deadline = _precommit(batcher, "r1", digest)
    assert accepted is True

    # The upload never lands; the receipt ages past its own deadline and is
    # pruned on the next registration.
    clock[0] = float(deadline) + 1.0

    assert _precommit(batcher, "r2", digest)[0] is True


def test_an_explicitly_expired_precommit_releases_its_digest(monkeypatch):
    """Same release on the caller-driven expiry path (the seal drain and
    ``resolve_upload_precommit(expired=True)``), not just the pruner."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _precommit_batcher()
    digest = "ab" * 32

    assert _precommit(batcher, "r1", digest)[0] is True
    assert batcher.resolve_upload_precommit("r1", expired=True) is True

    assert _precommit(batcher, "r2", digest)[0] is True


def test_a_revealed_precommit_keeps_its_digest_burned(monkeypatch):
    """The dedup itself must survive: once the body has been revealed, the
    same digest is a resubmission collecting the same tokens twice, however
    the receipt ends afterwards."""
    import reliquary.validator.batcher as batcher_module
    monkeypatch.setattr(batcher_module, "FILL_CLOSED_ENABLED", True)

    batcher = _precommit_batcher()
    digest = "ab" * 32

    assert _precommit(batcher, "r1", digest)[0] is True
    assert batcher.mark_upload_precommit_revealed("r1") is True
    assert batcher.resolve_upload_precommit("r1", expired=True) is True

    accepted, reason, _deadline = _precommit(batcher, "r2", digest)
    assert accepted is False
    assert reason == "precommit_duplicate_payload"
