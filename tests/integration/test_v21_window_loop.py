"""v2.1 window state-machine smoke: open → seal → train → publish → ready.

Fully-mocked — no R2, no torch, no real drand. Exercises the
orchestration contract: _open_window advances window_n, seal_event
drives _train_and_publish, checkpoint_n increments, state ends on READY.
"""

from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from reliquary.constants import B_BATCH, CHALLENGE_K, M_ROLLOUTS
from reliquary.protocol.submission import (
    BatchSubmissionRequest,
    RolloutSubmission,
    WindowState,
)


def _always_true_proof(commit, model, randomness):
    import torch
    from reliquary.validator.verifier import ProofResult
    return ProofResult(all_passed=True, passed=1, checked=1, logits=torch.empty(0))


@dataclass
class _FakeEnv:
    @property
    def name(self): return "fake"
    def __len__(self): return 1000
    def get_problem(self, i): return {"prompt": f"p{i}", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0 if "WIN" in c else 0.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _make_commit(
    *,
    tokens: list[int] | None = None,
    prompt_length: int = 4,
    success: bool = False,
    total_reward: float = 0.0,
) -> dict:
    """Build a minimal commit that passes CommitModel.model_validate."""
    if tokens is None:
        tokens = list(range(CHALLENGE_K + prompt_length))
    seq_len = len(tokens)
    completion_length = seq_len - prompt_length
    return {
        "tokens": tokens,
        "commitments": [{"sketch": 0} for _ in range(seq_len)],
        "proof_version": "v6",
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


def _rollouts(k, seed: int = 0):
    """Build M_ROLLOUTS rollouts.  ``seed`` offsets token values so
    submissions from different miners produce distinct hashes.
    Each rollout within a submission is also unique (different ``i`` offset)
    so the within-submission dedup check does not fire."""
    rollouts = []
    for i in range(M_ROLLOUTS):
        reward = 1.0 if i < k else 0.0
        tokens = [seed * 1000 + i * 100 + j for j in range(CHALLENGE_K + 4)]
        commit = _make_commit(success=reward > 0.5, total_reward=reward, tokens=tokens)
        rollouts.append(RolloutSubmission(
            tokens=commit["tokens"],
            reward=reward,
            commit=commit,
            env_name="openmathinstruct",
        ))
    return rollouts


def _make_service(checkpoint_hash="sha256:cp"):
    """Construct a ValidationService with a mocked checkpoint store (no HF) +
    training stub.

    Sets publish_every=1 so every window triggers a publish, keeping
    existing test assertions simple.
    """
    from reliquary.validator.service import ValidationService
    from reliquary.validator.checkpoint import ManifestEntry

    svc = ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )
    # Publish on every window so counter tests stay simple.
    svc._publish_every = 1

    # Mock checkpoint store — start with a manifest so batcher gets a revision,
    # then return incrementing ManifestEntries on publish.
    svc._checkpoint_store = MagicMock()
    svc._checkpoint_store.current_manifest = MagicMock(return_value=ManifestEntry(
        checkpoint_n=0,
        repo_id="aivolutionedge/reliquary-sn",
        revision=checkpoint_hash,
        signature="ed25519:sig0",
    ))
    svc._checkpoint_store.publish = AsyncMock(side_effect=lambda checkpoint_n, model: ManifestEntry(
        checkpoint_n=checkpoint_n,
        repo_id="aivolutionedge/reliquary-sn",
        revision=f"rev_sha_{checkpoint_n:03d}",
        signature=f"ed25519:sig{checkpoint_n}",
    ))

    # Mock R2 archive upload.
    import reliquary.validator.service as svc_mod
    svc_mod.storage.upload_window_dataset = AsyncMock(return_value=True)

    return svc


def _patch_open_grpo_window(svc):
    """Return a context manager that replaces open_grpo_window in the service
    module with a version that injects mock verifiers (no torch, no crypto).

    The mock batcher uses the same GrpoWindowBatcher constructor but overrides
    the three verifier callables so accept_submission works without torch.
    """
    import reliquary.validator.service as svc_mod
    from reliquary.validator.batcher import GrpoWindowBatcher
    from reliquary.validator.cooldown import CooldownMap
    from reliquary.constants import BATCH_PROMPT_COOLDOWN_WINDOWS

    real_open = svc_mod.open_grpo_window

    def _mock_open(
        window_start, env, model, *,
        cooldown_map, hash_set, tokenizer, bootstrap=False,
        queue_drained_predicate=None,
    ):
        b = GrpoWindowBatcher(
            window_start=window_start,
            env=env,
            model=model,
            cooldown_map=cooldown_map,
            hash_set=hash_set,
            bootstrap=bootstrap,
            queue_drained_predicate=queue_drained_predicate,
            # Stub out torch-dependent verifiers.
            verify_commitment_proofs_fn=_always_true_proof,
            verify_signature_fn=lambda c, h: True,
            # Decode via reward: FakeEnv.compute_reward returns 1.0 for "WIN".
            completion_text_fn=lambda rollout: "WIN" if rollout.reward > 0.5 else "",
            # Integration tests don't drive wall clock; disable the drand
            # timing gate so submissions with drand_round=0 still accept.
            drand_round_check_enabled=False,
        )
        # Match the per-window randomness used in ``_make_commit`` so the
        # WRONG_RANDOMNESS check from PR #23 accepts the test requests.
        b.randomness = "cd" * 16
        return b

    return patch.object(svc_mod, "open_grpo_window", side_effect=_mock_open)


@pytest.mark.asyncio
async def test_one_window_lap_bumps_counters(monkeypatch):
    """Open → manually fire seal_event → train_and_publish →
    window_n and checkpoint_n both bumped, state = READY.

    Patches B_BATCH to 0 so the empty sealed batch counts as "full" (real
    submission-driven bump is covered by test_batch_full_bumps_counters).
    """
    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)

    svc = _make_service()

    initial_wn = svc._window_n
    initial_cn = svc._checkpoint_n

    with _patch_open_grpo_window(svc):
        svc._open_window()
    svc._activate_window()
    assert svc._current_window_state == WindowState.OPEN
    assert svc._window_n == initial_wn + 1

    # Simulate the B-th valid submission arriving.
    svc._active_batcher.seal_event.set()

    await svc._train_and_publish()

    assert svc._checkpoint_n == initial_cn + 1
    assert svc._current_window_state == WindowState.READY
    assert svc._active_batcher is None
    svc._checkpoint_store.publish.assert_awaited_once()


@pytest.mark.asyncio
async def test_open_window_passes_verify_model_to_batcher(monkeypatch):
    """The batcher must receive verify_model (not train_model) so
    verify_commitment_proofs runs against the published checkpoint."""
    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)
    svc = _make_service()

    # Tag the two models so we can identify which one reaches the batcher.
    svc.train_model = MagicMock(name="train_model_sentinel")
    svc.verify_model = MagicMock(name="verify_model_sentinel")

    captured = {}

    import reliquary.validator.service as svc_mod
    real_open = svc_mod.open_grpo_window

    def _capture_open(
        window_start, env, model, *,
        cooldown_map, hash_set, tokenizer, bootstrap=False,
        queue_drained_predicate=None,
    ):
        captured["model"] = model
        # Return a minimal batcher stub so the rest of _open_window doesn't crash
        from reliquary.validator.batcher import GrpoWindowBatcher
        return GrpoWindowBatcher(
            window_start=window_start, env=env, model=model,
            cooldown_map=cooldown_map, hash_set=hash_set, bootstrap=bootstrap,
            queue_drained_predicate=queue_drained_predicate,
            verify_commitment_proofs_fn=_always_true_proof,
            verify_signature_fn=lambda c, h: True,
            completion_text_fn=lambda r: "",
            drand_round_check_enabled=False,
        )

    with patch.object(svc_mod, "open_grpo_window", side_effect=_capture_open):
        svc._open_window()

    assert captured["model"] is svc.verify_model
    assert captured["model"] is not svc.train_model


@pytest.mark.asyncio
async def test_verify_model_refreshed_only_after_publish(monkeypatch):
    """verify_model.load_state_dict(train_model.state_dict()) must run
    after a successful publish, and ONLY after a publish (not on windows
    where publish is skipped)."""
    import torch
    import torch.nn as nn

    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)
    svc = _make_service()

    # Real (tiny) models so load_state_dict has something to copy.
    train = nn.Linear(4, 4)
    verify = nn.Linear(4, 4)
    # Force the two to start identical so divergence is measurable.
    verify.load_state_dict(train.state_dict())
    svc.train_model = train
    svc.verify_model = verify

    # Stub train_step to mutate train_model (simulate a real grad step).
    def _fake_train_step(model, batch, *, ref_model, window_index=None):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        return model

    import reliquary.validator.service as svc_mod
    monkeypatch.setattr(svc_mod, "train_step", _fake_train_step)

    # publish_every=1 by default in _make_service → publish runs every window.
    with _patch_open_grpo_window(svc):
        svc._open_window()
    svc._active_batcher.seal_event.set()
    # Pretend the seal produced a full batch so trained=True.
    svc._active_batcher.seal_batch = MagicMock(return_value=([MagicMock()] * 100, {}))

    await svc._train_and_publish()

    # After publish, verify_model should equal the mutated train_model.
    for p_t, p_v in zip(train.parameters(), verify.parameters()):
        assert torch.equal(p_t, p_v), "verify_model not refreshed after publish"


@pytest.mark.asyncio
async def test_verify_model_NOT_refreshed_when_publish_skipped(monkeypatch):
    """When the publish-interval gate fails (e.g. window % 10 != 0),
    verify_model must keep its previous weights even though train_step ran."""
    import torch
    import torch.nn as nn

    monkeypatch.setattr("reliquary.validator.service.B_BATCH", 0)
    svc = _make_service()
    svc._publish_every = 10  # default-like cadence
    svc._window_n = 5  # not a multiple of 10 → publish skipped

    train = nn.Linear(4, 4)
    verify = nn.Linear(4, 4)
    verify.load_state_dict(train.state_dict())
    svc.train_model = train
    svc.verify_model = verify
    verify_snapshot = {k: v.detach().clone() for k, v in verify.state_dict().items()}

    def _fake_train_step(model, batch, *, ref_model, window_index=None):
        with torch.no_grad():
            for p in model.parameters():
                p.add_(1.0)
        return model

    import reliquary.validator.service as svc_mod
    monkeypatch.setattr(svc_mod, "train_step", _fake_train_step)

    with _patch_open_grpo_window(svc):
        svc._open_window()  # bumps to 6
    svc._active_batcher.seal_event.set()
    svc._active_batcher.seal_batch = MagicMock(return_value=([MagicMock()] * 100, {}))

    await svc._train_and_publish()

    # verify_model must be unchanged
    for k, before in verify_snapshot.items():
        assert torch.equal(verify.state_dict()[k], before), \
            f"verify_model param '{k}' changed despite publish being skipped"


@pytest.mark.asyncio
async def test_submission_with_matching_hash_accepted_during_open():
    """Inject an 8-submission batch into an OPEN batcher; seal_event fires."""
    svc = _make_service(checkpoint_hash="sha256:cpA")

    # Open a window (hash wired to current_manifest by _open_window)
    with _patch_open_grpo_window(svc):
        svc._open_window()
    batcher = svc._active_batcher
    assert batcher.current_checkpoint_hash == "sha256:cpA"

    # Feed B distinct submissions (k=4 → in zone); unique seed per miner.
    for i in range(B_BATCH):
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{i}",
            prompt_idx=i,
            window_start=batcher.window_start,
            merkle_root="00" * 32,
            rollouts=_rollouts(k=4, seed=i),
            checkpoint_hash="sha256:cpA",
        )
        resp = batcher.accept_submission(req)
        assert resp.accepted, f"unexpected reject for hk{i}: {resp.reason}"

    # seal_event should now be set.
    assert batcher.seal_event.is_set()


@pytest.mark.asyncio
async def test_submission_with_wrong_hash_rejected():
    svc = _make_service(checkpoint_hash="sha256:cpA")
    with _patch_open_grpo_window(svc):
        svc._open_window()
    batcher = svc._active_batcher

    req = BatchSubmissionRequest(
        miner_hotkey="hk", prompt_idx=0,
        window_start=batcher.window_start,
        merkle_root="00" * 32,
        rollouts=_rollouts(k=4),
        checkpoint_hash="sha256:stale",  # mismatched
    )
    resp = batcher.accept_submission(req)
    assert resp.accepted is False
    assert resp.reason.value == "wrong_checkpoint"


@pytest.mark.asyncio
async def test_timeout_partial_seal_skips_train_and_publish():
    """Partial seal (len(batch) < B_BATCH) → state machine advances to READY
    but train_step is NOT called and checkpoint_n stays unchanged.

    This is the skip-if-partial contract: a smaller-than-target batch gives
    a noisy gradient and a different effective LR than a full-batch step,
    so we refuse to step on it. Miners who did submit still earn slots via
    _update_ema — this assertion is about training persistence, not payout.
    """
    svc = _make_service(checkpoint_hash="sha256:cpA")
    with _patch_open_grpo_window(svc):
        svc._open_window()
    batcher = svc._active_batcher
    initial_cn = svc._checkpoint_n

    # Feed only 3 submissions — definitely less than any reasonable B_BATCH
    for i in range(3):
        req = BatchSubmissionRequest(
            miner_hotkey=f"hk{i}", prompt_idx=i,
            window_start=batcher.window_start,
            merkle_root="00" * 32,
            rollouts=_rollouts(k=4, seed=i),
            checkpoint_hash="sha256:cpA",
        )
        batcher.accept_submission(req)

    # seal_event NOT set (fewer than B)
    assert not batcher.seal_event.is_set()

    # Run loop calls _train_and_publish after the timeout fires regardless.
    await svc._train_and_publish()
    assert svc._current_window_state == WindowState.READY
    # New contract: no training → no publish → checkpoint_n unchanged.
    assert svc._checkpoint_n == initial_cn
    svc._checkpoint_store.publish.assert_not_awaited()
