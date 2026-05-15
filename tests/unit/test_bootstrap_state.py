"""ValidationService._bootstrap_state_from_external: derives window_n + checkpoint_n + EMA from R2 + HF."""

from collections import defaultdict
from dataclasses import dataclass
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@dataclass
class _FakeEnv:
    @property
    def name(self): return "fake"
    def __len__(self): return 100
    def get_problem(self, i): return {"prompt": "p", "ground_truth": "", "id": f"p{i}"}
    def compute_reward(self, p, c): return 1.0


class _FakeWallet:
    class _Hk:
        ss58_address = "5FHk"
        @staticmethod
        def sign(d): return b"sig"
    hotkey = _Hk()


def _make_service():
    from reliquary.validator.service import ValidationService
    return ValidationService(
        wallet=_FakeWallet(), model=MagicMock(), tokenizer=MagicMock(),
        env=_FakeEnv(), netuid=99,
    )


@pytest.mark.asyncio
async def test_bootstrap_sets_window_n_from_r2():
    """window_n is set to max R2 window key."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[1, 5, 10, 42]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=[]),
    ):
        await svc._bootstrap_state_from_external()

    assert svc._window_n == 42


@pytest.mark.asyncio
async def test_bootstrap_window_n_zero_when_no_archives():
    """No archives → window_n stays 0."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=[]),
    ):
        await svc._bootstrap_state_from_external()

    assert svc._window_n == 0


def _ckpt_commit(n: int, sha: str | None = None) -> MagicMock:
    """A MagicMock that mimics a HuggingFace commit with title 'checkpoint N'
    and a real 40-hex SHA. The auto-resume path in _bootstrap_state_from_external
    parses sha from ``c.commit_id`` and passes it to parse_resume_source, which
    enforces the 40-hex pattern, so we can't fall back to MagicMock defaults."""
    c = MagicMock()
    c.title = f"checkpoint {n}"
    c.commit_id = sha if sha is not None else f"{n:040x}"
    return c


@pytest.mark.asyncio
async def test_bootstrap_checkpoint_n_from_hf_commits():
    """checkpoint_n is set to the HIGHEST N seen in a 'checkpoint N' commit
    title, not the raw commit count. Mixed titles (e.g. an initial 'seed'
    commit) are skipped; non-matching commits don't shift the counter."""
    svc = _make_service()

    fake_commits = [
        _ckpt_commit(3),
        _ckpt_commit(2),
        _ckpt_commit(1),
        MagicMock(title="seed from Qwen/Qwen3-4B"),  # not a checkpoint commit
    ]

    # Mock _apply_resume_from so we can isolate the counting / discovery logic
    # from the model-loading machinery (which would need real HF downloads).
    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=fake_commits),
        patch.object(svc, "_apply_resume_from", new=AsyncMock(
            side_effect=lambda: setattr(svc, "_checkpoint_n", 3),
        )),
    ):
        await svc._bootstrap_state_from_external()

    assert svc._checkpoint_n == 3


@pytest.mark.asyncio
async def test_bootstrap_tolerates_r2_failure():
    """R2 failure during window key fetch → window_n stays 0, no exception raised."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(side_effect=RuntimeError("R2 down")),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=[]),
    ):
        await svc._bootstrap_state_from_external()  # must not raise

    assert svc._window_n == 0


@pytest.mark.asyncio
async def test_bootstrap_tolerates_hf_failure():
    """HF failure → checkpoint_n stays 0, no exception raised."""
    svc = _make_service()

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[1, 2]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "huggingface_hub.HfApi.list_repo_commits",
            side_effect=Exception("HF auth failed"),
        ),
    ):
        await svc._bootstrap_state_from_external()  # must not raise

    assert svc._checkpoint_n == 0


# --- PR #24: auto-resume to latest HF checkpoint ---------------------------


@pytest.mark.asyncio
async def test_bootstrap_auto_resumes_to_latest_hf_when_env_unset():
    """No RELIQUARY_RESUME_FROM env var: validator must auto-resume from the
    highest-numbered 'checkpoint N' commit on HF. This is the path that
    prevents the ckpt 45 → ckpt 26 regression we saw on the PR #23 redeploy,
    where a stale env var pinned the validator to an old commit even though
    HF had 19 newer ones."""
    svc = _make_service()
    assert not svc._resume_from   # no env

    fake_commits = [
        _ckpt_commit(45, sha="cfbb23f031b5553b93dc4bd57614dc4006825555"),
        _ckpt_commit(44, sha="44" * 20),
        _ckpt_commit(26, sha="d5af5fec9892385a85acc21aee0a5e91f4b47888"),
    ]
    apply_calls: list[str] = []

    async def _fake_apply_resume_from():
        # Mimic the real _apply_resume_from side effects we care about for
        # the bootstrap test: record the SHA it was called with, set
        # _checkpoint_n, install a fake manifest.
        from reliquary.validator.checkpoint import ManifestEntry
        apply_calls.append(svc._resume_from)
        svc._checkpoint_n = 45
        svc._checkpoint_store._current = ManifestEntry(
            checkpoint_n=45,
            repo_id=svc._checkpoint_store.repo_id,
            revision="cfbb23f031b5553b93dc4bd57614dc4006825555",
            signature="ed25519:00",
        )

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=fake_commits),
        patch.object(svc, "_apply_resume_from", new=_fake_apply_resume_from),
    ):
        await svc._bootstrap_state_from_external()

    assert apply_calls == ["sha:cfbb23f031b5553b93dc4bd57614dc4006825555"]
    assert svc._checkpoint_n == 45


@pytest.mark.asyncio
async def test_bootstrap_overrides_stale_env_with_newer_hf():
    """env-set RELIQUARY_RESUME_FROM pointing at ckpt 26 but HF has ckpt 45:
    validator must override and resume from HF latest. This is the exact
    scenario that caused the production regression — the deploy artifact
    had ckpt 26 baked in while HF had progressed to ckpt 45."""
    svc = _make_service()

    # Simulate: _apply_resume_from already ran (from env) and installed ckpt 26.
    from reliquary.validator.checkpoint import ManifestEntry
    svc._checkpoint_n = 26
    svc._checkpoint_store._current = ManifestEntry(
        checkpoint_n=26,
        repo_id=svc._checkpoint_store.repo_id,
        revision="d5af5fec9892385a85acc21aee0a5e91f4b47888",
        signature="ed25519:00",
    )

    fake_commits = [
        _ckpt_commit(45, sha="cfbb23f031b5553b93dc4bd57614dc4006825555"),
        _ckpt_commit(26, sha="d5af5fec9892385a85acc21aee0a5e91f4b47888"),
    ]
    apply_calls: list[str] = []

    async def _fake_apply_resume_from():
        apply_calls.append(svc._resume_from)
        svc._checkpoint_n = 45
        svc._checkpoint_store._current = ManifestEntry(
            checkpoint_n=45,
            repo_id=svc._checkpoint_store.repo_id,
            revision="cfbb23f031b5553b93dc4bd57614dc4006825555",
            signature="ed25519:00",
        )

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=fake_commits),
        patch.object(svc, "_apply_resume_from", new=_fake_apply_resume_from),
    ):
        await svc._bootstrap_state_from_external()

    assert apply_calls == ["sha:cfbb23f031b5553b93dc4bd57614dc4006825555"]
    assert svc._checkpoint_n == 45


@pytest.mark.asyncio
async def test_bootstrap_trusts_operator_pin_when_env_is_newer_than_hf():
    """env-set RELIQUARY_RESUME_FROM at ckpt 47 (newer than HF latest 45):
    leave the operator's pin alone. Lets a release-candidate validator be
    tested against a known commit without auto-discovery rewriting it."""
    svc = _make_service()

    from reliquary.validator.checkpoint import ManifestEntry
    svc._checkpoint_n = 47   # operator pinned to something newer than HF
    svc._checkpoint_store._current = ManifestEntry(
        checkpoint_n=47,
        repo_id=svc._checkpoint_store.repo_id,
        revision="aa" * 20,
        signature="ed25519:00",
    )

    fake_commits = [
        _ckpt_commit(45, sha="cfbb23f031b5553b93dc4bd57614dc4006825555"),
        _ckpt_commit(44),
    ]
    apply_calls: list[str] = []

    async def _fake_apply_resume_from():
        apply_calls.append(svc._resume_from)

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=fake_commits),
        patch.object(svc, "_apply_resume_from", new=_fake_apply_resume_from),
    ):
        await svc._bootstrap_state_from_external()

    assert apply_calls == []   # operator pin preserved, no override fired
    assert svc._checkpoint_n == 47


@pytest.mark.asyncio
async def test_bootstrap_no_op_when_hf_has_no_checkpoint_commits():
    """Fresh HF repo (only the initial 'seed from base model' commit):
    don't try to auto-resume from anything. The validator stays on
    whatever the constructor / env gave it."""
    svc = _make_service()

    fake_commits = [
        MagicMock(title="seed from Qwen/Qwen3-4B"),
        MagicMock(title="initial commit"),
    ]
    apply_calls: list[str] = []

    async def _fake_apply_resume_from():
        apply_calls.append(svc._resume_from)

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=fake_commits),
        patch.object(svc, "_apply_resume_from", new=_fake_apply_resume_from),
    ):
        await svc._bootstrap_state_from_external()

    assert apply_calls == []
    assert svc._checkpoint_n == 0


@pytest.mark.asyncio
async def test_bootstrap_picks_highest_n_not_latest_in_log():
    """Commits are typically reverse-chronological from HF, but the contract
    is "highest checkpoint N", not "first commit in the list". Defensive
    test: ensure an out-of-order log (e.g. force-push, rebase) still selects
    the right SHA."""
    svc = _make_service()

    fake_commits = [
        _ckpt_commit(40, sha="40" * 20),
        _ckpt_commit(45, sha="45" * 20),   # highest, not first
        _ckpt_commit(42, sha="42" * 20),
        _ckpt_commit(38, sha="38" * 20),
    ]
    captured_sha: list[str] = []

    async def _fake_apply_resume_from():
        captured_sha.append(svc._resume_from)
        svc._checkpoint_n = 45

    with (
        patch(
            "reliquary.infrastructure.storage.list_all_window_keys",
            new=AsyncMock(return_value=[]),
        ),
        patch(
            "reliquary.infrastructure.storage.list_recent_datasets",
            new=AsyncMock(return_value=[]),
        ),
        patch("huggingface_hub.HfApi.list_repo_commits", return_value=fake_commits),
        patch.object(svc, "_apply_resume_from", new=_fake_apply_resume_from),
    ):
        await svc._bootstrap_state_from_external()

    assert captured_sha == ["sha:" + "45" * 20]
    assert svc._checkpoint_n == 45
