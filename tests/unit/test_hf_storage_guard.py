"""Read-only Hugging Face organization storage guard."""

from types import SimpleNamespace

import pytest

from reliquary.trainer.storage_guard import HfStorageGuard, HfStoragePolicy


def test_policy_is_opt_in_and_uses_decimal_terabytes():
    assert HfStoragePolicy.from_env({}) == HfStoragePolicy()
    assert HfStoragePolicy.from_env(
        {"RELIQUARY_HF_STORAGE_FREEZE_TB": "11.0"}
    ).freeze_bytes == 11_000_000_000_000


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "invalid"])
def test_policy_rejects_invalid_ceiling(value):
    with pytest.raises((ValueError, OverflowError)):
        HfStoragePolicy.from_env(
            {"RELIQUARY_HF_STORAGE_FREEZE_TB": value}
        )


class _Api:
    def list_models(self, *, author):
        assert author == "ReliquaryForge"
        return [SimpleNamespace(id="ReliquaryForge/model", used_storage=100)]

    def list_datasets(self, *, author):
        assert author == "ReliquaryForge"
        return [SimpleNamespace(id="ReliquaryForge/data", used_storage=None)]

    def dataset_info(self, repo_id):
        assert repo_id == "ReliquaryForge/data"
        return SimpleNamespace(used_storage=200)

    def list_spaces(self, *, author):
        assert author == "ReliquaryForge"
        return [SimpleNamespace(id="ReliquaryForge/space", used_storage=300)]

    def model_info(self, repo_id):  # pragma: no cover - list value is complete
        raise AssertionError(repo_id)

    def space_info(self, repo_id):  # pragma: no cover - list value is complete
        raise AssertionError(repo_id)


def test_guard_sums_all_visible_hf_storage_and_projects_upload():
    guard = HfStorageGuard(
        policy=HfStoragePolicy(freeze_bytes=1_000),
        api=_Api(),
    )

    assert guard.organization_storage_bytes("ReliquaryForge") == 600
    assert guard.assert_upload_allowed(
        repo_id="ReliquaryForge/model",
        upload_bytes=399,
    ) == 600

    with pytest.raises(RuntimeError, match="active history was not changed"):
        guard.assert_upload_allowed(
            repo_id="ReliquaryForge/model",
            upload_bytes=400,
        )


def test_disabled_guard_does_not_construct_or_query_hf_api():
    guard = HfStorageGuard(policy=HfStoragePolicy())

    assert guard.api is None
    assert guard.assert_upload_allowed(
        repo_id="ReliquaryForge/model",
        upload_bytes=1,
    ) is None
