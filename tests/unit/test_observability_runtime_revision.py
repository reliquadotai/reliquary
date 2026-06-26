from reliquary.validator.observability import runtime_revision


def test_runtime_revision_prefers_image_baked_revision(monkeypatch):
    runtime_revision.cache_clear()
    monkeypatch.setenv("RELIQUARY_BUILD_REVISION", "a" * 40)
    monkeypatch.setenv("RELIQUARY_IMAGE_REVISION", "b" * 40)

    try:
        assert runtime_revision() == "a" * 40
    finally:
        runtime_revision.cache_clear()


def test_runtime_revision_falls_back_to_deployment_revision(monkeypatch):
    runtime_revision.cache_clear()
    monkeypatch.setenv("RELIQUARY_BUILD_REVISION", "unknown")
    monkeypatch.setenv("RELIQUARY_IMAGE_REVISION", "c" * 40)

    try:
        assert runtime_revision() == "c" * 40
    finally:
        runtime_revision.cache_clear()


def test_runtime_revision_truncates_build_revision(monkeypatch):
    runtime_revision.cache_clear()
    monkeypatch.setenv("RELIQUARY_BUILD_REVISION", "d" * 64)

    try:
        assert runtime_revision() == "d" * 40
    finally:
        runtime_revision.cache_clear()
