"""An audit executor packs smaller batches on a card the model leaves little room on (a 27B on an 80 GB H100)."""

from types import SimpleNamespace

from reliquary.validator import corpus_audit_executor as ex

GB = 10**9


def test_a_card_with_little_room_left_packs_smaller_batches():
    assert ex.batch_tokens_for(26 * GB, 131072) == 65536     # 27B on an H100
    assert ex.batch_tokens_for(60 * GB, 131072) == 131072    # 9B on an H100
    assert ex.batch_tokens_for(87 * GB, 131072) == 131072    # 27B on an H200
    assert ex.batch_tokens_for(26 * GB, 32768) == 32768      # never larger than asked
    assert ex.batch_tokens_for(None, 131072) == 131072       # unknown: as before


class _Http:
    async def post(self, path, json=None, **kw):
        return SimpleNamespace(status_code=200, raise_for_status=lambda: None,
                               json=lambda: {"model_id": "m", "model_revision": "r"})


def _executor(**kw):
    return ex.AuditExecutor(http=_Http(), executor_id="pod-1", token="x", model_id="m", model_revision="r",
                            load_model=lambda m, r: "model", **kw)


async def test_the_batch_is_sized_to_the_card_once_the_model_is_loaded(monkeypatch):
    monkeypatch.delenv(ex.BATCH_TOKENS_ENV, raising=False)
    executor = _executor(free_bytes=lambda: 26 * GB)
    await executor.start()
    assert executor._batch_tokens == 65536
    roomy = _executor(free_bytes=lambda: 87 * GB)
    await roomy.start()
    assert roomy._batch_tokens == 131072


async def test_a_batch_set_by_the_caller_or_the_operator_is_kept(monkeypatch):
    monkeypatch.delenv(ex.BATCH_TOKENS_ENV, raising=False)
    given = _executor(batch_tokens=131072, free_bytes=lambda: 26 * GB)
    await given.start()
    assert given._batch_tokens == 131072
    monkeypatch.setenv(ex.BATCH_TOKENS_ENV, "131072")
    operator = _executor(free_bytes=lambda: 26 * GB)
    await operator.start()
    assert operator._batch_tokens == 131072


def test_the_executor_grows_its_cuda_segments_unless_the_operator_chose(monkeypatch):
    monkeypatch.setattr(ex, "serve_executor", lambda url, build: None)
    monkeypatch.delenv(ex.ALLOC_CONF_ENV, raising=False)
    ex.run_audit_executor(control_url="http://c", executor_id="pod-1")
    assert __import__("os").environ[ex.ALLOC_CONF_ENV] == "expandable_segments:True"
    monkeypatch.setenv(ex.ALLOC_CONF_ENV, "max_split_size_mb:512")
    ex.run_audit_executor(control_url="http://c", executor_id="pod-1")
    assert __import__("os").environ[ex.ALLOC_CONF_ENV] == "max_split_size_mb:512"
