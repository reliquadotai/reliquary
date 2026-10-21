"""Each torchrun rank compiles into a Triton cache of its own (distributed.per_rank_triton_cache)."""
from reliquary.trainer.distributed import per_rank_triton_cache


def test_each_rank_gets_its_own_cache_once(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "8")
    monkeypatch.setenv("TRITON_CACHE_DIR", "/c/triton")
    assert per_rank_triton_cache() == "/c/triton-rank3"
    assert per_rank_triton_cache() == "/c/triton-rank3"


def test_a_single_process_keeps_its_cache(monkeypatch):
    monkeypatch.delenv("LOCAL_RANK", raising=False)
    monkeypatch.setenv("TRITON_CACHE_DIR", "/c/triton")
    assert per_rank_triton_cache() is None
    monkeypatch.setenv("LOCAL_RANK", "0")
    monkeypatch.setenv("LOCAL_WORLD_SIZE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    assert per_rank_triton_cache() is None
