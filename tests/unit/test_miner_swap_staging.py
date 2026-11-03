"""The swap's shards are written ahead, off the generation path (rl_async.StagedShards)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from reliquary.miner.rl_async import FakeGroupGenerator, StagedShards, VllmGroupGenerator
from reliquary.miner.rl_async_episodes import MixedGenerator
from reliquary.shared import ckpt_delta

from tests.unit.test_rl_async_miner import index, miner_for


@pytest.fixture
def writes(monkeypatch):
    calls = []
    real = ckpt_delta.write_anchor_shards

    def counted(state, folder, **kw):
        calls.append(str(folder))
        return real(state, folder, **kw)

    monkeypatch.setattr(ckpt_delta, "write_anchor_shards", counted)
    return calls


class Llm:
    def __init__(self):
        self.loaded = []

    def collective_rpc(self, fn, args=()):
        self.loaded.append([p for p in args[0]])
        assert all(__import__("os").path.isfile(p) for p in args[0])   # the shards exist while loading

    def reset_prefix_cache(self, reset_running_requests=False):
        return True


def vllm_gen(tmp_path, name="single"):
    gen = VllmGroupGenerator(base_model_dir=str(tmp_path / "base"), work_dir=str(tmp_path / "w"),
                             scratch_dir=str(tmp_path / name))
    gen.tokenizer = object()
    gen._engine = SimpleNamespace(_llm=Llm(), has_running_groups=lambda: True)
    return gen


STATE = {"w": torch.ones(4, dtype=torch.bfloat16)}


def test_a_staged_checkpoint_is_not_written_again_by_the_swap(tmp_path, writes):
    gen = vllm_gen(tmp_path)
    gen.stage_weights(STATE, 5)
    assert len(writes) == 1 and gen.staged.version == 5
    gen.load_weights(STATE, 5)
    assert len(writes) == 1 and len(gen._engine._llm.loaded) == 1
    assert not gen.staged.folder.exists()                          # cleared after the swap


def test_without_staging_or_for_another_version_the_swap_writes_its_shards(tmp_path, writes):
    gen = vllm_gen(tmp_path)
    gen.load_weights(STATE, 5)
    gen.stage_weights(STATE, 6)
    gen.load_weights(STATE, 7)
    assert len(writes) == 3 and not gen.staged.folder.exists()


def test_nothing_is_staged_before_the_first_load_or_for_a_reload(tmp_path, writes):
    gen = vllm_gen(tmp_path)
    gen._engine = None
    gen.stage_weights(STATE, 1)
    reload = vllm_gen(tmp_path, "reload")
    reload.swap = "reload"
    reload.stage_weights(STATE, 1)
    assert writes == []


def test_a_failed_push_clears_the_staged_shards(tmp_path, writes):
    gen = vllm_gen(tmp_path)
    gen._engine._llm.collective_rpc = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("push failed"))
    gen.stage_weights(STATE, 2)
    with pytest.raises(RuntimeError, match="push failed"):
        gen.load_weights(STATE, 2)                                   # groups running: raised, not rebuilt
    assert not gen.staged.folder.exists()


class Loader:
    """``EpisodeWeights``' staging surface without an engine."""

    def __init__(self, folder):
        self.staged, self._owns_staged, self.pushed = StagedShards(folder), True, []

    def stage(self, state, version):
        self.staged.stage(state, version)

    def share_staged(self, staged):
        self.staged, self._owns_staged = staged, False

    def __call__(self, state, version):
        self.pushed.append(self.staged.paths(state, version))
        if self._owns_staged:
            self.staged.clear()


def test_a_mixed_miner_writes_one_copy_for_both_engines(tmp_path, writes):
    from reliquary.miner.rl_async_episodes import EpisodeLeaseGenerator

    single = vllm_gen(tmp_path)
    loader = Loader(tmp_path / "episodes")
    episodes = EpisodeLeaseGenerator.__new__(EpisodeLeaseGenerator)
    episodes._load = loader
    mixed = MixedGenerator(episodes=episodes, single=single)
    assert single.staged is loader.staged
    mixed.stage_weights(STATE, 9)
    mixed.load_weights(STATE, 9)
    assert len(writes) == 1
    assert single._engine._llm.loaded == loader.pushed               # the same shards for both
    assert not single.staged.folder.exists()


class StagingGenerator(FakeGroupGenerator):
    def __init__(self):
        super().__init__()
        self.staged = []

    def stage_weights(self, state, version):
        self.staged.append(int(version))


class Sync:
    def sync(self, response, state, version):
        latest = response["index"]["latest"]["version"]
        return {"v": latest}, latest, SimpleNamespace(anchor=None, deltas=[object()], bytes=1)


async def test_the_follower_stages_before_offering_the_weights():
    miner, transport, generator, _ = miner_for(sync=Sync(), generator=StagingGenerator())
    transport.checkpoints = [index(1)]
    assert await miner.follow_checkpoint_once() == 1
    assert generator.staged == [1] and miner._ready.is_set() and generator.loads == [0]


async def test_a_failed_staging_still_offers_the_weights():
    gen = StagingGenerator()
    gen.stage_weights = lambda *a: (_ for _ in ()).throw(OSError("no space"))
    miner, transport, _, _ = miner_for(sync=Sync(), generator=gen)
    transport.checkpoints = [index(1)]
    assert await miner.follow_checkpoint_once() == 1 and miner._ready.is_set()
