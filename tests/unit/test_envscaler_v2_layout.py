"""envscaler_tools_v2's index space: split by world, three ranges, pinned.

The corpus is fetched, not vendored, so most of this reads ``selection.json``
— the pinned choice of scenarios — and the constants restated from it. The
checks that need the corpus itself run when RELIQUARY_ENVSCALER_DATA points
at it and skip otherwise.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import pytest

from reliquary.environment.agentic.envs.envscaler_tools_v2 import (
    corpus as layout,
    environment as v2,
)
from reliquary.environment.registry import get_environment_spec
from reliquary.eval import sets
from reliquary.protocol.environment_catalog import ENVIRONMENT_CATALOG
from reliquary.protocol.profiles import PROFILES
from tests.unit import envscaler_v2_fixtures as fixtures

ENVIRONMENT = "envscaler_tools_v2"
REPOSITORY = Path(__file__).resolve().parents[2]
SELECTION = layout.read_selection()
MANIFEST = json.loads(
    (REPOSITORY / "reliquary/environment/manifests/envscaler_tools_v2.json")
    .read_text(encoding="utf-8")
)


def _counts() -> dict[str, int]:
    return {name: len(SELECTION["ranges"][name]["positions"]) for name in layout.RANGE_ORDER}


# ── split ────────────────────────────────────────────────────────────────────


def test_the_world_split_is_disjoint_and_covers_every_rl_world() -> None:
    split = SELECTION["world_split"]
    rl, sft = set(split["rl"]), set(split["sft"])

    assert not rl & sft
    assert len(rl) == 26 and len(sft) == 25
    assert rl | sft == {f"env_{n}_rl" for n in range(141, 192)}


def test_the_split_rule_is_the_one_written_down() -> None:
    """Sorted by number, even rank to RL — not lexicographic order."""
    rl, sft = layout.split_worlds(["env_10_rl", "env_9_rl", "env_100_rl", "env_11_rl"])

    assert rl == ("env_9_rl", "env_11_rl")
    assert sft == ("env_10_rl", "env_100_rl")
    assert (tuple(SELECTION["world_split"]["rl"]),
            tuple(SELECTION["world_split"]["sft"])) == layout.split_worlds(
        SELECTION["world_split"]["rl"] + SELECTION["world_split"]["sft"]
    )


def test_no_upstream_scenario_is_in_two_ranges() -> None:
    rl = set(SELECTION["ranges"]["rl"]["positions"])
    sft = set(SELECTION["ranges"]["sft_graded"]["positions"])

    assert len(rl) == len(SELECTION["ranges"]["rl"]["positions"])
    assert len(sft) == len(SELECTION["ranges"]["sft_graded"]["positions"])
    assert not rl & sft
    # The corpus range restates sft_graded, in order, minus the excluded worlds.
    corpus = SELECTION["ranges"]["sft_corpus"]["positions"]
    assert set(corpus) < sft
    assert corpus == [p for p in SELECTION["ranges"]["sft_graded"]["positions"] if p in set(corpus)]
    assert SELECTION["ranges"]["sft_corpus"]["excluded_worlds"] == sorted(
        layout.SFT_CORPUS_EXCLUDED_WORLDS
    )
    assert layout.SFT_CORPUS_EXCLUDED_WORLDS < set(SELECTION["world_split"]["sft"])
    # The ungraded range indexes another file entirely.
    assert SELECTION["ranges"]["sft_ungraded"]["file"] == "sft_scen.json"


def test_ranges_interleave_worlds() -> None:
    """A contiguous slice draws from every world, not from one."""
    entries = [("a", 0), ("a", 1), ("a", 2), ("b", 3), ("c", 4), ("b", 5)]

    assert layout.interleave(entries) == [0, 3, 4, 1, 5, 2]


# ── boundaries ───────────────────────────────────────────────────────────────


def test_the_boundaries_are_the_selection() -> None:
    counts = _counts()

    assert v2.RL_RANGE == (0, counts["rl"])
    assert v2.SFT_GRADED_RANGE == (counts["rl"], counts["rl"] + counts["sft_graded"])
    assert v2.SFT_UNGRADED_RANGE == (
        v2.SFT_GRADED_RANGE[1], v2.SFT_GRADED_RANGE[1] + counts["sft_ungraded"]
    )
    assert v2.SFT_CORPUS_RANGE == (v2.SFT_UNGRADED_RANGE[1], v2.LENGTH)
    assert counts["sft_corpus"] == 600
    assert v2.LENGTH == sum(counts.values())
    assert MANIFEST["corpus"]["virtual_length"] == v2.LENGTH
    assert MANIFEST["corpus"]["ranges"] == {
        "rl": list(v2.RL_RANGE),
        "sft_graded": list(v2.SFT_GRADED_RANGE),
        "sft_ungraded": list(v2.SFT_UNGRADED_RANGE),
        "sft_corpus": list(v2.SFT_CORPUS_RANGE),
    }


def test_the_range_guards_restate_the_boundaries() -> None:
    (reserved,) = [r for r in sets.RL_ONLY_RANGES if r.source == ENVIRONMENT]
    servable = sets.CORPUS_SERVABLE_RANGES[ENVIRONMENT]

    assert (reserved.start, reserved.end) == v2.RL_RANGE
    # Only the corpus range: the whole graded SFT range and the ungraded one
    # are held in reserve.
    assert [(r.start, r.end) for r in servable] == [v2.SFT_CORPUS_RANGE]


@pytest.mark.parametrize(("start", "count"), [
    (0, 10),                        # the RL range
    (v2.RL_END - 5, 10),            # straddles RL and SFT
    (v2.SFT_GRADED_END - 5, 10),    # straddles graded and ungraded
    (v2.SFT_UNGRADED_START, 10),    # the reserve
    (v2.SFT_GRADED_START, 10),      # the graded SFT range, excluded worlds included
    (v2.SFT_CORPUS_START - 5, 10),  # straddles ungraded and corpus
])
def test_a_corpus_job_outside_the_corpus_range_is_refused(start, count) -> None:
    with pytest.raises(ValueError):
        sets.refuse_held_out_overlap(ENVIRONMENT, start, count)


def test_a_corpus_job_inside_the_corpus_range_is_accepted() -> None:
    start, end = v2.SFT_CORPUS_RANGE
    sets.refuse_held_out_overlap(ENVIRONMENT, start, end - start)


def test_the_corpus_layout_follows_the_range_order() -> None:
    corpus = fixtures.corpus()

    assert dict(corpus.ranges) == {
        "rl": (0, 2), "sft_graded": (2, 3), "sft_ungraded": (3, 4), "sft_corpus": (4, 4),
    }
    assert [s.range for s in corpus.scenarios] == [
        "rl", "rl", "sft_graded", "sft_ungraded",
    ]
    assert corpus.scenarios[3].checks is None


# ── pins ─────────────────────────────────────────────────────────────────────


def _fetch_module():
    path = REPOSITORY / "scripts" / "fetch_envscaler_corpus.py"
    spec = importlib.util.spec_from_file_location("fetch_envscaler_corpus", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_the_selection_is_bound_to_the_fetched_bytes() -> None:
    fetch = _fetch_module()

    assert dict(layout.CORPUS_FILES) == dict(fetch.NORMALIZED_SHA256)
    assert SELECTION["sources"] == dict(fetch.NORMALIZED_SHA256)
    for name, entry in MANIFEST["corpus"]["files"].items():
        assert entry["normalized_sha256"] == fetch.NORMALIZED_SHA256[name]
        assert entry["upstream_sha256"] == fetch.SOURCES[name]["sha256"]
        assert entry["revision"] == fetch.SOURCES[name]["revision"]


def test_a_selection_over_other_bytes_is_refused(tmp_path) -> None:
    forged = dict(SELECTION, sources={**SELECTION["sources"], "rl_scen.json": "0" * 64})
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(forged))

    with pytest.raises(ValueError, match="other corpus bytes"):
        layout.read_selection(path)


def test_the_corpus_files_are_verified_before_use(tmp_path) -> None:
    fixtures.write_sources(tmp_path)

    with pytest.raises(RuntimeError, match="not the pinned corpus"):
        layout.read_sources(tmp_path)


def test_vendor_mentions_are_caught() -> None:
    assert layout.mentions_vendor("Ask ChatGPT about it")
    assert layout.mentions_vendor("Built by OpenAI")
    assert layout.mentions_vendor("compare with GPT-4")
    assert not layout.mentions_vendor("Grouped partitions table")
    assert layout.scenario_mentions_vendor(
        {"task": "Tidy the bookmarks.", "init_config": {"b": {"title": "OpenAI Blog"}}}
    )


def test_the_vendor_drops_are_recorded() -> None:
    dropped = SELECTION["dropped"]

    assert len(dropped["rl"]["vendor_mention"]) == 4
    assert len(dropped["sft_graded"]["vendor_mention"]) == 0
    assert len(dropped["sft_ungraded"]["vendor_mention"]) == 5


# ── registration ─────────────────────────────────────────────────────────────


def test_the_environment_is_registered_as_a_chatml_binary_episode() -> None:
    spec = get_environment_spec(ENVIRONMENT)

    assert spec.interaction_mode == "episode"
    assert spec.renderer_id == "reliquary-chatml-tools-v1"
    assert spec.reward_lattice_policy == "binary-v1"
    assert spec.attainable_rewards == (0.0, 1.0)
    assert spec.episode_replay_path == get_environment_spec(
        "reliquary_telecom_solo_v1"
    ).episode_replay_path
    assert spec.required_data_env_var == layout.DATA_ENV_VAR
    assert MANIFEST["renderer"] == spec.renderer_id
    assert MANIFEST["reward"] == v2.REWARD_MODE


def test_no_profile_and_no_catalog_entry_names_it() -> None:
    assert not [p for p in PROFILES.values() if ENVIRONMENT in p.environments]
    assert ENVIRONMENT not in ENVIRONMENT_CATALOG


def test_creating_it_without_its_corpus_is_refused(monkeypatch) -> None:
    monkeypatch.delenv(layout.DATA_ENV_VAR, raising=False)

    with pytest.raises(RuntimeError, match=layout.DATA_ENV_VAR):
        get_environment_spec(ENVIRONMENT).create()


# ── the real corpus, when it is here ─────────────────────────────────────────


@pytest.fixture(scope="module")
def real_corpus():
    if not os.environ.get(layout.DATA_ENV_VAR):
        pytest.skip(f"{layout.DATA_ENV_VAR} not set")
    return layout.default_corpus()


def test_every_range_holds_only_its_worlds(real_corpus) -> None:
    split = SELECTION["world_split"]
    worlds = {
        "rl": set(split["rl"]),
        "sft_graded": set(split["sft"]),
        "sft_ungraded": {w for w in real_corpus.worlds if w.endswith("_sft")},
        "sft_corpus": set(split["sft"]) - layout.SFT_CORPUS_EXCLUDED_WORLDS,
    }
    for name in layout.RANGE_ORDER:
        start, end = real_corpus.ranges[name]
        assert (start, end) == {
            "rl": v2.RL_RANGE, "sft_graded": v2.SFT_GRADED_RANGE,
            "sft_ungraded": v2.SFT_UNGRADED_RANGE, "sft_corpus": v2.SFT_CORPUS_RANGE,
        }[name]
        held = {s.env_id for s in real_corpus.scenarios[start:end]}
        assert held == worlds[name], name
    # The corpus range names the scenarios sft_graded names, in its order.
    graded = real_corpus.scenarios[slice(*v2.SFT_GRADED_RANGE)]
    corpus = real_corpus.scenarios[slice(*v2.SFT_CORPUS_RANGE)]
    assert [s.task_id for s in corpus] == [
        s.task_id for s in graded if s.env_id not in layout.SFT_CORPUS_EXCLUDED_WORLDS
    ]
    assert all(s.checks for s in corpus)


def test_the_real_corpus_serves_a_chatml_task(real_corpus) -> None:
    environment = v2.EnvScalerToolsV2Environment(real_corpus)
    task = environment.get_task(v2.SFT_GRADED_START)

    assert task.metadata["range"] == "sft_graded"
    assert task.tools
