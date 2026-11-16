import pytest

from reliquary import env_catalog as cat
from reliquary.environment.registry import ENVIRONMENT_SPECS

pytestmark = pytest.mark.usefixtures("real_rl_prompt_ranges")


def test_every_installed_environment_is_catalogued_or_named_legacy():
    """A new environment needs a decision on what it serves before it ships."""
    catalogued = {name for name, env in cat.CATALOG.items() if env.kind != "episode"}
    assert set(ENVIRONMENT_SPECS) == catalogued | cat.LEGACY_ENVIRONMENTS
    assert not catalogued & cat.LEGACY_ENVIRONMENTS


def test_the_run_environments_all_serve_rl():
    from reliquary.protocol.profiles import PROFILES

    run = PROFILES["teutonic-9b-reliquary-suite-v10-gumbel-dev1"]
    assert all("rl" in cat.CATALOG[name].uses for name in run.environments)


def test_flags_are_the_uses_of_the_rows():
    assert cat.CATALOG["reliquary-swe"].uses == {"rl", "batch", "eval"}
    assert cat.CATALOG["reliquary-terminal"].uses == {"rl", "eval"}
    assert cat.CATALOG["reliquary-deepswe"].uses == {"eval"}
    assert cat.CATALOG["reliquary_telecom_solo_v1"].uses == {"rl"}


@pytest.mark.parametrize("rows", [
    (cat.Rows("rl", "train", 0, 10), cat.Rows("eval", "train", 5, 10)),
    (cat.Rows("batch", "train", 0, 10), cat.Rows("eval", "*")),
    (cat.Rows("rl", "train", 0, 10), cat.Rows("batch", "train", 9, 10)),
    (cat.Rows("rl", "train", 0, 10), cat.Rows("rl", "train", 9, 10)),
    (cat.Rows("batch", "train", 0, 20),),
])
def test_overlaps_and_overruns_are_refused(rows):
    env = cat.CatalogEnv("x", "single_turn", "x", rows, lengths={"train": 15})
    with pytest.raises(ValueError):
        cat.check_catalog({"x": env})


def test_rl_and_batch_may_share_rows_only_under_a_ruling():
    rows = (cat.Rows("rl", "train"), cat.Rows("batch", "train"))
    with pytest.raises(ValueError, match="without a ruling"):
        cat.check_catalog({"x": cat.CatalogEnv("x", "single_turn", "x", rows)})
    cat.check_catalog({"x": cat.CatalogEnv("x", "single_turn", "x", rows, shared="ruled")})


def test_a_batch_job_takes_one_batch_range():
    """general's chat/direct and chat/thinking touch, but one job has one renderer."""
    with pytest.raises(ValueError, match="one of its batch ranges"):
        cat.refuse_batch_rows("reliquary_general_v1", "train", 30_000, 200)
    cat.refuse_batch_rows("reliquary_general_v1", "train", 30_141, 200)


def test_adjacent_ranges_join():
    assert cat.inside("reliquary-swe", "rl", "train", 9_276, 11_248)
    assert not cat.inside("reliquary-swe", "rl", "train", 9_276, 11_249)
    assert cat.inside("reliquary_competitive_code_v1", "batch", "train", 6_899, 8_792)


# --- the lists the repository reads, derived from the catalog ------------------------------------------------


def test_the_rl_prompt_ranges_are_the_catalogs():
    from reliquary.protocol.rl_prompt_ranges import rl_prompt_range

    assert rl_prompt_range("reliquary_code_v1", 2_481_806) == (200_000, 2_381_806)
    assert rl_prompt_range("envscaler_tools_v2", 7_825) == (0, 1_296)
    assert rl_prompt_range("reliquary_telecom_solo_v2", 1_713) == (1_220, 1_713)
    assert rl_prompt_range("reliquary_science_v1", 10_871) == (0, 10_871)


def test_the_generation_order_sources_and_bridged_packages_are_the_catalogs():
    from reliquary.corpus.job import BRIDGED_EVAL_PACKAGES
    from reliquary.eval.qualification import ORDER_ENVIRONMENTS

    assert ORDER_ENVIRONMENTS == {"reliquary_logic_v2", "reliquary_dapo_math_v1",
                                  "reliquary_instruction_following_v1", "reliquary_code_v1"}
    assert BRIDGED_EVAL_PACKAGES == {"reliquary-terminal", "reliquary-deepswe", "reliquary-swebench-pro",
                                     "reliquary-nl2repo", "reliquary-terminal-bench"}


def test_the_held_out_regions_are_the_catalogs_eval_rows():
    from reliquary.eval.sets import HELD_OUT

    assert set(HELD_OUT) == {"math", "code", "logic", "instruction_following"}
    code = HELD_OUT["code"]
    assert (code.source, code.split, code.start, code.count, code.source_length) == (
        "reliquary_code_v1", "train", 2_381_806, 100_000, 2_481_806)
    assert HELD_OUT["math"].split == "eval" and HELD_OUT["math"].count is None


# --- the two gaps the catalog closes -------------------------------------------------------------------------


def test_our_corpus_jobs_on_code_never_take_rl_rows():
    from reliquary.eval.sets import refuse_held_out_overlap

    refuse_held_out_overlap("reliquary_code_v1", 0, 200_000)
    with pytest.raises(ValueError, match="RL share"):
        refuse_held_out_overlap("reliquary_code_v1", 200_000, 10)
    with pytest.raises(ValueError):
        refuse_held_out_overlap("reliquary_code_v1", 199_990, 20)


def test_a_customer_order_names_any_rows_but_eval_rows():
    """Keeping its training data apart is the customer's call: RL rows are open to an order, eval rows never."""
    from reliquary.eval.sets import refuse_order_overlap

    refuse_order_overlap("reliquary_code_v1", 500_000, 1_000)              # RL rows
    refuse_order_overlap("reliquary_code_v1", 199_000, 2_000)              # batch then RL rows
    refuse_order_overlap("reliquary_hard_math_v1", 20_000, 100)
    with pytest.raises(ValueError, match="held out"):
        refuse_order_overlap("reliquary_code_v1", 2_381_000, 1_000)
    with pytest.raises(ValueError, match="RL or batch rows"):
        refuse_order_overlap("reliquary_hard_math_v1", 23_000, 1_000)     # past the split


def test_a_generation_order_takes_rl_rows_but_never_eval_rows():
    from reliquary.eval.qualification import new_generation_request

    record = new_generation_request(qualification_id="q", model="m", revision="r", env="reliquary_code_v1",
                                    prompt_start=500_000, problems=8, sampling={}, max_new_tokens=16,
                                    thinking=False)
    assert record["prompt_start"] == 500_000
    with pytest.raises(ValueError, match="held out"):
        new_generation_request(qualification_id="q", model="m", revision="r", env="reliquary_code_v1",
                               prompt_start=2_400_000, problems=8, sampling={}, max_new_tokens=16,
                               thinking=False)


def _block_episode(package, split):
    return {"env_package": f"{package}==0.3.0+g0123456789abcdef.vfb2", "split": split}


def test_an_rl_episode_env_never_draws_an_eval_split():
    with pytest.raises(ValueError, match="not an RL split"):
        cat.refuse_rl_rows("reliquary-swe", "eval", [], whole_split=True)
    with pytest.raises(ValueError, match="not an RL split"):
        cat.refuse_rl_rows("reliquary-terminal", "tmax_sft", [], whole_split=True)
    with pytest.raises(ValueError, match="not an RL split"):
        cat.refuse_rl_rows("reliquary-deepswe", "test", [], whole_split=True)
    cat.refuse_rl_rows("reliquary-terminal", "tmax_rl", [], whole_split=True)


def test_an_rl_episode_env_draws_only_rl_rows_of_a_shared_split():
    from reliquary.validator.rl_async_episodes import _refuse_outside_rl_rows
    from reliquary.protocol.rl_prompt_ranges import PromptRows

    episode = _block_episode("reliquary-swe", "train")
    with pytest.raises(ValueError, match="declare rows"):
        _refuse_outside_rl_rows("swe_v1", episode, None)
    with pytest.raises(ValueError, match="not RL rows"):
        _refuse_outside_rl_rows("swe_v1", episode, PromptRows(0, 10_000))
    with pytest.raises(ValueError, match="not RL rows"):
        _refuse_outside_rl_rows("swe_v1", episode, PromptRows(10_016, 17_512))
    # The 2026-10-22 launch's rows: the pypika corpus rows excluded.
    _refuse_outside_rl_rows("swe_v1", episode, PromptRows(10_016, 17_512, ((11_248, 11_448), (16_792, 16_793))))


def test_an_rl_episode_env_outside_the_catalog_is_refused():
    with pytest.raises(ValueError, match="not in the environment catalog"):
        cat.refuse_rl_rows("reliquary-unknown", "train", [], whole_split=True)


def test_the_document_carries_flags_and_rows():
    doc = cat.catalog_document()
    assert doc["schema"] == cat.CATALOG_SCHEMA
    by_name = {e["name"]: e for e in doc["environments"]}
    swe = by_name["reliquary-swe"]
    assert swe["uses"] == ["batch", "eval", "rl"] and swe["live_uses"] == ["batch", "eval", "rl"]
    r2e = next(r for r in swe["rows"] if r["split"] == "r2e")
    assert r2e == {"use": "batch", "split": "r2e", "start": 0, "end": None, "live": False,
                   "what": "R2E-Gym subset (10 Python repositories, none in Verified)"}
    assert by_name["reliquary_code_v1"]["orders"] is True
    assert next(r for r in by_name["reliquary_code_v1"]["rows"] if r["use"] == "eval")["eval_set"] == "code"
