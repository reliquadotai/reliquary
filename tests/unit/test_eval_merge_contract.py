"""Reference jobs and pinned direct episodes keep separate grading authority."""

import pytest

from reliquary.environment import bridged_swe
from reliquary.environment.registry import get_environment_spec
from reliquary.eval import agentic
from reliquary.eval.grading import GradeRequestError, require_gradable
from reliquary.eval.sets import build_episode_set, build_source_set
from tests.unit.test_eval_agentic import TelecomTools
from tests.unit.test_eval_reference_sets import FakeEpisodeEnv


@pytest.mark.parametrize("source,split", [(agentic.TELECOM_ENVIRONMENT, "benchmark"),
                                          ("reliquary-swe", "train")])
def test_reference_builder_stays_unsealed_and_requires_job_grades(tmp_path, monkeypatch,
                                                                source, split):
    monkeypatch.setattr(bridged_swe, "split_length", lambda _: 10)
    monkeypatch.setattr(bridged_swe, "instance_id_of", lambda _, index: f"fixture-{index}")
    monkeypatch.setattr(bridged_swe, "package_version", lambda: "fixture")
    kwargs = dict(split=split, count=1, open_environment=lambda *_: FakeEpisodeEnv())
    card = build_source_set(source, out=tmp_path / "reference", **kwargs)
    assert not {"protocol", "protocol_json", "protocol_sha256"}.intersection(card)
    with pytest.raises(GradeRequestError, match="grade it from its job"):
        require_gradable({card["set_id"]: (card, [])})
    require_gradable({card["set_id"]: (card, [])}, pregraded=True)
    with pytest.raises(ValueError, match="reference-only"):
        build_source_set(source, out=tmp_path / "not-pinned", protocol={}, **kwargs)


def test_pinned_direct_episode_cannot_take_job_pregraded_authority(tmp_path):
    class Environment(TelecomTools):
        def __len__(self):
            return 1

    source = agentic.TelecomBenchmark(Environment(),
        get_environment_spec(agentic.TELECOM_ENVIRONMENT).environment_manifest_sha256)
    card = build_episode_set(agentic.TELECOM_ENVIRONMENT, out=tmp_path / "direct",
        interaction="episode", count=1, seed=7, open_episode=lambda *_: source)
    assert card["protocol"]["harness"]["id"] == "telecom-episode-v1"
    require_gradable({card["set_id"]: (card, [])})
    with pytest.raises(GradeRequestError, match="not an episode set"):
        require_gradable({card["set_id"]: (card, [])}, pregraded=True)


@pytest.mark.parametrize("partial", [{}, {"protocol": {}}, {"protocol_json": "{}"},
                                     {"protocol_sha256": "a" * 64}])
def test_signed_episode_requires_its_complete_frozen_protocol(partial):
    card = {"interaction": "signed_episode", **partial}
    with pytest.raises(GradeRequestError, match="frozen signed episode protocol"):
        require_gradable({"signed": (card, [])})
