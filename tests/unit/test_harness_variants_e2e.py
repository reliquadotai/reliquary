"""Surface variants end to end: the miner's own loop plays a variant against reliquary-sandbox's
real gateway (FakeBoxes), the gateway only ever sees canonical tools, and the validator's signed
parse admits the episode under that variant's renderer and refuses it under any other."""

from __future__ import annotations

import asyncio
import dataclasses
from types import SimpleNamespace

import pytest

pytest.importorskip("renderers")
pytest.importorskip("reliquary_sandbox_service.episodes.testing")

from reliquary_harness.spec import BASH_SURFACE_V1  # noqa: E402
from reliquary_harness.variant import variant_count  # noqa: E402
from reliquary_sandbox_service.episodes import testing  # noqa: E402
from reliquary_sandbox_service.episodes.local_gateway import serve_app  # noqa: E402

from reliquary.corpus.signed_parse import parse_signed_trajectory, signed_records  # noqa: E402
from reliquary.corpus.trajectory_parse import TrajectoryRefused  # noqa: E402
from reliquary.environment.agentic_swe import load_turn_renderer  # noqa: E402
from reliquary.miner.harness_episode import HarnessEpisodeRunner  # noqa: E402
from tests.unit import harness_episode_parity as parity  # noqa: E402
from tests.unit.test_harness_goldens import TOKENIZERS  # noqa: E402

pytestmark = pytest.mark.skipif(not TOKENIZERS, reason="set RELIQUARY_QWEN38_TOKENIZER")
EXIT_CODES = dataclasses.replace(BASH_SURFACE_V1, version="test-exit-codes",
                                 variants={"axes": [{"name": "observation", "values": ["plain", "exit-code"]}]})


def _directory() -> str:
    return next(iter(TOKENIZERS.values()))


def _script(spec, variant):
    names = spec.surface_names(_renderer(spec, variant).variant.values)
    return [
        parity._turn("Write it and look.", parity._call(names["bash"], command="write /work/answer.txt 42"),
                     parity._call(names["bash"], command="cat /missing")),
        parity._turn("Fix the text.", parity._call(names["edit"], path="/work/answer.txt", old_str="42", new_str="42")),
        parity._turn("Done.", content="Written."),
    ]


def _renderer(spec, variant):
    return load_turn_renderer(_directory(), ("bash", "edit"), spec=spec, variant=variant)


async def _play(tmp_path, spec, variant):
    tokenizer = parity._tokenizer(_directory())
    renderer = _renderer(spec, variant)
    engine = parity.FakeEngine(tokenizer, _script(spec, variant), renderer.stop_ids, 1 << 20)
    with testing.fake_gateway(tmp_path) as (gateway, _), serve_app(engine.app) as engine_url:
        token = gateway.issue(env="fake-env", split="train", index=1, image=testing.FAKE_IMAGE,
                              budgets=parity._budgets())
        async with HarnessEpisodeRunner(gateway_url=gateway.url, model=parity.MODEL, renderer_dir=_directory(),
                                        generate_url=engine_url,
                                        sampling=SimpleNamespace(temperature=1.0, top_p=1.0),
                                        max_tokens=256, max_turns=6, spec=spec, variant=variant) as runner:
            result = await runner.run(token=token, prompt=parity.PROMPT)
    return result, engine


def _trajectory(engine):
    prompt = engine.requests[0]["token_ids"]
    full = engine.requests[-1]["token_ids"] + engine.completions[-1]
    spans = []
    for request, completion in zip(engine.requests, engine.completions):
        start = len(request["token_ids"]) - len(prompt)
        spans.append((start, start + len(completion)))
    return prompt, full[len(prompt):], spans


def _parse(renderer, result, engine):
    prompt, tokens, spans = _trajectory(engine)
    found = signed_records(result.transcript)
    return parse_signed_trajectory(renderer, prompt_ids=prompt, tokens=tokens, spans=spans,
                                   stop=result.trace.stop_condition, max_turns=6, calls=found.calls,
                                   offered=found.tools, final=found.final)


@pytest.mark.parametrize("variant", range(variant_count(BASH_SURFACE_V1)))
def test_a_variant_played_by_the_miner_is_admitted_under_that_variant_only(tmp_path, variant):
    result, engine = asyncio.run(_play(tmp_path, BASH_SURFACE_V1, variant))
    assert result.error is None and result.trace.stop_condition == "agent_completed"
    assert result.final["body"]["status"] == "graded"
    # The gateway ran canonical tools only, whatever the model was shown.
    calls = [r["body"] for r in result.transcript["records"] if r["body"]["kind"] == "call"]
    assert [c["tool"] for c in calls] == ["bash", "bash", "edit"]
    # The prompt the miner played is the validator's render of the variant.
    renderer = _renderer(BASH_SURFACE_V1, variant)
    assert engine.requests[0]["token_ids"] == renderer.initial_ids(parity.PROMPT)
    parsed = _parse(renderer, result, engine)
    assert [a.tool for a in parsed.actions] == ["bash", "bash", "edit"]
    for other in range(variant_count(BASH_SURFACE_V1)):
        if other == variant:
            continue
        other_renderer = _renderer(BASH_SURFACE_V1, other)
        assert other_renderer.initial_ids(parity.PROMPT) != engine.requests[0]["token_ids"]
        try:
            _parse(other_renderer, result, engine)
        except TrajectoryRefused:
            continue
        # Same tool names, other prompt: the prompt fidelity check (the validator's own render) refuses it.
        assert BASH_SURFACE_V1.surface_names(other_renderer.variant.values) == \
            BASH_SURFACE_V1.surface_names(renderer.variant.values)


def test_an_exit_code_variant_shows_the_signed_code_and_is_read_back_only_as_such(tmp_path):
    result, engine = asyncio.run(_play(tmp_path, EXIT_CODES, 1))
    assert result.error is None and result.trace.stop_condition == "agent_completed"
    tokenizer = parity._tokenizer(_directory())
    second = tokenizer.decode(engine.requests[1]["token_ids"][len(engine.requests[0]["token_ids"]):],
                              skip_special_tokens=False)
    assert "[exit code: 0]\n(no output)" in second and "[exit code: 1]\n" in second
    _parse(_renderer(EXIT_CODES, 1), result, engine)
    with pytest.raises(TrajectoryRefused) as refused:
        _parse(_renderer(EXIT_CODES, 0), result, engine)
    assert refused.value.reason == "bad_observation"
