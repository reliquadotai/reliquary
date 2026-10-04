"""A job on a Verifiers set declares the external eval environment in its contract."""

from __future__ import annotations

import asyncio

import pytest

from reliquary.eval import prompt_source as ps
from reliquary.eval.sets import build_source_set
from reliquary.eval.storage import SubnetEvalStore, publish_set
from reliquary.infrastructure import corpus_job_store as job_store
from reliquary.protocol.composition import RUN_POLICIES, ModelSpec, compose_profile
from reliquary.protocol.external_eval import (
    EXTERNAL_EVAL_BODY,
    EXTERNAL_EVAL_ENVIRONMENT,
    contract_environment_for,
)
from tests.unit.test_admin_eval_jobs import (  # noqa: F401
    MODEL, REVISION, SAMPLING, THRESHOLDS, _qualify, admin,
)
from tests.unit.test_eval_verifiers_source import FakeTask, fake_handle
from tests.unit.test_eval_verifiers_source import opener as taskset_opener
from tests.unit.test_jobs_cli import registry  # noqa: F401


def model():
    return ModelSpec("m/M", "r" * 40, "Qwen3ForCausalLM", "chat_template")


def test_the_external_environment_composes():
    profile = compose_profile(profile_id="order-eval-x", model=model(),
                              run=RUN_POLICIES["corpus-v1"],
                              environments=[EXTERNAL_EVAL_ENVIRONMENT], external_eval=True)
    assert profile.environments == {EXTERNAL_EVAL_ENVIRONMENT: EXTERNAL_EVAL_BODY}
    body = profile.to_generation_contract()["environments"][EXTERNAL_EVAL_ENVIRONMENT]
    assert body["environment_contract_id"] == "reliquary/external-eval/v1"


def test_it_takes_no_overrides():
    with pytest.raises(ValueError, match="takes no overrides"):
        compose_profile(profile_id="x", model=model(), run=RUN_POLICIES["corpus-v1"],
                        environments=[EXTERNAL_EVAL_ENVIRONMENT], external_eval=True,
                        overrides={EXTERNAL_EVAL_ENVIRONMENT: {"max_new_tokens": 9}})


def test_only_an_eval_set_job_may_declare_it():
    """An RL task or a corpus job naming it is refused at declaration."""
    with pytest.raises(ValueError, match="only by a job reading an eval set"):
        compose_profile(profile_id="x", model=model(), run=RUN_POLICIES["corpus-v1"],
                        environments=[EXTERNAL_EVAL_ENVIRONMENT])


def test_an_unknown_environment_is_still_refused():
    with pytest.raises(ValueError, match="no catalog entry"):
        compose_profile(profile_id="x", model=model(), run=RUN_POLICIES["corpus-v1"],
                        environments=["reliquary_external_eval_v2"])


def test_which_environment_a_set_declares():
    assert contract_environment_for({"source": "reliquary_logic_v2"}) == "reliquary_logic_v2"
    assert contract_environment_for({"source": "reliquary_logic_v2",
                                     "source_kind": "catalog"}) == "reliquary_logic_v2"
    assert contract_environment_for({"source": "verifiers:aime26",
                                     "source_kind": "verifiers"}) == EXTERNAL_EVAL_ENVIRONMENT


def _publish_verifiers_set(root):
    from reliquary.corpus.delivery import LocalDirectorySink

    card = build_source_set("verifiers:fake", out=root / "vset", clock=lambda: 1.0,
                            open_taskset=taskset_opener(fake_handle(
                                [FakeTask(i, system="be brief" if i == 0 else None)
                                 for i in range(6)])))
    asyncio.run(publish_set(root / "vset", platform=LocalDirectorySink(root / "p"),
                            subnet=SubnetEvalStore()))
    return card


def test_an_eval_job_on_a_verifiers_set(admin):  # noqa: F811
    from reliquary.validator.eval_control import order_job_refusal

    card = _publish_verifiers_set(admin.root)
    qualification = {"qualification_id": "order-q1", "model": MODEL, "revision": REVISION,
                     "set_id": card["set_id"], "problems": 4, "completions": 8,
                     "sampling": SAMPLING, "max_new_tokens": 512, "thinking": True}
    assert admin("POST", "/admin/v1/qualifications", qualification).status_code == 201
    _qualify(admin)
    created = admin("POST", "/admin/v1/jobs", {
        "job_id": "order-eval-v", "model": MODEL, "env": "verifiers:fake", "prompt_count": 4,
        "samples_per_prompt": 2, "max_new_tokens": 512, "thinking": True,
        "sampling": SAMPLING, "eval_set_id": card["set_id"], "qualification_id": "order-q1"})
    assert created.status_code == 201, created.text
    entry = admin.registry["entries"]["order-eval-v"]
    assert list(entry.contract["environments"]) == [EXTERNAL_EVAL_ENVIRONMENT]
    job, _ = asyncio.run(job_store.read_job("order-eval-v"))
    assert ps.parse_eval_source(job.prompt_source).set_id == card["set_id"]
    assert job.renderer_id == "chat-template-thinking-v1"
    assert order_job_refusal(entry, job) is None
