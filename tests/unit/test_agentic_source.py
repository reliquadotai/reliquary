"""The agentic prompt source: resolved only with `episode`, rows from SWE-smith."""

import json

import pytest

from reliquary.corpus.job import parse_job
from reliquary.environment import agentic_swe
from reliquary.environment.agentic_swe import SweSource
from reliquary.protocol.agentic_source import AGENTIC_SWE_ENVIRONMENT
from reliquary.validator import corpus_service
from reliquary.validator.corpus_service import (
    AgenticPromptJob,
    CorpusPromptSourceError,
    EpisodePromptRenderer,
    prompt_job_for_spec,
    renderer_for_job,
    resolve_prompt_source,
)
from tests.unit.test_corpus_job_episode import COMMIT, _manifest

ROWS = [("repo__a.1", "fix a"), ("repo__b.2", "fix b"), ("repo__c.3", "fix c")]


@pytest.fixture
def supported(monkeypatch):
    monkeypatch.setattr(agentic_swe, "episode_support_refusal", lambda episode, need_verifiers: None)
    monkeypatch.setattr(agentic_swe, "load_swe_source", lambda num_images: SweSource(ROWS))


def _job(**overrides):
    return parse_job(_manifest(**{"prompt_count": 3, **overrides}))


def test_the_agentic_source_needs_an_episode(supported):
    with pytest.raises(CorpusPromptSourceError, match="episode"):
        resolve_prompt_source(AGENTIC_SWE_ENVIRONMENT, renderer_id="renderers:qwen38@0.1.11")


def test_an_episode_needs_the_agentic_source(supported):
    with pytest.raises(CorpusPromptSourceError, match=AGENTIC_SWE_ENVIRONMENT):
        resolve_prompt_source("openmathinstruct", renderer_id="x", episode=_job().episode)


def test_an_unsupported_pin_is_refused(monkeypatch):
    monkeypatch.setattr(agentic_swe, "episode_support_refusal",
                        lambda episode, need_verifiers: "env commit differs")
    with pytest.raises(CorpusPromptSourceError, match="env commit differs"):
        resolve_prompt_source(AGENTIC_SWE_ENVIRONMENT, renderer_id="renderers:qwen38@0.1.11",
                              episode=_job().episode)


def test_rows_come_from_the_source_and_stay_in_bounds(supported):
    prompts = prompt_job_for_spec(_job())
    assert isinstance(prompts, AgenticPromptJob)
    task = prompts.task_for(1)
    assert (task.id, task.prompt) == ("repo__b.2", "fix b")
    with pytest.raises(CorpusPromptSourceError):
        prompts.task_for(3)


def test_a_job_claiming_more_rows_than_the_source_is_refused(supported):
    with pytest.raises(CorpusPromptSourceError, match="has 3"):
        prompt_job_for_spec(_job(prompt_count=4))


def test_the_generic_renderer_of_an_episode_job_refuses_to_render(supported):
    renderer = renderer_for_job(_job(), encode=lambda text: [])
    assert isinstance(renderer, EpisodePromptRenderer)
    with pytest.raises(CorpusPromptSourceError):
        renderer.initial_text(None)


def test_support_refusal_names_each_pin(monkeypatch):
    episode = _job().episode
    monkeypatch.setattr(agentic_swe, "installed_env_commit", lambda: COMMIT)
    monkeypatch.setattr(agentic_swe, "installed_verifiers_commit", lambda: agentic_swe.SUPPORTED_VERIFIERS)
    monkeypatch.setattr(agentic_swe, "_renderers_version", lambda: agentic_swe.RENDERERS_VERSION)
    assert agentic_swe.episode_support_refusal(episode, need_verifiers=True) is None
    monkeypatch.setattr(agentic_swe, "installed_env_commit", lambda: "f" * 40)
    assert "reliquary-swe" in agentic_swe.episode_support_refusal(episode, need_verifiers=False)
    monkeypatch.setattr(agentic_swe, "installed_env_commit", lambda: COMMIT)
    monkeypatch.setattr(agentic_swe, "installed_verifiers_commit", lambda: None)
    assert "verifiers" in agentic_swe.episode_support_refusal(episode, need_verifiers=True)
    assert agentic_swe.episode_support_refusal(episode, need_verifiers=False) is None


def test_an_editable_checkout_reports_its_git_commit(monkeypatch, tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    head = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "HEAD"], capture_output=True,
                          text=True, check=True).stdout.strip()

    class _Dist:
        def read_text(self, name):
            return json.dumps({"url": f"file://{tmp_path}", "dir_info": {"editable": True}})

    monkeypatch.setattr(agentic_swe.importlib.metadata, "distribution", lambda name: _Dist())
    assert agentic_swe.installed_env_commit() == head


def test_a_vcs_install_reports_its_pinned_commit(monkeypatch):
    class _Dist:
        def read_text(self, name):
            return json.dumps({"url": "https://github.com/x/y", "vcs_info": {"commit_id": "9" * 40}})

    monkeypatch.setattr(agentic_swe.importlib.metadata, "distribution", lambda name: _Dist())
    assert agentic_swe.installed_verifiers_commit() == "9" * 40


def test_the_contract_composes_with_the_agentic_body_only_when_asked():
    from reliquary.protocol.composition import RUN_POLICIES, ModelSpec, compose_profile
    from reliquary.protocol.profiles import profile_from_contract

    model = ModelSpec("Qwen/Qwen3.8-27B", "rev1", "Qwen3_5ForConditionalGeneration", "raw")
    with pytest.raises(ValueError, match="agentic"):
        compose_profile(profile_id="t", model=model, run=RUN_POLICIES["corpus-v1"],
                        environments=[AGENTIC_SWE_ENVIRONMENT])
    profile = compose_profile(profile_id="t", model=model, run=RUN_POLICIES["corpus-v1"],
                              environments=[AGENTIC_SWE_ENVIRONMENT], agentic=True)
    rebuilt = profile_from_contract(profile.to_generation_contract())
    assert AGENTIC_SWE_ENVIRONMENT in rebuilt.environments


def test_prepare_corpus_job_writes_the_episode_and_the_agentic_contract(supported):
    from reliquary.cli.main import prepare_corpus_job
    from tests.unit.test_corpus_job_episode import _episode

    manifest, entry = prepare_corpus_job(
        job_id="swe-agentic-v1", task_id=None, model="Qwen/Qwen3.8-27B", model_revision="rev1",
        model_architecture="Qwen3_5ForConditionalGeneration", checkpoint_sha256="c" * 64,
        from_profile=None, prompt_encoding=None, prompt_source=AGENTIC_SWE_ENVIRONMENT,
        prompt_count=3, prompt_start=0, renderer_id="renderers:qwen38@0.1.11",
        eos_token_id=248046, slots_per_prompt=2, max_new_tokens=8192, cap=0.05,
        min_incentive_share=0.0, audit_params={"audit_q": 1.0}, prompt_order="free",
        episode=_episode(),
    )
    assert manifest["episode"] == _episode()
    assert list(entry.contract["environments"]) == [AGENTIC_SWE_ENVIRONMENT]


def _prepared_entry():
    from reliquary.cli.main import prepare_corpus_job
    from tests.unit.test_corpus_job_episode import _episode

    _, entry = prepare_corpus_job(
        job_id="swe-agentic-v1", task_id=None, model="Qwen/Qwen3.8-27B", model_revision="rev1",
        model_architecture="Qwen3_5ForConditionalGeneration", checkpoint_sha256="c" * 64,
        from_profile=None, prompt_encoding=None, prompt_source=AGENTIC_SWE_ENVIRONMENT,
        prompt_count=3, prompt_start=0, renderer_id="renderers:qwen38@0.1.11",
        eos_token_id=248046, slots_per_prompt=2, max_new_tokens=8192, cap=0.05,
        min_incentive_share=0.0, audit_params={"audit_q": 1.0}, prompt_order="free",
        episode=_episode(),
    )
    return entry


def test_a_corpus_task_accepts_the_agentic_environment(supported):
    from reliquary.validator.task_config import resolve_task_config

    entry = _prepared_entry()
    config = resolve_task_config({entry.task_id: entry}, entry.task_id, profile_id=entry.profile_id,
                                 generation_contract=entry.contract)
    assert config is not None


def test_an_rl_task_refuses_the_agentic_environment():
    from reliquary.environment.abi import canonical_sha256
    from reliquary.shared.task_registry import MECHANISM_RL_DISCOVERED_PRICE, TaskEntry
    from reliquary.validator.task_config import TaskConfigError, resolve_task_config

    contract = {"model_id": "demo", "environments": {AGENTIC_SWE_ENVIRONMENT: {}}}
    params = {"start": 1.0, "decay": 0.99, "rounds_per_step": 1000, "deadband": 0.80,
              "snap": 1.20, "floor": 0.05, "cap": 0.6, "median_rounds": 4800,
              "last_good_fills": 50}
    entry = TaskEntry(task_id="t", profile_id="p", profile_sha256=canonical_sha256(contract),
                      mechanism=MECHANISM_RL_DISCOVERED_PRICE, params=params, status="active",
                      retired_at=None, contract=contract)
    with pytest.raises(TaskConfigError, match="does not install"):
        resolve_task_config({"t": entry}, "t", profile_id="p", generation_contract=contract)


def test_the_renderer_pin_is_enforced(monkeypatch):
    from tests.unit.test_corpus_job_episode import _episode

    episode = parse_job(_manifest(episode=_episode(renderer="renderers:qwen38@0.1.10"),
                                  renderer_id="renderers:qwen38@0.1.10")).episode
    assert "renderer" in agentic_swe.episode_support_refusal(episode, need_verifiers=False)
    monkeypatch.setattr(agentic_swe, "installed_env_commit", lambda: COMMIT)
    monkeypatch.setattr(agentic_swe, "_renderers_version", lambda: "0.1.10")
    assert "renderers" in agentic_swe.episode_support_refusal(_job().episode, need_verifiers=False)


def _editable(monkeypatch, path):
    class _Dist:
        def read_text(self, name):
            return json.dumps({"url": f"file://{path}", "dir_info": {"editable": True}})

    monkeypatch.setattr(agentic_swe.importlib.metadata, "distribution", lambda name: _Dist())


def test_a_dirty_editable_checkout_reports_no_commit(monkeypatch, tmp_path):
    import subprocess

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(["git", "-C", str(tmp_path), "-c", "user.name=t", "-c", "user.email=t@t",
                    "commit", "-q", "--allow-empty", "-m", "x"], check=True)
    _editable(monkeypatch, tmp_path)
    assert agentic_swe.installed_env_commit() is not None
    (tmp_path / "modified.py").write_text("x = 1\n")
    assert agentic_swe.installed_env_commit() is None


def test_a_hanging_git_reports_no_commit(monkeypatch, tmp_path):
    _editable(monkeypatch, tmp_path)

    def hang(*args, **kwargs):
        assert kwargs.get("timeout")
        raise agentic_swe.subprocess.TimeoutExpired(args[0], kwargs["timeout"])

    monkeypatch.setattr(agentic_swe.subprocess, "run", hang)
    assert agentic_swe.installed_env_commit() is None


def test_an_eval_job_cannot_carry_the_agentic_environment(supported):
    from reliquary.cli.main import prepare_corpus_job

    with pytest.raises(ValueError, match="agentic"):
        prepare_corpus_job(
            job_id="eval-x", task_id=None, model="Qwen/Qwen3.8-27B", model_revision="rev1",
            model_architecture="Qwen3_5ForConditionalGeneration", checkpoint_sha256="c" * 64,
            from_profile=None, prompt_encoding=None, prompt_source="eval-set:s:3:" + "a" * 64,
            prompt_count=3, prompt_start=0, renderer_id="chat-template-v1", eos_token_id=1,
            slots_per_prompt=2, max_new_tokens=8192, cap=0.05, min_incentive_share=0.0,
            audit_params={"audit_q": 1.0}, contract_environment=AGENTIC_SWE_ENVIRONMENT,
        )


def test_a_served_agentic_job_without_episode_is_refused(supported):
    with pytest.raises(CorpusPromptSourceError, match="episode"):
        prompt_job_for_spec(parse_job(_manifest(with_episode=False)))


def test_a_job_with_episode_on_another_source_is_refused(supported):
    with pytest.raises(CorpusPromptSourceError, match=AGENTIC_SWE_ENVIRONMENT):
        prompt_job_for_spec(_job(prompt_source="openmathinstruct"))


# -- ruling P13: the prompt the model sees carries verifiers' restricted-network notice --

def test_the_source_prompt_carries_the_network_notice():
    notice, _ = agentic_swe.network_notice()
    assert SweSource(ROWS).prompt(1) == "fix b\n\n" + notice
    assert SweSource([("x", "")]).prompt(0) == notice      # append_user_notice on empty content
    assert SweSource(ROWS).task_for(1).prompt == "fix b"   # the task itself is unchanged


def test_the_pinned_notice_is_verifiers_own():
    base = pytest.importorskip("verifiers.v1.dialects.base")
    assert agentic_swe.PINNED_NETWORK_NOTICE == base.CAPABILITY_NOTICE
    agentic_swe.network_notice.cache_clear()
    assert agentic_swe.network_notice() == (base.CAPABILITY_NOTICE, "verifiers")


def test_without_verifiers_the_pinned_notice_is_used(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def no_verifiers(name, *args, **kwargs):
        if name.startswith("verifiers"):
            raise ImportError(name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_verifiers)
    agentic_swe.network_notice.cache_clear()
    try:
        assert agentic_swe.network_notice() == (agentic_swe.PINNED_NETWORK_NOTICE, "pinned")
    finally:
        monkeypatch.undo()
        agentic_swe.network_notice.cache_clear()


def test_the_messages_mirror_append_user_notice():
    base = pytest.importorskip("verifiers.v1.dialects.base")
    for prompt in ("fix b", ""):
        messages = [{"role": "user", "content": prompt}]
        base.append_user_notice(messages)
        assert SweSource([("x", prompt)]).prompt(0) == messages[0]["content"]
