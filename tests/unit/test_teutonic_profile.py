"""The Teutonic-I profile, and the per-environment reasoning mode it needs.

Every per-environment number in the profile was measured on this policy; these
tests pin what the numbers are for, so a later edit that moves one has to say
why rather than pass silently.
"""

from reliquary.protocol.profiles import PROFILES

PROFILE = PROFILES["teutonic-9b-reliquary-suite-v9-dev1"]


def test_the_profile_speaks_to_an_instruct_policy() -> None:
    """Every v4+ profile encoded raw for a base model. This policy was trained
    on the chat template and would see raw completion as a prompt it never met."""
    assert PROFILE.prompt_encoding == "chat_template"
    assert PROFILE.model_id == "ReliquaryForge/teutonic-i-graft-sft-cot-v2"
    assert len(PROFILE.model_revision) == 40


def test_it_declares_exactly_the_four_packaged_environments() -> None:
    assert set(PROFILE.environments) == {
        "reliquary_dapo_math_v1",
        "reliquary_instruction_following_v1",
        "reliquary_code_v1",
        "reliquary_telecom_solo_v1",
    }


def test_only_instruction_following_runs_direct() -> None:
    """The one environment whose grader reads the whole completion. Measured,
    band 4.2% thinking against 37.5% direct; telecom, graded on a database,
    goes the other way."""
    modes = {name: env.thinking for name, env in PROFILE.environments.items()}
    assert modes["reliquary_instruction_following_v1"] is False
    assert all(
        mode is None
        for name, mode in modes.items()
        if name != "reliquary_instruction_following_v1"
    )


def test_the_measured_budgets() -> None:
    envs = PROFILE.environments
    assert envs["reliquary_dapo_math_v1"].max_new_tokens == 32768
    assert envs["reliquary_instruction_following_v1"].max_new_tokens == 8192
    assert envs["reliquary_code_v1"].max_new_tokens == 8192


def test_telecom_is_an_episode_rendered_in_chatml() -> None:
    episode = PROFILE.environments["reliquary_telecom_solo_v1"].episode
    assert episode is not None
    assert episode.renderer_id == "reliquary-chatml-tools-v1"
    assert episode.max_turns == 40
    assert episode.max_action_tokens == 4096
    # The whole transcript: a 10,118-token opening, up to 24,324 generated and
    # small tool results come to about 37,000 at worst.
    assert episode.max_episode_tokens >= 37_000
    assert (
        PROFILE.environments["reliquary_telecom_solo_v1"].max_new_tokens
        == episode.max_episode_tokens
    )


def test_small_corpora_rotate_instead_of_running_dry() -> None:
    """One pass through each train split at 16 a window. The global horizon —
    a million windows — would exhaust telecom's 1,827 tickets in days and then
    serve nothing."""
    envs = PROFILE.environments
    assert envs["reliquary_telecom_solo_v1"].prompt_cooldown_windows == 1827 // 16
    assert envs["reliquary_dapo_math_v1"].prompt_cooldown_windows == 13931 // 16
    assert (
        envs["reliquary_instruction_following_v1"].prompt_cooldown_windows
        == 29435 // 16
    )
    assert envs["reliquary_code_v1"].prompt_cooldown_windows is None


def test_each_profile_entry_binds_the_spec_it_names() -> None:
    """The registry refuses a profile whose contract or manifest digest differs
    from the installed code; pinning it here catches the edit that forgets to
    move both."""
    from reliquary.environment.registry import get_environment_spec

    for name, env in PROFILE.environments.items():
        spec = get_environment_spec(name)
        assert env.environment_contract_id == spec.contract_version, name
        assert env.environment_manifest_sha256 == spec.environment_manifest_sha256, name
        if env.episode is not None:
            assert env.episode.renderer_id == spec.renderer_id, name


def test_the_reasoning_mode_reaches_the_generation_contract() -> None:
    """Miners and validators agree on it through the signed contract, as they
    do on `batch_target`; an undeclared mode leaves the contract untouched."""
    contract = PROFILE.to_generation_contract()["environments"]
    assert contract["reliquary_instruction_following_v1"]["thinking"] is False
    assert "thinking" not in contract["reliquary_dapo_math_v1"]


def test_every_historical_profile_keeps_thinking_on() -> None:
    """The flag is new and optional. Nothing that existed before it declares
    it, so every earlier contract keeps its exact bytes."""
    for profile_id, profile in PROFILES.items():
        if profile_id == PROFILE.profile_id:
            continue
        for env in profile.environments.values():
            assert env.thinking is None, profile_id


class _RecordingTemplate:
    """Records what each renderer asked the chat template for."""

    chat_template = "{% if enable_thinking %}<think>{% endif %}"

    def __init__(self) -> None:
        self.seen: list[object] = []

    def apply_chat_template(self, messages, *, add_generation_prompt, tokenize,
                            return_dict=False, enable_thinking=None):
        self.seen.append(enable_thinking)
        body = messages[0]["content"]
        return [1] + list(body.encode("utf-8")) if tokenize else f"<{enable_thinking}>{body}"


def test_both_renderers_ask_the_template_for_the_same_mode() -> None:
    """The tokens a miner generates from and the text `prompt_content_sha256`
    hashes come from two functions; a disagreement between them is not
    rejected, it is dropped silently at seal. So both take the mode, and both
    pass it on unchanged."""
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.validator.prompt_content import render_canonical_prompt

    for thinking in (True, False):
        tokenizer = _RecordingTemplate()
        encode_prompt(tokenizer, "p", thinking=thinking)
        render_canonical_prompt(tokenizer, "p", thinking=thinking)
        assert tokenizer.seen == [thinking, thinking]


def test_an_undeclared_mode_keeps_thinking_on() -> None:
    """Both used to hard-code `True`; the default must still be that."""
    from reliquary.protocol.tokens import encode_prompt
    from reliquary.validator.prompt_content import render_canonical_prompt

    tokenizer = _RecordingTemplate()
    encode_prompt(tokenizer, "p")
    render_canonical_prompt(tokenizer, "p")
    assert tokenizer.seen == [True, True]


def test_the_lookup_reads_the_active_profile(monkeypatch) -> None:
    import reliquary.constants as constants

    monkeypatch.setattr(constants, "ACTIVE_PROTOCOL_PROFILE", PROFILE)
    assert constants.thinking_for_environment("reliquary_instruction_following_v1") is False
    assert constants.thinking_for_environment("reliquary_dapo_math_v1") is True
    assert constants.thinking_for_environment("not-an-environment") is True


def _optimizer_knobs(profile_id: str, **overrides: str) -> list:
    import json
    import os
    import subprocess
    import sys

    env = {k: v for k, v in os.environ.items() if not k.startswith("RELIQUARY_")}
    env["RELIQUARY_PROTOCOL_PROFILE"] = profile_id
    env.update(overrides)
    script = (
        "import json; from reliquary import constants as c; "
        "print(json.dumps([c.OPTIMIZER_MASTER_WEIGHTS, c.OPTIMIZER_STATE_8BIT]))"
    )
    completed = subprocess.run(
        [sys.executable, "-c", script], check=True, capture_output=True,
        text=True, env=env,
    )
    return json.loads(completed.stdout)


def test_teutonic_steps_fp32_masters_with_8bit_moments() -> None:
    # 8.96B params: 2 weights + 2 gradients + 4 masters + 2 moments = 90 GB.
    assert _optimizer_knobs(PROFILE.profile_id) == [True, True]


def test_the_live_4b_run_keeps_its_optimizer() -> None:
    assert _optimizer_knobs(
        "qwen3-4b-base-dapo-reliquary-v1",
        RELIQUARY_EXPERIMENTAL_FILL_CLOSED_ENABLED="1",
    ) == [False, False]
