"""An external environment may hand over its cases instead of grading itself.

A packaged code environment is the one that needs this. If it graded its own
answers, the model-written Python would run behind the package's own rlimits
inside the validator process — which the package's own README says is not a
containment boundary — instead of behind gVisor. So the wheel supplies the
corpus, the task and the cases, and this repository executes them.
"""

import pytest

from reliquary.environment.registry import EnvironmentSpec

INTERNAL_SCORER = "reliquary.validator.admission:_score_opencode_adapter"
WHEEL_SCORER = "reliquary.environment.agentic.external:score_external_answers"


def _materials_spec(**overrides) -> EnvironmentSpec:
    base = dict(
        name="external_code_v1",
        factory_path="reliquary_code:CodeEnvironment",
        scorer_path=INTERNAL_SCORER,
        validator_authoritative_reward=True,
        admission_resource_class="sandbox",
        termination_policy="eos_or_cap",
        final_answer_policy="fenced_python",
        reward_lattice_policy="fractional-by-case-count-v1",
        attainable_rewards=(),
        contract_version="reliquary/python-cases/v1",
        environment_manifest_sha256="0" * 64,
        external_distribution="reliquary-code",
        external_artifact_resource="reliquary_code/artifact.json",
        reward_materializer_method="admission_reward_cases",
    )
    base.update(overrides)
    return EnvironmentSpec(**base)


def test_the_materials_shape_is_admitted() -> None:
    assert _materials_spec().reward_materializer_method == "admission_reward_cases"


def test_a_materials_environment_may_not_be_scored_by_its_own_wheel() -> None:
    """The whole point of handing over materials is that execution happens
    here. A wheel that both supplied and graded would have kept the code
    inside its own process."""
    with pytest.raises(ValueError, match="scored here"):
        _materials_spec(scorer_path=WHEEL_SCORER)


def test_a_materials_environment_must_ask_for_the_sandbox() -> None:
    with pytest.raises(ValueError, match="sandbox"):
        _materials_spec(admission_resource_class="cpu")


def test_a_wheel_may_not_both_grade_itself_and_declare_a_loose_lattice() -> None:
    """Either this repository can bound the number the wheel returns, or it
    computes the number itself. Declaring a fractional lattice while still
    grading would hand back a reward nothing here can check."""
    with pytest.raises(ValueError, match="binary rewards or hand over"):
        _materials_spec(
            scorer_path=WHEEL_SCORER,
            reward_materializer_method=None,
        )


def test_an_unknown_relay_name_is_refused() -> None:
    """`getattr` on a name the wrapper does not define returns None, and the
    environment would run with no materials at all rather than fail."""
    with pytest.raises(ValueError, match="hand over reward materials"):
        _materials_spec(reward_materializer_method="give_me_the_cases")


def test_binary_environments_are_unchanged() -> None:
    spec = _materials_spec(
        scorer_path=WHEEL_SCORER,
        admission_resource_class="cpu",
        final_answer_policy="json",
        reward_lattice_policy="binary-v1",
        attainable_rewards=(0.0, 1.0),
        contract_version="reliquary/answer-json/v1",
        reward_materializer_method=None,
    )
    assert spec.attainable_rewards == (0.0, 1.0)


def test_sandbox_environments_are_what_fills_the_legacy_cases_channel() -> None:
    """The name test this replaced would never have matched a packaged code
    environment, which is the same environment under another name."""
    from reliquary.validator.prompt_content import _cases_carry_the_target

    assert _cases_carry_the_target("opencodeinstruct") is True
    assert _cases_carry_the_target("openmathinstruct") is False
    assert _cases_carry_the_target("reliquarylogic_v1") is False
    assert _cases_carry_the_target("not-an-environment") is False


class _Backend:
    """The minimum a packaged code environment has to expose."""

    name = "external_code_v1"
    max_turns = 1
    validator_authoritative_reward = True

    def __init__(self, cases=None) -> None:
        self._cases = [{"entry": "solve", "args": [1], "expected": 2}] if cases is None else cases

    def __len__(self) -> int:
        return 4

    def task(self, index: int) -> dict:
        return {"id": f"t{index}", "prompt": f"problem {index}", "metadata": {}}

    def grade(self, index: int, completion: str) -> dict:
        return {"reward": 0.5, "success": False, "state_digest": "0" * 64}

    def admission_reward_cases(self, index: int):
        return self._cases


def _wrapper(backend=None):
    from reliquary.environment.agentic.external import ExternalAnswerEnvironment

    return ExternalAnswerEnvironment(backend or _Backend(), _materials_spec())


def test_the_relay_returns_the_packaged_cases() -> None:
    environment = _wrapper()
    problem = environment.get_problem(1)
    assert environment.admission_reward_cases(problem) == [
        {"entry": "solve", "args": [1], "expected": 2}
    ]


def test_the_relay_refuses_a_problem_the_environment_never_issued() -> None:
    """Reconstructed from its index before the backend is asked, so a caller
    cannot hand in a forged problem and read back somebody else's cases."""
    environment = _wrapper()
    forged = {**environment.get_problem(1), "prompt": "something else"}
    assert environment.admission_reward_cases(forged) == []


@pytest.mark.parametrize("cases", [[], "not-a-list", [["not-an-object"]], [None]])
def test_the_relay_refuses_malformed_materials(cases) -> None:
    environment = _wrapper(_Backend(cases=cases))
    with pytest.raises((TypeError, ValueError)):
        environment.admission_reward_cases(environment.get_problem(0))


def test_a_fractional_grade_is_bounded_to_the_unit_interval() -> None:
    """No fixed lattice to test membership against — the one that applies is
    derived from the case count — but a grade outside [0, 1] is still wrong."""
    environment = _wrapper()
    assert environment.compute_reward(environment.get_problem(0), "anything") == 0.5

    class _Rogue(_Backend):
        def grade(self, index: int, completion: str) -> dict:
            return {"reward": 4.0, "success": False, "state_digest": "0" * 64}

    rogue = _wrapper(_Rogue())
    with pytest.raises(ValueError, match=r"outside \[0, 1\]"):
        rogue.compute_reward(rogue.get_problem(0), "anything")
