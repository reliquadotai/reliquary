"""verify_reward_claim: re-runs env.compute_reward and checks miner's claim."""

from reliquary.validator.verifier import verify_reward_claim


class FakeEnv:
    """Deterministic env: reward = 1.0 iff completion contains 'CORRECT'."""

    def compute_reward(self, problem, completion):
        return 1.0 if "CORRECT" in completion else 0.0


def test_reward_matches_claim_accepted():
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    assert verify_reward_claim(env, problem, "this is CORRECT", claimed=1.0) is True


def test_reward_mismatches_claim_rejected():
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    # Miner claims 1.0 but env says 0.0 (text doesn't contain CORRECT)
    assert verify_reward_claim(env, problem, "wrong answer", claimed=1.0) is False


def test_float_tolerance():
    """Continuous rewards: match within 1e-6."""
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    # FakeEnv returns exact 1.0; miner claims 1.0000001 (within tolerance)
    assert verify_reward_claim(
        env, problem, "CORRECT", claimed=1.0000001
    ) is True


def test_wide_float_divergence_rejected():
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    assert verify_reward_claim(
        env, problem, "CORRECT", claimed=0.5  # env says 1.0
    ) is False


def test_non_finite_claim_rejected():
    env = FakeEnv()
    problem = {"prompt": "q", "ground_truth": "a"}
    assert verify_reward_claim(
        env, problem, "CORRECT", claimed=float("nan")
    ) is False


def test_non_finite_environment_reward_rejected():
    class NonFiniteEnv:
        def compute_reward(self, problem, completion):
            return float("inf")

    assert verify_reward_claim(
        NonFiniteEnv(), {}, "completion", claimed=1.0
    ) is False
