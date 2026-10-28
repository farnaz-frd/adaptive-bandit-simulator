import numpy as np
import pytest

from bandit_sim.environments import BernoulliBanditEnv


def test_stationary_probabilities_never_change():
    env = BernoulliBanditEnv("stationary")
    expected = [0.1, 0.3, 0.5, 0.7, 0.9]

    assert env.probabilities_at(1).tolist() == expected
    assert env.probabilities_at(500).tolist() == expected
    assert env.probabilities_at(1000).tolist() == expected


def test_non_stationary_change_occurs_at_step_501():
    env = BernoulliBanditEnv("non_stationary")
    original = [0.1, 0.3, 0.5, 0.7, 0.9]
    reversed_probs = [0.9, 0.7, 0.5, 0.3, 0.1]

    assert env.probabilities_at(500).tolist() == original
    assert env.probabilities_at(501).tolist() == reversed_probs


def test_reward_is_binary_and_step_advances():
    env = BernoulliBanditEnv("stationary", seed=42)
    reward = env.step(0)

    assert reward in (0, 1)
    assert env.current_step == 1


def test_invalid_action_is_rejected():
    env = BernoulliBanditEnv("stationary")

    with pytest.raises(ValueError):
        env.step(5)


def test_horizon_cannot_be_exceeded():
    env = BernoulliBanditEnv("stationary", horizon=1, seed=42)
    env.step(0)

    with pytest.raises(RuntimeError):
        env.step(0)


def test_reset_reproduces_rewards():
    env = BernoulliBanditEnv("stationary", seed=123)
    first_run = [env.step(4) for _ in range(20)]

    env.reset(seed=123)
    second_run = [env.step(4) for _ in range(20)]

    assert first_run == second_run


def test_probability_arrays_cannot_mutate_environment():
    env = BernoulliBanditEnv("stationary")
    probabilities = env.probabilities_at(1)
    probabilities[0] = 1.0

    assert np.isclose(env.probabilities_at(1)[0], 0.1)