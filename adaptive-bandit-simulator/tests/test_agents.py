
import numpy as np
import pytest

from bandit_sim.agents import (
    RandomAgent,
    EpsilonGreedySampleAverage,
    EpsilonGreedyConstantStep,
)


def test_random_agent_selects_valid_actions():
    agent = RandomAgent(seed=42)

    actions = [agent.select_action() for _ in range(100)]

    assert all(0 <= action < 5 for action in actions)


def test_sample_average_update():
    agent = EpsilonGreedySampleAverage(n_arms=2, seed=1)

    agent.update(0, 1)
    agent.update(0, 0)
    agent.update(0, 1)

    assert np.isclose(agent.q_values[0], 2 / 3)
    assert agent.action_counts.tolist() == [3, 0]


def test_constant_step_update():
    agent = EpsilonGreedyConstantStep(
        n_arms=2, alpha=0.1, seed=1
    )

    agent.update(0, 1)
    assert np.isclose(agent.q_values[0], 0.1)

    agent.update(0, 0)
    assert np.isclose(agent.q_values[0], 0.09)


def test_unselected_arm_estimates_remain_unchanged():
    agent = EpsilonGreedySampleAverage(n_arms=3, seed=1)

    agent.update(1, 1)

    assert agent.q_values[0] == 0.0
    assert agent.q_values[1] == 1.0
    assert agent.q_values[2] == 0.0


def test_epsilon_greedy_returns_valid_actions():
    agents = [
        EpsilonGreedySampleAverage(epsilon=0.1, seed=10),
        EpsilonGreedyConstantStep(epsilon=0.1, seed=11),
    ]

    for agent in agents:
        for _ in range(100):
            assert agent.select_action() in range(5)


def test_invalid_epsilon_is_rejected():
    with pytest.raises(ValueError):
        EpsilonGreedySampleAverage(epsilon=1.1)

    with pytest.raises(ValueError):
        EpsilonGreedyConstantStep(epsilon=-0.1)


def test_invalid_reward_is_rejected():
    agent = EpsilonGreedySampleAverage()

    with pytest.raises(ValueError):
        agent.update(0, 2)


def test_invalid_action_is_rejected():
    agent = EpsilonGreedyConstantStep()

    with pytest.raises(ValueError):
        agent.update(5, 1)


def test_reset_clears_learning_state():
    agent = EpsilonGreedySampleAverage(n_arms=2, seed=1)
    agent.update(0, 1)

    agent.reset()

    assert agent.q_values.tolist() == [0.0, 0.0]
    assert agent.action_counts.tolist() == [0, 0]
