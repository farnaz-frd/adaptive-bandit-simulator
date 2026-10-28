
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator


class BanditAgent(ABC):
    """Common interface for all bandit agents."""

    def __init__(self, n_arms: int = 5, seed: int | None = None):
        if n_arms < 1:
            raise ValueError("n_arms must be at least 1")

        self.n_arms = n_arms
        self.rng: Generator = np.random.default_rng(seed)

    def _validate_observation(self, action: int, reward: int) -> None:
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        if not 0 <= action < self.n_arms:
            raise ValueError("action is outside the valid arm range")
        if not isinstance(reward, (int, np.integer, bool, np.bool_)):
            raise TypeError("reward must be binary")
        if reward not in (0, 1):
            raise ValueError("reward must be 0 or 1")

    @abstractmethod
    def select_action(self) -> int:
        """Select an arm using the agent's current knowledge."""

    @abstractmethod
    def update(self, action: int, reward: int) -> None:
        """Update the agent using an observed action and reward."""


class RandomAgent(BanditAgent):
    """Uniformly random baseline; does not learn from rewards."""

    def select_action(self) -> int:
        return int(self.rng.integers(self.n_arms))

    def update(self, action: int, reward: int) -> None:
        self._validate_observation(action, reward)
        # Intentionally does not learn.


class EpsilonGreedySampleAverage(BanditAgent):
    """Epsilon-greedy agent with sample-average action values."""

    def __init__(
        self,
        n_arms: int = 5,
        epsilon: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(n_arms, seed)

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")

        self.epsilon = epsilon
        self.reset()

    def reset(self) -> None:
        self.q_values = np.zeros(self.n_arms, dtype=float)
        self.action_counts = np.zeros(self.n_arms, dtype=int)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))

        best_value = np.max(self.q_values)
        best_actions = np.flatnonzero(self.q_values == best_value)
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: int) -> None:
        self._validate_observation(action, reward)

        self.action_counts[action] += 1
        count = self.action_counts[action]

        # Incremental form of the sample-average update.
        self.q_values[action] += (
            reward - self.q_values[action]
        ) / count


class EpsilonGreedyConstantStep(BanditAgent):
    """Epsilon-greedy agent with a constant learning rate."""

    def __init__(
        self,
        n_arms: int = 5,
        epsilon: float = 0.1,
        alpha: float = 0.1,
        seed: int | None = None,
    ):
        super().__init__(n_arms, seed)

        if not 0.0 <= epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1")
        if not 0.0 < alpha <= 1.0:
            raise ValueError("alpha must be in (0, 1]")

        self.epsilon = epsilon
        self.alpha = alpha
        self.reset()

    def reset(self) -> None:
        self.q_values = np.zeros(self.n_arms, dtype=float)
        self.action_counts = np.zeros(self.n_arms, dtype=int)

    def select_action(self) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_arms))

        best_value = np.max(self.q_values)
        best_actions = np.flatnonzero(self.q_values == best_value)
        return int(self.rng.choice(best_actions))

    def update(self, action: int, reward: int) -> None:
        self._validate_observation(action, reward)

        self.action_counts[action] += 1

        # Constant-step-size update.
        self.q_values[action] += self.alpha * (
            reward - self.q_values[action]
        )
