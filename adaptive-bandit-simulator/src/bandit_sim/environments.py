from typing import Literal

import numpy as np
from numpy.random import Generator


EnvironmentType = Literal["stationary", "non_stationary"]

INITIAL_PROBABILITIES = np.array(
    [0.1, 0.3, 0.5, 0.7, 0.9],
    dtype=float,
)


class BernoulliBanditEnv:
    """Five-arm bandit with stationary or changing reward probabilities."""

    def __init__(
        self,
        kind: EnvironmentType,
        horizon: int = 1000,
        seed: int | None = None,
    ) -> None:
        if kind not in ("stationary", "non_stationary"):
            raise ValueError(f"Unknown environment type: {kind}")
        if horizon < 1:
            raise ValueError("horizon must be at least 1")

        self.kind = kind
        self.horizon = horizon
        self._initial_seed = seed
        self.rng: Generator
        self.current_step = 0
        self.reset(seed=seed)

    @property
    def n_arms(self) -> int:
        return len(INITIAL_PROBABILITIES)

    def probabilities_at(self, step: int) -> np.ndarray:
        """Return true probabilities for evaluation, not agent decisions."""
        if not 1 <= step <= self.horizon:
            raise ValueError("step must be within the simulation horizon")

        if self.kind == "non_stationary" and step > 500:
            return INITIAL_PROBABILITIES[::-1].copy()

        return INITIAL_PROBABILITIES.copy()

    def reset(self, seed: int | None = None) -> None:
        """Start a fresh run and reset the random-number generator."""
        self.current_step = 0
        self.rng = np.random.default_rng(seed)

    def step(self, action: int) -> int:
        """Execute one action and return its binary reward."""
        if not isinstance(action, (int, np.integer)):
            raise TypeError("action must be an integer")
        if not 0 <= action < self.n_arms:
            raise ValueError(f"action must be between 0 and {self.n_arms - 1}")
        if self.current_step >= self.horizon:
            raise RuntimeError("Simulation horizon has been reached")

        next_step = self.current_step + 1
        probability = self.probabilities_at(next_step)[action]
        reward = int(self.rng.random() < probability)
        self.current_step = next_step
        return reward