
from collections.abc import Callable

import numpy as np
import pandas as pd

from bandit_sim.agents import (
    BanditAgent,
    RandomAgent,
    EpsilonGreedySampleAverage,
    EpsilonGreedyConstantStep,
)
from bandit_sim.environments import BernoulliBanditEnv


AgentFactory = Callable[[int], BanditAgent]

BASE_SEED = 20261028

AGENT_FACTORIES: dict[str, AgentFactory] = {
    "random": lambda seed: RandomAgent(seed=seed),
    "epsilon_greedy_sample_average": lambda seed: (
        EpsilonGreedySampleAverage(epsilon=0.1, seed=seed)
    ),
    "epsilon_greedy_constant_step": lambda seed: (
        EpsilonGreedyConstantStep(
            epsilon=0.1, alpha=0.1, seed=seed
        )
    ),
}

ENVIRONMENTS = ("stationary", "non_stationary")


def run_experiments(
    n_runs: int = 50,
    horizon: int = 1000,
    base_seed: int = BASE_SEED,
) -> pd.DataFrame:
    """Run every agent in every environment and return step-level data."""
    if n_runs < 1:
        raise ValueError("n_runs must be at least 1")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")

    records = []

    for env_index, env_name in enumerate(ENVIRONMENTS):
        for run_number in range(1, n_runs + 1):

            # Same environment seed across agents for this run.
            environment_seed = (
                base_seed + env_index * 100_000 + run_number
            )

            for agent_index, (agent_name, factory) in enumerate(
                AGENT_FACTORIES.items()
            ):
                # Distinct, deterministic agent seed per combination.
                agent_seed = (
                    base_seed
                    + 1_000_000
                    + agent_index * 100_000
                    + env_index * 10_000
                    + run_number
                )

                env = BernoulliBanditEnv(
                    kind=env_name,
                    horizon=horizon,
                    seed=environment_seed,
                )
                agent = factory(agent_seed)

                cumulative_reward = 0
                cumulative_regret = 0.0

                for step in range(1, horizon + 1):
                    action = agent.select_action()
                    reward = env.step(action)

                    # Evaluation-only access to the true probabilities.
                    probabilities = env.probabilities_at(step)
                    optimal_probability = float(probabilities.max())
                    selected_probability = float(probabilities[action])

                    is_optimal = int(
                        np.isclose(
                            selected_probability,
                            optimal_probability,
                        )
                    )
                    expected_regret = (
                        optimal_probability - selected_probability
                    )

                    agent.update(action, reward)

                    cumulative_reward += reward
                    cumulative_regret += expected_regret

                    records.append({
                        "environment": env_name,
                        "algorithm": agent_name,
                        "run_number": run_number,
                        "environment_seed": environment_seed,
                        "agent_seed": agent_seed,
                        "step": step,
                        "action": action,
                        "reward": reward,
                        "selected_probability": selected_probability,
                        "optimal_action": is_optimal,
                        "expected_regret": expected_regret,
                        "cumulative_reward": cumulative_reward,
                        "cumulative_expected_regret": cumulative_regret,
                    })

    return pd.DataFrame.from_records(records)
