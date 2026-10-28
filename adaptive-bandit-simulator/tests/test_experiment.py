import pandas as pd

from bandit_sim.experiment import run_experiments
from bandit_sim.metrics import calculate_metrics


def test_experiment_record_count():
    results = run_experiments(n_runs=2, horizon=10)

    assert len(results) == 3 * 2 * 2 * 10


def test_metrics_have_expected_shape():
    results = run_experiments(n_runs=2, horizon=10)
    metrics = calculate_metrics(results)

    assert len(metrics) == 3 * 2 * 10
    assert metrics["runs"].eq(2).all()


def test_rewards_and_actions_are_valid():
    results = run_experiments(n_runs=2, horizon=10)

    assert results["reward"].isin([0, 1]).all()
    assert results["action"].between(0, 4).all()
    assert results["optimal_action"].isin([0, 1]).all()


def test_expected_regret_is_nonnegative():
    results = run_experiments(n_runs=2, horizon=10)

    assert (results["expected_regret"] >= 0).all()


def test_shared_environment_seeds():
    results = run_experiments(n_runs=2, horizon=10)

    seed_counts = (
        results.groupby(["environment", "run_number"])[
            "environment_seed"
        ]
        .nunique()
    )

    assert seed_counts.eq(1).all()


def test_experiments_are_reproducible():
    first = run_experiments(n_runs=2, horizon=10)
    second = run_experiments(n_runs=2, horizon=10)

    pd.testing.assert_frame_equal(first, second)
