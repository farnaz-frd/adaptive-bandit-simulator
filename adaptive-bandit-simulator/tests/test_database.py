
import pandas as pd
import pytest

from bandit_sim.database import (
    initialize_database,
    save_results,
    load_step_results,
    load_average_rewards,
)
from bandit_sim.experiment import run_experiments


def test_save_and_reload_results(tmp_path):
    db_path = tmp_path / "test.db"
    results = run_experiments(n_runs=2, horizon=10)

    save_results(
        db_path, results, "test_experiment", 20261028, 2, 10
    )

    loaded = load_step_results(db_path, "test_experiment")

    assert len(loaded) == 120
    assert loaded["experiment_id"].eq("test_experiment").all()
    assert loaded["step"].between(1, 10).all()


def test_duplicate_experiment_id_is_rejected(tmp_path):
    db_path = tmp_path / "test.db"
    results = run_experiments(n_runs=1, horizon=5)

    save_results(db_path, results, "experiment_1", 20261028, 1, 5)

    with pytest.raises(ValueError, match="already exists"):
        save_results(
            db_path, results, "experiment_1", 20261028, 1, 5
        )


def test_average_reward_query(tmp_path):
    db_path = tmp_path / "test.db"
    results = run_experiments(n_runs=2, horizon=10)

    save_results(db_path, results, "experiment_1", 20261028, 2, 10)

    summary = load_average_rewards(db_path)

    assert len(summary) == 6
    assert summary["average_reward"].between(0, 1).all()
    assert summary["total_steps"].eq(20).all()


def test_empty_results_are_rejected(tmp_path):
    with pytest.raises(ValueError, match="empty"):
        save_results(
            tmp_path / "test.db",
            pd.DataFrame(),
            "empty_experiment",
            20261028,
            1,
            1,
        )


def test_database_can_be_initialized_twice(tmp_path):
    db_path = tmp_path / "test.db"

    initialize_database(db_path)
    initialize_database(db_path)

    # Repeated initialization should not fail.
