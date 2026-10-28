
import json
import sqlite3
from pathlib import Path

import pandas as pd


SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    experiment_id TEXT PRIMARY KEY,
    base_seed INTEGER NOT NULL,
    n_runs INTEGER NOT NULL CHECK (n_runs > 0),
    horizon INTEGER NOT NULL CHECK (horizon > 0),
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS algorithm_configs (
    config_id INTEGER PRIMARY KEY AUTOINCREMENT,
    algorithm TEXT NOT NULL,
    parameters_json TEXT NOT NULL,
    UNIQUE (algorithm, parameters_json)
);

CREATE TABLE IF NOT EXISTS simulation_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT NOT NULL,
    config_id INTEGER NOT NULL,
    environment TEXT NOT NULL,
    run_number INTEGER NOT NULL,
    environment_seed INTEGER NOT NULL,
    agent_seed INTEGER NOT NULL,

    FOREIGN KEY (experiment_id)
        REFERENCES experiments(experiment_id),
    FOREIGN KEY (config_id)
        REFERENCES algorithm_configs(config_id),

    UNIQUE (experiment_id, config_id, environment, run_number)
);

CREATE TABLE IF NOT EXISTS step_results (
    run_id INTEGER NOT NULL,
    step INTEGER NOT NULL CHECK (step > 0),
    action INTEGER NOT NULL CHECK (action BETWEEN 0 AND 4),
    reward INTEGER NOT NULL CHECK (reward IN (0, 1)),
    selected_probability REAL NOT NULL,
    optimal_action INTEGER NOT NULL CHECK (optimal_action IN (0, 1)),
    expected_regret REAL NOT NULL CHECK (expected_regret >= 0),
    cumulative_reward INTEGER NOT NULL,
    cumulative_expected_regret REAL NOT NULL,

    PRIMARY KEY (run_id, step),
    FOREIGN KEY (run_id)
        REFERENCES simulation_runs(run_id)
);

CREATE INDEX IF NOT EXISTS idx_runs_experiment
    ON simulation_runs(experiment_id);

CREATE INDEX IF NOT EXISTS idx_runs_comparison
    ON simulation_runs(environment, config_id);

CREATE INDEX IF NOT EXISTS idx_steps_step
    ON step_results(step);
"""


def connect_db(db_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database with foreign-key enforcement enabled."""
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(path)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database(db_path: str | Path) -> None:
    """Create tables and indexes if they do not already exist."""
    with connect_db(db_path) as connection:
        connection.executescript(SCHEMA)


def save_results(
    db_path: str | Path,
    results: pd.DataFrame,
    experiment_id: str,
    base_seed: int,
    n_runs: int,
    horizon: int,
) -> None:
    """Persist one complete experiment in a single transaction."""
    if results.empty:
        raise ValueError("Cannot save empty results")

    if results.duplicated(
        ["environment", "algorithm", "run_number", "step"]
    ).any():
        raise ValueError("Duplicate step records detected")

    expected_rows = 2 * 3 * n_runs * horizon
    if len(results) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(results)}"
        )

    initialize_database(db_path)

    with connect_db(db_path) as connection:
        # Fail rather than silently overwrite an existing experiment.
        if connection.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone():
            raise ValueError(
                f"Experiment {experiment_id!r} already exists"
            )

        # Insert experiment metadata.
        connection.execute(
            """
            INSERT INTO experiments
                (experiment_id, base_seed, n_runs, horizon)
            VALUES (?, ?, ?, ?)
            """,
            (experiment_id, base_seed, n_runs, horizon),
        )

        config_ids = {}

        for algorithm in sorted(results["algorithm"].unique()):
            if algorithm == "random":
                params = {}
            elif algorithm == "epsilon_greedy_sample_average":
                params = {"epsilon": 0.1, "initial_q": 0.0,
                          "tie_breaking": "random"}
            elif algorithm == "epsilon_greedy_constant_step":
                params = {"epsilon": 0.1, "alpha": 0.1,
                          "initial_q": 0.0, "tie_breaking": "random"}
            else:
                raise ValueError(f"Unknown algorithm: {algorithm}")

            parameters_json = json.dumps(
                params, sort_keys=True, separators=(",", ":")
            )

            connection.execute(
                """
                INSERT INTO algorithm_configs
                    (algorithm, parameters_json)
                VALUES (?, ?)
                ON CONFLICT (algorithm, parameters_json) DO NOTHING
                """,
                (algorithm, parameters_json),
            )

            config_id = connection.execute(
                """
                SELECT config_id FROM algorithm_configs
                WHERE algorithm = ? AND parameters_json = ?
                """,
                (algorithm, parameters_json),
            ).fetchone()[0]

            config_ids[algorithm] = config_id

        run_ids = {}

        run_columns = [
            "environment", "algorithm", "run_number",
            "environment_seed", "agent_seed",
        ]

        runs = results[run_columns].drop_duplicates()

        for row in runs.itertuples(index=False, name=None):
            environment, algorithm, run_number, env_seed, agent_seed = row

            cursor = connection.execute(
                """
                INSERT INTO simulation_runs
                    (experiment_id, config_id, environment,
                     run_number, environment_seed, agent_seed)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    experiment_id,
                    config_ids[algorithm],
                    environment,
                    int(run_number),
                    int(env_seed),
                    int(agent_seed),
                ),
            )

            run_ids[
                (environment, algorithm, int(run_number))
            ] = cursor.lastrowid

        step_records = []

        for row in results.itertuples(index=False):
            run_id = run_ids[
                (row.environment, row.algorithm, int(row.run_number))
            ]

            step_records.append((
                run_id,
                int(row.step),
                int(row.action),
                int(row.reward),
                float(row.selected_probability),
                int(row.optimal_action),
                float(row.expected_regret),
                int(row.cumulative_reward),
                float(row.cumulative_expected_regret),
            ))

        connection.executemany(
            """
            INSERT INTO step_results (
                run_id, step, action, reward, selected_probability,
                optimal_action, expected_regret, cumulative_reward,
                cumulative_expected_regret
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            step_records,
        )


def load_step_results(
    db_path: str | Path,
    experiment_id: str,
) -> pd.DataFrame:
    """Retrieve step-level results for a saved experiment."""
    with connect_db(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                e.experiment_id,
                r.environment,
                c.algorithm,
                r.run_number,
                r.environment_seed,
                r.agent_seed,
                s.step,
                s.action,
                s.reward,
                s.selected_probability,
                s.optimal_action,
                s.expected_regret,
                s.cumulative_reward,
                s.cumulative_expected_regret
            FROM step_results AS s
            JOIN simulation_runs AS r USING (run_id)
            JOIN experiments AS e USING (experiment_id)
            JOIN algorithm_configs AS c USING (config_id)
            WHERE e.experiment_id = ?
            ORDER BY r.environment, c.algorithm, r.run_number, s.step
            """,
            connection,
            params=(experiment_id,),
        )


def load_average_rewards(db_path: str | Path) -> pd.DataFrame:
    """Compare average observed rewards by algorithm and environment."""
    with connect_db(db_path) as connection:
        return pd.read_sql_query(
            """
            SELECT
                r.environment,
                c.algorithm,
                AVG(s.reward) AS average_reward,
                COUNT(*) AS total_steps
            FROM step_results AS s
            JOIN simulation_runs AS r USING (run_id)
            JOIN algorithm_configs AS c USING (config_id)
            GROUP BY r.environment, c.algorithm
            ORDER BY r.environment, average_reward DESC
            """,
            connection,
        )
