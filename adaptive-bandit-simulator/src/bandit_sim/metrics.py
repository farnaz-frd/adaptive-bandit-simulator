
import pandas as pd


GROUP_COLUMNS = ["environment", "algorithm", "step"]


def calculate_metrics(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate step-level records across independent simulation runs."""
    required_columns = {
        *GROUP_COLUMNS,
        "run_number",
        "reward",
        "cumulative_reward",
        "optimal_action",
        "cumulative_expected_regret",
    }

    missing = required_columns.difference(results.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    metrics = (
        results.groupby(GROUP_COLUMNS, as_index=False)
        .agg(
            average_reward=("reward", "mean"),
            average_cumulative_reward=("cumulative_reward", "mean"),
            optimal_action_rate=("optimal_action", "mean"),
            average_cumulative_expected_regret=(
                "cumulative_expected_regret", "mean"
            ),
            runs=("run_number", "nunique"),
        )
        .sort_values(GROUP_COLUMNS)
        .reset_index(drop=True)
    )

    return metrics
