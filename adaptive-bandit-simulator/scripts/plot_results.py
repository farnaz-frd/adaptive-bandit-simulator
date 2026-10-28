
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_sim.database import load_step_results
from bandit_sim.metrics import calculate_metrics


DB_PATH = PROJECT_ROOT / "results" / "bandit_results.db"
PLOT_DIR = PROJECT_ROOT / "results" / "plots"
EXPERIMENT_ID = "bandit_baseline_v1"

METRICS = {
    "average_reward": {
        "column": "average_reward",
        "title": "Average Reward per Step",
        "ylabel": "Average reward",
        "filename": "average_reward.png",
    },
    "average_cumulative_reward": {
        "column": "average_cumulative_reward",
        "title": "Average Cumulative Reward",
        "ylabel": "Cumulative reward",
        "filename": "average_cumulative_reward.png",
    },
    "optimal_action_rate": {
        "column": "optimal_action_rate",
        "title": "Optimal-Action Rate",
        "ylabel": "Rate",
        "filename": "optimal_action_rate.png",
    },
    "average_cumulative_expected_regret": {
        "column": "average_cumulative_expected_regret",
        "title": "Average Cumulative Expected Regret",
        "ylabel": "Expected regret",
        "filename": "average_cumulative_expected_regret.png",
    },
}


def main():
    PLOT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading step results from SQLite...")
    results = load_step_results(DB_PATH, EXPERIMENT_ID)
    print(results.head())

    print(f"Loaded {len(results):,} records.")
    metrics = calculate_metrics(results)

    environments = ["stationary", "non_stationary"]
    algorithms = sorted(metrics["algorithm"].unique())

    for spec in METRICS.values():
        fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=False)

        for ax, environment in zip(axes, environments):
            subset = metrics[metrics["environment"] == environment]

            for algorithm in algorithms:
                agent_data = subset[subset["algorithm"] == algorithm]
                ax.plot(
                    agent_data["step"],
                    agent_data[spec["column"]],
                    label=algorithm.replace("_", " "),
                    linewidth=1.8,
                )

            if environment == "non_stationary":
                ax.axvline(
                    x=501,
                    color="black",
                    linestyle="--",
                    linewidth=1.4,
                    label="Probability reversal (step 501)",
                )

            ax.set_title(environment.replace("_", " ").title())
            ax.set_xlabel("Step")
            ax.set_ylabel(spec["ylabel"])
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)

        fig.suptitle(spec["title"])
        fig.tight_layout()

        output_path = PLOT_DIR / spec["filename"]
        fig.savefig(output_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved: {output_path}")

    print("All plots generated.")


if __name__ == "__main__":
    main()
