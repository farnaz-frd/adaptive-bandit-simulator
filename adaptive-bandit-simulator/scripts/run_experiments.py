
from pathlib import Path
import sys

# Allow running this script directly from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bandit_sim.experiment import run_experiments, BASE_SEED
from bandit_sim.database import save_results, load_average_rewards


def main():
    n_runs = 50
    horizon = 1000
    experiment_id = "bandit_baseline_v1"
    db_path = PROJECT_ROOT / "results" / "bandit_results.db"

    print("Running experiments...")
    results = run_experiments(
        n_runs=n_runs,
        horizon=horizon,
        base_seed=BASE_SEED,
    )
    print(f"Generated {len(results):,} step records.")

    print("Saving results to SQLite...")
    save_results(
        db_path=db_path,
        results=results,
        experiment_id=experiment_id,
        base_seed=BASE_SEED,
        n_runs=n_runs,
        horizon=horizon,
    )

    print(f"Database saved: {db_path}")
    print("\nAverage rewards queried from SQLite:")
    print(load_average_rewards(db_path).to_string(index=False))


if __name__ == "__main__":
    main()
