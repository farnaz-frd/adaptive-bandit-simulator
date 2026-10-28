# Adaptive Multi-Armed Bandit Simulator

A reproducible simulation framework for comparing multi-armed bandit algorithms in stationary and non-stationary environments, with step-level SQLite persistence, metric calculation, SQL-based analysis, and visualization.

## Project Objectives

This project evaluates three action-selection strategies across two Bernoulli bandit environments:

- Random action selection
- Epsilon-greedy with sample-average updates
- Epsilon-greedy with constant-step updates

The objective is to compare their reward, cumulative reward, optimal-action rate, and expected regret under stable and changing reward distributions.

## Environment Configuration

Both environments have five arms.

**Stationary environment**

The arm reward probabilities remain fixed:

`[0.1, 0.3, 0.5, 0.7, 0.9]`

**Non-stationary environment**

The reward probabilities reverse after step 500. Starting at step 501, they become:

`[0.9, 0.7, 0.5, 0.3, 0.1]`

Each reward is generated from a Bernoulli distribution using the selected arm's current probability.

## Algorithms

| Algorithm | Configuration | Update rule |
|---|---|---|
| Random | No learning parameters | Selects an arm randomly |
| Epsilon-greedy sample average | ε = 0.1 | Updates estimates using incremental sample averages |
| Epsilon-greedy constant step | ε = 0.1, α = 0.1 | Updates estimates using a fixed learning rate |

Both epsilon-greedy algorithms explore with probability ε and otherwise select an arm with the highest estimated value. Ties are broken randomly.

## Experimental Design

- Independent runs per algorithm and environment: 50
- Steps per run: 1,000
- Algorithms: 3
- Environments: 2
- Total step-level records: 300,000

Random seeds are assigned deterministically to support reproducibility.

## Project Structure

```text
adaptive-bandit-simulator/
├── src/
│   └── bandit_sim/
│       ├── environments.py
│       ├── agents.py
│       ├── experiment.py
│       ├── metrics.py
│       └── database.py
├── tests/
├── scripts/
│   ├── run_experiments.py
│   └── plot_results.py
├── sql/
│   └── average_reward_comparison.sql
├── results/
│   ├── bandit_results.db
│   └── plots/
├── reports/
│   └── analysis.md
├── requirements.txt
└── README.md
```

## Installation

Python 3.12 is recommended.

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Run the Tests

From the project root, run:

```powershell
$env:PYTHONPATH = "src"
python -m pytest -q
```

The current test suite contains 27 tests covering environments, agents, experiments, and database operations.

## Run the Experiments

```powershell
python scripts/run_experiments.py
```

This runs the full experiment and saves the step-level results to:

`results/bandit_results.db`

The experiment script uses the ID `bandit_baseline_v1`. The database layer rejects duplicate experiment IDs to avoid accidentally overwriting an existing experiment. To create a new experiment, change the ID in the script.

## Generate the Plots

```powershell
python scripts/plot_results.py
```

The plotting script loads step-level records from SQLite, calculates aggregate metrics, and saves the figures to `results/plots/`.

Generated figures:

- `average_reward.png`
- `average_cumulative_reward.png`
- `optimal_action_rate.png`
- `average_cumulative_expected_regret.png`

The non-stationary plots mark the probability reversal at step 501.

## Metrics

- **Average reward per step:** Mean observed reward at each step across runs.
- **Average cumulative reward:** Mean accumulated reward up to each step.
- **Optimal-action rate:** Fraction of runs selecting an arm with the highest expected reward at that step.
- **Average cumulative expected regret:** Mean accumulated difference between the best available expected reward and the expected reward of the selected arm.

## Results

The following average rewards were queried from SQLite for `bandit_baseline_v1`.

| Environment | Algorithm | Average reward |
|---|---|---:|
| Stationary | Epsilon-greedy sample average | 0.84768 |
| Stationary | Epsilon-greedy constant step | 0.80936 |
| Stationary | Random | 0.50212 |
| Non-stationary | Epsilon-greedy constant step | 0.74528 |
| Non-stationary | Epsilon-greedy sample average | 0.53990 |
| Non-stationary | Random | 0.49852 |

In the stationary environment, the sample-average algorithm achieves the highest average reward. In the non-stationary environment, the constant-step algorithm performs best because its fixed learning rate allows it to adapt more quickly to changed reward probabilities.

See `reports/analysis.md` for further discussion and `results/plots/` for metric visualizations.

## Database and SQL Analysis

SQLite stores experiment metadata, algorithm configurations, simulation runs, and step-level results in separate related tables.

The SQL comparison query is available in:

`sql/average_reward_comparison.sql`

It joins step results with simulation runs, algorithm configurations, and experiment metadata to compare average rewards by algorithm and environment.

## Limitations

The results depend on the chosen learning parameters, random seeds, number of runs, and experiment horizon. They demonstrate the behavior of these configurations rather than proving that one algorithm is universally superior.
