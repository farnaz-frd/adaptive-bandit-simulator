# Adaptive Multi-Armed Bandit Simulator — Analysis Report

## 1. Objective

This experiment compares three action-selection strategies in stationary and non-stationary Bernoulli multi-armed bandit environments:

- Random selection
- Epsilon-greedy with sample-average updates
- Epsilon-greedy with constant-step updates

Each environment contains five arms. The experiment uses 50 independent runs and 1,000 steps per run, producing 300,000 step-level records across the three algorithms and two environments.

## 2. Experimental Setup

In the stationary environment, the arm reward probabilities remain `[0.1, 0.3, 0.5, 0.7, 0.9]`.

In the non-stationary environment, the probabilities reverse at step 501 to `[0.9, 0.7, 0.5, 0.3, 0.1]`.

Both epsilon-greedy agents use epsilon = 0.1. The sample-average agent updates action values using incremental sample averages. The constant-step agent uses alpha = 0.1, allowing recent rewards to have greater influence.

The random agent provides a baseline. Results are stored in SQLite and retrieved for metric calculation and visualization.

## 3. Average Reward Results

| Environment | Algorithm | Average reward |
|---|---|---:|
| Stationary | Epsilon-greedy sample average | 0.84768 |
| Stationary | Epsilon-greedy constant step | 0.80936 |
| Stationary | Random | 0.50212 |
| Non-stationary | Epsilon-greedy constant step | 0.74528 |
| Non-stationary | Epsilon-greedy sample average | 0.53990 |
| Non-stationary | Random | 0.49852 |

## 4. Interpretation

### Stationary environment

The sample-average agent achieves the highest average reward. Because the reward probabilities remain fixed, accumulating more observations generally improves its estimates of each arm's expected reward.

The constant-step agent also performs well, but its updates continue to give substantial weight to recent observations rather than allowing the influence of older observations to diminish naturally.

### Non-stationary environment

The constant-step agent achieves the highest average reward. When the reward probabilities reverse at step 501, previously learned estimates become outdated. Its fixed learning rate enables it to adapt more quickly to the changed reward distribution.

The sample-average agent adapts more slowly because its estimates incorporate the full history of rewards for each action. The random baseline does not effectively exploit the changing reward probabilities.

## 5. Evaluation Metrics

Four metrics are calculated for each algorithm and environment:

- **Average reward per step:** mean observed reward at each step across runs.
- **Average cumulative reward:** mean accumulated reward up to each step.
- **Optimal-action rate:** fraction of runs selecting an action with the highest expected reward at that step.
- **Average cumulative expected regret:** accumulated difference between the best available expected reward and the expected reward of the selected action.

The plots should be used together: reward measures realized outcomes, optimal-action rate measures action selection, and expected regret measures the opportunity cost of those choices. The non-stationary plots mark the probability reversal at step 501.

## 6. Conclusion

The results demonstrate that the choice of update rule should reflect the environment's dynamics. Sample-average updates are effective when reward probabilities are stationary, while constant-step updates are better suited to changing reward distributions.

The experiment also demonstrates a reproducible workflow for simulation, metric calculation, SQLite persistence, SQL-based comparison, and visualization.

## 7. Limitations

Results depend on the selected random seeds, epsilon, alpha, number of runs, and experiment horizon. The findings describe this configuration rather than proving that one update rule is universally superior. Additional experiments with different hyperparameters and more runs could test the robustness of the conclusions.
