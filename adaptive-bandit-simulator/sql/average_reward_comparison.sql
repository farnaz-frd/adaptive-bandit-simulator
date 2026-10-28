SELECT
    r.environment,
    c.algorithm,
    ROUND(AVG(s.reward), 5) AS average_reward,
    COUNT(*) AS total_steps
FROM step_results AS s
JOIN simulation_runs AS r
    ON s.run_id = r.run_id
JOIN algorithm_configs AS c
    ON r.config_id = c.config_id
JOIN experiments AS e
    ON r.experiment_id = e.experiment_id
WHERE e.experiment_id = 'bandit_baseline_v1'
GROUP BY r.environment, c.algorithm
ORDER BY r.environment, average_reward DESC;
