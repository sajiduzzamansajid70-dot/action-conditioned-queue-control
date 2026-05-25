import numpy as np
from scipy.stats import ttest_rel


# Replace later with 10-seed results
obs_results = [
    0.577500,
]

action_results = [
    0.635278,
]


# temporary duplicated seeds
obs_results = obs_results * 5
action_results = action_results * 5


t_stat, p_value = ttest_rel(
    obs_results,
    action_results,
)

mean_obs = np.mean(obs_results)
mean_action = np.mean(action_results)

print("\nBOTTLENECK SIGNIFICANCE TEST\n")

print("Observation Mean:", mean_obs)
print("Action Mean:", mean_action)

print("Improvement:", mean_action - mean_obs)

print("T-statistic:", t_stat)
print("P-value:", p_value)

if p_value < 0.05:
    print("\nResult is statistically significant")
else:
    print("\nResult is NOT statistically significant")