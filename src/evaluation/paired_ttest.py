import pandas as pd
from scipy.stats import ttest_rel


def main():
    df = pd.read_csv("data/processed/multi_seed_results.csv")

    obs = df["observation_loss"]
    action = df["action_loss"]

    t_stat, p_value = ttest_rel(obs, action)

    print("\nPAIRED T-TEST RESULTS\n")
    print("Observation mean:", obs.mean())
    print("Action mean:", action.mean())
    print("Mean difference:", (obs - action).mean())
    print("T-statistic:", t_stat)
    print("P-value:", p_value)

    if p_value < 0.05:
        print("\nResult: Statistically significant at p < 0.05")
    else:
        print("\nResult: Not statistically significant at p < 0.05")


if __name__ == "__main__":
    main()