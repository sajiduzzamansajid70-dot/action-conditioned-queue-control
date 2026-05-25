import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon


def cohens_d_paired(a, b):
    diff = a - b
    return diff.mean() / diff.std(ddof=1)


def main():
    df = pd.read_csv("data/processed/multiseed_bottleneck_anticipation.csv")

    metrics = [
        ("accuracy", "action_accuracy", "obs_accuracy"),
        ("recall", "action_recall", "obs_recall"),
        ("precision", "action_precision", "obs_precision"),
    ]

    print("\nBOTTLENECK STATISTICAL SUMMARY\n")

    for name, action_col, obs_col in metrics:
        action = df[action_col]
        obs = df[obs_col]

        improvement = action - obs

        t_stat, t_p = ttest_rel(action, obs)
        w_stat, w_p = wilcoxon(action, obs)
        d = cohens_d_paired(action, obs)

        print(name.upper())
        print("Observation mean:", obs.mean())
        print("Action mean:", action.mean())
        print("Mean improvement:", improvement.mean())
        print("Paired t-test p:", t_p)
        print("Wilcoxon p:", w_p)
        print("Cohen's d:", d)
        print("-" * 40)


if __name__ == "__main__":
    main()