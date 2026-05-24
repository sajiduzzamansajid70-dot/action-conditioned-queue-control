import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/processed/results_summary.csv")

    x = range(len(df))

    width = 0.35

    plt.figure(figsize=(8, 5))

    plt.bar(
        [i - width / 2 for i in x],
        df["observation_loss"],
        width=width,
        label="Observation-Only",
    )

    plt.bar(
        [i + width / 2 for i in x],
        df["action_conditioned_loss"],
        width=width,
        label="Action-Conditioned",
    )

    plt.xticks(x, df["scenario"])

    plt.title("Loss Comparison Across Stress Scenarios")

    plt.xlabel("Scenario")

    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    output_path = "data/processed/loss_comparison_plot.png"

    plt.savefig(output_path, dpi=300)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()