import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/processed/results_summary.csv")

    plt.figure(figsize=(8, 5))

    plt.bar(
        df["scenario"],
        df["improvement_percent"],
    )

    plt.title("Action-Conditioned LSTM Improvement Across Stress Tests")
    plt.xlabel("Scenario")
    plt.ylabel("Improvement over Observation-Only Model (%)")

    plt.tight_layout()

    output_path = "data/processed/improvement_plot.png"
    plt.savefig(output_path, dpi=300)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()