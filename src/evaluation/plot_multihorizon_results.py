import pandas as pd
import matplotlib.pyplot as plt


def main():
    df = pd.read_csv("data/processed/multihorizon_results.csv")

    plt.figure(figsize=(8, 5))

    plt.plot(
        df["horizon"],
        df["observation_mse"],
        marker="o",
        label="Observation-Only",
    )

    plt.plot(
        df["horizon"],
        df["action_conditioned_mse"],
        marker="o",
        label="Action-Conditioned",
    )

    plt.title("Multi-Horizon Forecasting Error")
    plt.xlabel("Forecast Horizon")
    plt.ylabel("MSE")
    plt.legend()
    plt.tight_layout()

    output_path = "data/processed/multihorizon_plot.png"

    plt.savefig(output_path, dpi=300)

    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()