import sys
from pathlib import Path

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from horizon_dataset import HorizonQueueDataset
from rssm import RSSM


HORIZONS = [1, 3, 5, 10]


def evaluate_horizon(model, loader):

    model.eval()

    loss_fn = nn.MSELoss()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            x_obs = batch["x_obs"]

            x_action = batch["x_action"]

            y_future = batch["y_next"]

            pred_next, _, _ = model(
                x_obs,
                x_action,
            )

            loss = loss_fn(
                pred_next,
                y_future,
            )

            total_loss += loss.item()

    return total_loss / len(loader)


def main():

    print("RSSM MULTI-HORIZON EVALUATION")

    train_dataset = HorizonQueueDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        horizon=1,
        fit_scaler=True,
    )

    model = RSSM()

    model.load_state_dict(
        torch.load("data/processed/rssm.pt")
    )

    rows = []

    for horizon in HORIZONS:

        print(f"\nEvaluating horizon t+{horizon}")

        dataset = HorizonQueueDataset(
            csv_path="data/raw/test_normal.csv",
            seq_len=20,
            horizon=horizon,
            scaler=train_dataset.scaler,
            fit_scaler=False,
        )

        loader = DataLoader(
            dataset,
            batch_size=32,
            shuffle=False,
        )

        mse = evaluate_horizon(
            model,
            loader,
        )

        rows.append({
            "horizon": horizon,
            "rssm_mse": mse,
        })

        print("RSSM MSE:", mse)

    df = pd.DataFrame(rows)

    print("\nRSSM MULTI-HORIZON RESULTS")
    print(df)

    df.to_csv(
        "data/processed/rssm_multihorizon.csv",
        index=False,
    )

    print("\nSaved RSSM results")


if __name__ == "__main__":
    main()