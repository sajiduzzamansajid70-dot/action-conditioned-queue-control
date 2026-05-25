import sys
from pathlib import Path

import torch
import pandas as pd
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from dataset import QueueSequenceDataset
from observation_lstm import ObservationOnlyLSTM
from action_conditioned_lstm import ActionConditionedLSTM
from observation_gru import ObservationOnlyGRU
from action_conditioned_gru import ActionConditionedGRU


QUEUE_COLS = ["s1", "s2", "s3a", "s3b", "s4"]


def make_loader(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=False)


def evaluate_model(model, loader, model_type):
    model.eval()

    squared_errors = torch.zeros(5)
    count = 0

    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"]
            x_action = batch["x_action"]
            y_next = batch["y_next"]

            if model_type == "obs":
                pred_next, _ = model(x_obs)
            else:
                pred_next, _ = model(x_obs, x_action)

            # first 5 columns are queue predictions
            pred_q = pred_next[:, :5]
            true_q = y_next[:, :5]

            squared_errors += ((pred_q - true_q) ** 2).sum(dim=0)
            count += pred_q.shape[0]

    rmse = torch.sqrt(squared_errors / count)

    return rmse.tolist()


def main():
    print("STATION-LEVEL RMSE ANALYSIS")

    train_dataset = QueueSequenceDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        fit_scaler=True,
    )

    test_dataset = QueueSequenceDataset(
        csv_path="data/raw/test_normal.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    loader = make_loader(test_dataset)

    models = [
        ("Observation LSTM", ObservationOnlyLSTM(), "obs", "data/processed/observation_lstm.pt"),
        ("Action LSTM", ActionConditionedLSTM(), "action", "data/processed/action_conditioned_lstm.pt"),
        ("Observation GRU", ObservationOnlyGRU(), "obs", "data/processed/observation_gru.pt"),
        ("Action GRU", ActionConditionedGRU(), "action", "data/processed/action_conditioned_gru.pt"),
    ]

    rows = []

    for name, model, model_type, path in models:
        model.load_state_dict(torch.load(path))

        rmse_values = evaluate_model(
            model=model,
            loader=loader,
            model_type=model_type,
        )

        row = {"model": name}

        for station, value in zip(QUEUE_COLS, rmse_values):
            row[station] = value

        rows.append(row)

    df = pd.DataFrame(rows)

    print("\nSTATION RMSE TABLE\n")
    print(df)

    df.to_csv(
        "data/processed/station_level_rmse.csv",
        index=False,
    )

    print("\nSaved to data/processed/station_level_rmse.csv")


if __name__ == "__main__":
    main()