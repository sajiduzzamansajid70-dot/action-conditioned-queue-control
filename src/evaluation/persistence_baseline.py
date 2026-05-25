import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src" / "data"))

from dataset import QueueSequenceDataset


def evaluate_persistence(loader):
    loss_fn = nn.MSELoss()
    total_loss = 0.0

    for batch in loader:
        x_obs = batch["x_obs"]
        y_next = batch["y_next"]

        # Use last observed state as prediction
        pred_next = x_obs[:, -1, :]

        loss = loss_fn(pred_next, y_next)
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    train_dataset = QueueSequenceDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        fit_scaler=True,
    )

    datasets = {
        "VALIDATION": "data/raw/validation.csv",
        "TEST_NORMAL": "data/raw/test_normal.csv",
        "TEST_HIGH_LOAD": "data/raw/test_high_load.csv",
        "TEST_HIGH_FAILURE": "data/raw/test_high_failure.csv",
    }

    print("\nPERSISTENCE BASELINE RESULTS\n")

    for name, path in datasets.items():
        dataset = QueueSequenceDataset(
            csv_path=path,
            seq_len=20,
            scaler=train_dataset.scaler,
            fit_scaler=False,
        )

        loader = DataLoader(dataset, batch_size=32, shuffle=False)

        loss = evaluate_persistence(loader)

        print(name)
        print("Persistence MSE:", loss)
        print("-" * 40)


if __name__ == "__main__":
    main()