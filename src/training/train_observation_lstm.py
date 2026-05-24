import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from dataset import QueueSequenceDataset
from observation_lstm import ObservationOnlyLSTM


def main():
    train_dataset = QueueSequenceDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        fit_scaler=True,
    )

    val_dataset = QueueSequenceDataset(
        csv_path="data/raw/validation.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    model = ObservationOnlyLSTM()

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    total_loss = 0.0

    for batch in train_loader:
        x_obs = batch["x_obs"]
        y_next = batch["y_next"]
        y_kpi = batch["y_kpi"]

        pred_next, pred_kpi = model(x_obs)

        loss_next = loss_fn(pred_next, y_next)
        loss_kpi = loss_fn(pred_kpi, y_kpi)
        loss = loss_next + loss_kpi

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_train_loss = total_loss / len(train_loader)
    print("Average train loss:", avg_train_loss)

    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for batch in val_loader:
            x_obs = batch["x_obs"]
            y_next = batch["y_next"]
            y_kpi = batch["y_kpi"]

            pred_next, pred_kpi = model(x_obs)

            loss_next = loss_fn(pred_next, y_next)
            loss_kpi = loss_fn(pred_kpi, y_kpi)
            loss = loss_next + loss_kpi

            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)
    print("Average validation loss:", avg_val_loss)

    torch.save(
        model.state_dict(),
        "data/processed/observation_lstm.pt",
    )

    print("Saved observation model")


if __name__ == "__main__":
    main()