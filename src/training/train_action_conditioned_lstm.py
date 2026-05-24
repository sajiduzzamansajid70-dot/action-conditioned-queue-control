import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from dataset import QueueSequenceDataset
from action_conditioned_lstm import ActionConditionedLSTM


def make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
    )


def main():
    print("Training Action-Conditioned LSTM")

    train_dataset = QueueSequenceDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        fit_scaler=True,
    )

    validation_dataset = QueueSequenceDataset(
        csv_path="data/raw/validation.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    train_loader = make_loader(train_dataset, shuffle=True)
    val_loader = make_loader(validation_dataset)

    model = ActionConditionedLSTM()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.MSELoss()

    epochs = 10

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0.0

        for batch in train_loader:
            x_obs = batch["x_obs"]
            x_action = batch["x_action"]
            y_next = batch["y_next"]
            y_kpi = batch["y_kpi"]

            optimizer.zero_grad()

            pred_next, pred_kpi = model(
                x_obs,
                x_action,
            )

            loss_next = loss_fn(pred_next, y_next)
            loss_kpi = loss_fn(pred_kpi, y_kpi)

            loss = loss_next + loss_kpi

            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        print("Average train loss:", avg_train_loss)

        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_obs = batch["x_obs"]
                x_action = batch["x_action"]
                y_next = batch["y_next"]
                y_kpi = batch["y_kpi"]

                pred_next, pred_kpi = model(
                    x_obs,
                    x_action,
                )

                loss_next = loss_fn(pred_next, y_next)
                loss_kpi = loss_fn(pred_kpi, y_kpi)

                loss = loss_next + loss_kpi

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        print("Average validation loss:", avg_val_loss)

    torch.save(
        model.state_dict(),
        "data/processed/action_conditioned_lstm.pt",
    )

    print("Saved action-conditioned model")


if __name__ == "__main__":
    main()