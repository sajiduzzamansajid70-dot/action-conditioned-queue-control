import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from dataset import QueueSequenceDataset
from rssm import RSSM


def main():
    print("Training RSSM")

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

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    model = RSSM()

    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
    loss_fn = nn.MSELoss()

    epochs = 30
    beta = 0.03

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")

        model.train()
        train_loss = 0.0
        train_kl = 0.0

        for batch in train_loader:
            x_obs = batch["x_obs"]
            x_action = batch["x_action"]
            y_next = batch["y_next"]
            y_kpi = batch["y_kpi"]

            pred_next, pred_kpi, kl_loss = model(x_obs, x_action)

            recon_loss = (
                loss_fn(pred_next, y_next)
                + loss_fn(pred_kpi, y_kpi)
            )

            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_kl += kl_loss.item()

        print("Average train loss:", train_loss / len(train_loader))
        print("Average KL loss:", train_kl / len(train_loader))

        model.eval()
        val_loss = 0.0
        val_kl = 0.0

        with torch.no_grad():
            for batch in val_loader:
                x_obs = batch["x_obs"]
                x_action = batch["x_action"]
                y_next = batch["y_next"]
                y_kpi = batch["y_kpi"]

                pred_next, pred_kpi, kl_loss = model(x_obs, x_action)

                recon_loss = (
                    loss_fn(pred_next, y_next)
                    + loss_fn(pred_kpi, y_kpi)
                )

                loss = recon_loss + beta * kl_loss

                val_loss += loss.item()
                val_kl += kl_loss.item()

        print("Average validation loss:", val_loss / len(val_loader))
        print("Average validation KL:", val_kl / len(val_loader))

    torch.save(
        model.state_dict(),
        "data/processed/rssm.pt",
    )

    print("\nSaved RSSM model")


if __name__ == "__main__":
    main()