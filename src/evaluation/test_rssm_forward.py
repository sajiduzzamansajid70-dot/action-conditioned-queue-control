import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from dataset import QueueSequenceDataset
from rssm import RSSM


def main():
    print("Testing RSSM forward pass")

    train_dataset = QueueSequenceDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        fit_scaler=True,
    )

    loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
    )

    batch = next(iter(loader))

    model = RSSM()

    pred_next, pred_kpi, kl_loss = model(
        batch["x_obs"],
        batch["x_action"],
    )

    print("pred_next shape:", pred_next.shape)
    print("pred_kpi shape:", pred_kpi.shape)
    print("kl_loss:", kl_loss.item())


if __name__ == "__main__":
    main()