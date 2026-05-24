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
from action_conditioned_lstm import ActionConditionedLSTM


def make_loader(dataset):
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
    )


def evaluate_observation_model(model, loader, loss_fn):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

            x_obs = batch["x_obs"]

            y_next = batch["y_next"]

            y_kpi = batch["y_kpi"]

            pred_next, pred_kpi = model(x_obs)

            loss_next = loss_fn(pred_next, y_next)

            loss_kpi = loss_fn(pred_kpi, y_kpi)

            loss = loss_next + loss_kpi

            total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_action_model(model, loader, loss_fn):

    model.eval()

    total_loss = 0.0

    with torch.no_grad():

        for batch in loader:

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

            total_loss += loss.item()

    return total_loss / len(loader)


def main():

    print("Stress Test Evaluation")

    train_dataset = QueueSequenceDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        fit_scaler=True,
    )

    test_normal_dataset = QueueSequenceDataset(
        csv_path="data/raw/test_normal.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    test_high_load_dataset = QueueSequenceDataset(
        csv_path="data/raw/test_high_load.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    test_high_failure_dataset = QueueSequenceDataset(
        csv_path="data/raw/test_high_failure.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    loaders = {
        "NORMAL": make_loader(test_normal_dataset),
        "HIGH LOAD": make_loader(test_high_load_dataset),
        "HIGH FAILURE": make_loader(test_high_failure_dataset),
    }

    obs_model = ObservationOnlyLSTM()

    action_model = ActionConditionedLSTM()

    loss_fn = nn.MSELoss()

    for name, loader in loaders.items():

        obs_loss = evaluate_observation_model(
            obs_model,
            loader,
            loss_fn,
        )

        action_loss = evaluate_action_model(
            action_model,
            loader,
            loss_fn,
        )

        print(f"\n{name}")

        print("Observation model loss:", obs_loss)

        print("Action-conditioned model loss:", action_loss)


if __name__ == "__main__":
    main()