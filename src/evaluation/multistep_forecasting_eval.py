import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from dataset import QueueSequenceDataset
from horizon_dataset import HorizonQueueDataset
from observation_lstm import ObservationOnlyLSTM
from action_conditioned_lstm import ActionConditionedLSTM


def make_loader(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=False)


def evaluate_obs(model, loader, loss_fn):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for batch in loader:
            pred_next, _ = model(batch["x_obs"])
            loss = loss_fn(pred_next, batch["y_next"])
            total += loss.item()

    return total / len(loader)


def evaluate_action(model, loader, loss_fn):
    model.eval()
    total = 0.0

    with torch.no_grad():
        for batch in loader:
            pred_next, _ = model(batch["x_obs"], batch["x_action"])
            loss = loss_fn(pred_next, batch["y_next"])
            total += loss.item()

    return total / len(loader)


def main():
    print("TRUE MULTI-HORIZON FORECASTING EVALUATION")

    train_dataset = QueueSequenceDataset(
        csv_path="data/raw/train.csv",
        seq_len=20,
        fit_scaler=True,
    )

    obs_model = ObservationOnlyLSTM()
    action_model = ActionConditionedLSTM()

    obs_model.load_state_dict(torch.load("data/processed/observation_lstm.pt"))
    action_model.load_state_dict(torch.load("data/processed/action_conditioned_lstm.pt"))

    loss_fn = nn.MSELoss()

    horizons = [1, 3, 5, 10]

    for horizon in horizons:
        dataset = HorizonQueueDataset(
            csv_path="data/raw/test_normal.csv",
            seq_len=20,
            horizon=horizon,
            scaler=train_dataset.scaler,
            fit_scaler=False,
        )

        loader = make_loader(dataset)

        obs_loss = evaluate_obs(obs_model, loader, loss_fn)
        action_loss = evaluate_action(action_model, loader, loss_fn)

        improvement = ((obs_loss - action_loss) / obs_loss) * 100

        print(f"\nHorizon t+{horizon}")
        print("Observation MSE:", obs_loss)
        print("Action-conditioned MSE:", action_loss)
        print("Improvement (%):", improvement)


if __name__ == "__main__":
    main()