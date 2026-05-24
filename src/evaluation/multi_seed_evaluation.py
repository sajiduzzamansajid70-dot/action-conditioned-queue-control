import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from torch.utils.data import DataLoader

from dataset import QueueSequenceDataset
from observation_lstm import ObservationOnlyLSTM
from action_conditioned_lstm import ActionConditionedLSTM


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, shuffle=False):
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=shuffle,
    )


def train_observation_model(train_loader, val_loader):

    model = ObservationOnlyLSTM()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.MSELoss()

    epochs = 5

    for epoch in range(epochs):

        model.train()

        for batch in train_loader:

            x_obs = batch["x_obs"]
            y_next = batch["y_next"]
            y_kpi = batch["y_kpi"]

            optimizer.zero_grad()

            pred_next, pred_kpi = model(x_obs)

            loss_next = loss_fn(pred_next, y_next)
            loss_kpi = loss_fn(pred_kpi, y_kpi)

            loss = loss_next + loss_kpi

            loss.backward()

            optimizer.step()

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

    return avg_val_loss


def train_action_model(train_loader, val_loader):

    model = ActionConditionedLSTM()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1e-3,
    )

    loss_fn = nn.MSELoss()

    epochs = 5

    for epoch in range(epochs):

        model.train()

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

    return avg_val_loss


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

    train_loader = make_loader(train_dataset, shuffle=True)

    val_loader = make_loader(val_dataset)

    seeds = [1, 2, 3, 4, 5]

    obs_results = []
    action_results = []

    for seed in seeds:

        print(f"\nRunning Seed {seed}")

        set_seed(seed)

        obs_loss = train_observation_model(
            train_loader,
            val_loader,
        )

        action_loss = train_action_model(
            train_loader,
            val_loader,
        )

        obs_results.append(obs_loss)
        action_results.append(action_loss)

        print("Observation loss:", obs_loss)
        print("Action-conditioned loss:", action_loss)

    print("\nFINAL MULTI-SEED RESULTS")

    print("\nObservation Results")
    print(obs_results)

    print("\nAction-conditioned Results")
    print(action_results)

    print("\nObservation Mean:", np.mean(obs_results))
    print("Observation Std:", np.std(obs_results))

    print("\nAction Mean:", np.mean(action_results))
    print("Action Std:", np.std(action_results))

    rows = []

    for seed, obs_loss, action_loss in zip(
        seeds,
        obs_results,
        action_results,
    ):

        rows.append({
            "seed": seed,
            "observation_loss": obs_loss,
            "action_loss": action_loss,
            "difference": obs_loss - action_loss,
            "improvement_percent": (
                (obs_loss - action_loss) / obs_loss
            ) * 100,
        })

    df = pd.DataFrame(rows)

    df.to_csv(
        "data/processed/multi_seed_results.csv",
        index=False,
    )

    print("\nSaved multi-seed results CSV")


if __name__ == "__main__":
    main()