import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import ttest_rel
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parents[2]

sys.path.append(str(ROOT_DIR / "src" / "data"))
sys.path.append(str(ROOT_DIR / "src" / "models"))

from dataset import QueueSequenceDataset
from observation_lstm import ObservationOnlyLSTM
from action_conditioned_lstm import ActionConditionedLSTM


WARNING_THRESHOLD = 2.5


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_loader(dataset, shuffle=False):
    return DataLoader(dataset, batch_size=32, shuffle=shuffle)


def train_observation_model(train_loader):
    model = ObservationOnlyLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 5

    for _ in range(epochs):
        model.train()

        for batch in train_loader:
            x_obs = batch["x_obs"]
            y_next = batch["y_next"]
            y_kpi = batch["y_kpi"]

            pred_next, pred_kpi = model(x_obs)

            loss = loss_fn(pred_next, y_next) + loss_fn(pred_kpi, y_kpi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def train_action_model(train_loader):
    model = ActionConditionedLSTM()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    epochs = 5

    for _ in range(epochs):
        model.train()

        for batch in train_loader:
            x_obs = batch["x_obs"]
            x_action = batch["x_action"]
            y_next = batch["y_next"]
            y_kpi = batch["y_kpi"]

            pred_next, pred_kpi = model(x_obs, x_action)

            loss = loss_fn(pred_next, y_next) + loss_fn(pred_kpi, y_kpi)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model


def evaluate_bottleneck(model, loader, model_type):
    model.eval()

    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0

    with torch.no_grad():
        for batch in loader:
            x_obs = batch["x_obs"]
            x_action = batch["x_action"]
            y_next = batch["y_next"]

            if model_type == "obs":
                pred_next, _ = model(x_obs)
            else:
                pred_next, _ = model(x_obs, x_action)

            pred_q = pred_next[:, :5]
            true_q = y_next[:, :5]

            pred_bottleneck = (pred_q >= WARNING_THRESHOLD).any(dim=1)
            true_bottleneck = (true_q >= WARNING_THRESHOLD).any(dim=1)

            true_positive += ((pred_bottleneck == 1) & (true_bottleneck == 1)).sum().item()
            true_negative += ((pred_bottleneck == 0) & (true_bottleneck == 0)).sum().item()
            false_positive += ((pred_bottleneck == 1) & (true_bottleneck == 0)).sum().item()
            false_negative += ((pred_bottleneck == 0) & (true_bottleneck == 1)).sum().item()

    total = true_positive + true_negative + false_positive + false_negative

    accuracy = (true_positive + true_negative) / total
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)

    return accuracy, precision, recall


def main():
    print("MULTI-SEED BOTTLENECK ANTICIPATION")
    print("Warning threshold:", WARNING_THRESHOLD)

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

    train_loader = make_loader(train_dataset, shuffle=True)
    test_loader = make_loader(test_dataset, shuffle=False)

    seeds = [1, 2, 3, 4, 5]

    rows = []

    for seed in seeds:
        print(f"\nRunning seed {seed}")

        set_seed(seed)

        obs_model = train_observation_model(train_loader)
        action_model = train_action_model(train_loader)

        obs_accuracy, obs_precision, obs_recall = evaluate_bottleneck(
            obs_model,
            test_loader,
            model_type="obs",
        )

        action_accuracy, action_precision, action_recall = evaluate_bottleneck(
            action_model,
            test_loader,
            model_type="action",
        )

        rows.append({
            "seed": seed,
            "obs_accuracy": obs_accuracy,
            "action_accuracy": action_accuracy,
            "obs_precision": obs_precision,
            "action_precision": action_precision,
            "obs_recall": obs_recall,
            "action_recall": action_recall,
            "accuracy_improvement": action_accuracy - obs_accuracy,
            "recall_improvement": action_recall - obs_recall,
        })

        print("Obs accuracy:", obs_accuracy)
        print("Action accuracy:", action_accuracy)
        print("Obs recall:", obs_recall)
        print("Action recall:", action_recall)

    df = pd.DataFrame(rows)

    print("\nRESULT TABLE")
    print(df)

    df.to_csv(
        "data/processed/multiseed_bottleneck_anticipation.csv",
        index=False,
    )

    acc_t, acc_p = ttest_rel(
        df["action_accuracy"],
        df["obs_accuracy"],
    )

    recall_t, recall_p = ttest_rel(
        df["action_recall"],
        df["obs_recall"],
    )

    print("\nSTATISTICAL TESTS")

    print("Accuracy improvement mean:", df["accuracy_improvement"].mean())
    print("Accuracy p-value:", acc_p)

    print("Recall improvement mean:", df["recall_improvement"].mean())
    print("Recall p-value:", recall_p)

    print("\nSaved to data/processed/multiseed_bottleneck_anticipation.csv")


if __name__ == "__main__":
    main()