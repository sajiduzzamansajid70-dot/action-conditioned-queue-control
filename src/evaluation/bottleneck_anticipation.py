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


QUEUE_CAPACITY = 2.5


def make_loader(dataset):
    return DataLoader(dataset, batch_size=32, shuffle=False)


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

            pred_bottleneck = (pred_q >= QUEUE_CAPACITY).any(dim=1)
            true_bottleneck = (true_q >= QUEUE_CAPACITY).any(dim=1)

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
    print("BOTTLENECK ANTICIPATION ANALYSIS")

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

        accuracy, precision, recall = evaluate_bottleneck(
            model=model,
            loader=loader,
            model_type=model_type,
        )

        rows.append({
            "model": name,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
        })

    df = pd.DataFrame(rows)

    print("\nBOTTLENECK ANTICIPATION TABLE\n")
    print(df)

    df.to_csv(
        "data/processed/bottleneck_anticipation.csv",
        index=False,
    )

    print("\nSaved to data/processed/bottleneck_anticipation.csv")


if __name__ == "__main__":
    main()