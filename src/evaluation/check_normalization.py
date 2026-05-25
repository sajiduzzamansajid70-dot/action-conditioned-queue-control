import sys
from pathlib import Path

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR / "src" / "data"))

from dataset import QueueSequenceDataset


def summarize(name, dataset):
    features = dataset.features

    print(f"\n{name}")
    print("Feature mean:", np.mean(features))
    print("Feature std:", np.std(features))
    print("Feature min:", np.min(features))
    print("Feature max:", np.max(features))


def main():
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

    summarize("TRAIN", train_dataset)
    summarize("VALIDATION", validation_dataset)
    summarize("TEST NORMAL", test_normal_dataset)
    summarize("TEST HIGH LOAD", test_high_load_dataset)
    summarize("TEST HIGH FAILURE", test_high_failure_dataset)


if __name__ == "__main__":
    main()