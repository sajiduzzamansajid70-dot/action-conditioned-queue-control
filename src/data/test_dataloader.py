from torch.utils.data import DataLoader

from dataset import QueueSequenceDataset


def make_loader(dataset, batch_size=32, shuffle=False):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def print_batch_shapes(name, loader):
    batch = next(iter(loader))

    print(f"\n{name}")
    print("x_obs shape:", batch["x_obs"].shape)
    print("x_action shape:", batch["x_action"].shape)
    print("y_next shape:", batch["y_next"].shape)
    print("y_kpi shape:", batch["y_kpi"].shape)


def main():
    train_dataset = QueueSequenceDataset("data/raw/train.csv", seq_len=20, fit_scaler=True)

    validation_dataset = QueueSequenceDataset(
        "data/raw/validation.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    test_normal_dataset = QueueSequenceDataset(
        "data/raw/test_normal.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    test_high_load_dataset = QueueSequenceDataset(
        "data/raw/test_high_load.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    test_high_failure_dataset = QueueSequenceDataset(
        "data/raw/test_high_failure.csv",
        seq_len=20,
        scaler=train_dataset.scaler,
        fit_scaler=False,
    )

    loaders = {
        "TRAIN BATCH": make_loader(train_dataset, shuffle=True),
        "VALIDATION BATCH": make_loader(validation_dataset),
        "TEST NORMAL BATCH": make_loader(test_normal_dataset),
        "TEST HIGH LOAD BATCH": make_loader(test_high_load_dataset),
        "TEST HIGH FAILURE BATCH": make_loader(test_high_failure_dataset),
    }

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(validation_dataset))
    print("Test normal dataset size:", len(test_normal_dataset))
    print("Test high load dataset size:", len(test_high_load_dataset))
    print("Test high failure dataset size:", len(test_high_failure_dataset))

    for name, loader in loaders.items():
        print_batch_shapes(name, loader)


if __name__ == "__main__":
    main()