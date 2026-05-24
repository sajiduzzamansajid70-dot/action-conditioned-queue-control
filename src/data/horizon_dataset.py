import torch

from dataset import QueueSequenceDataset


class HorizonQueueDataset(QueueSequenceDataset):

    def __init__(
        self,
        csv_path,
        seq_len=20,
        horizon=1,
        scaler=None,
        fit_scaler=False,
    ):
        self.horizon = horizon

        super().__init__(
            csv_path=csv_path,
            seq_len=seq_len,
            scaler=scaler,
            fit_scaler=fit_scaler,
        )

    def _make_valid_indices(self):
        valid = []

        episode_ids = self.df["episode_id"].values

        max_index = len(self.df) - self.horizon

        for i in range(self.seq_len, max_index):
            same_start_episode = (
                episode_ids[i] == episode_ids[i - self.seq_len]
            )

            same_target_episode = (
                episode_ids[i] == episode_ids[i + self.horizon]
            )

            if same_start_episode and same_target_episode:
                valid.append(i)

        return valid

    def __getitem__(self, idx):
        end_idx = self.valid_indices[idx]
        start_idx = end_idx - self.seq_len

        target_idx = end_idx + self.horizon

        x_obs = self.features[start_idx:end_idx]
        x_action = self.actions[start_idx:end_idx]

        y_next = self.targets[target_idx]
        y_kpi = self.kpis[target_idx]

        return {
            "x_obs": torch.tensor(x_obs, dtype=torch.float32),
            "x_action": torch.tensor(x_action, dtype=torch.long),
            "y_next": torch.tensor(y_next, dtype=torch.float32),
            "y_kpi": torch.tensor(y_kpi, dtype=torch.float32),
        }