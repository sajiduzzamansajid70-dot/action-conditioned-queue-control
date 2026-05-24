import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset


OBS_COLS = [
    "obs_q_s1",
    "obs_q_s2",
    "obs_q_s3a",
    "obs_q_s3b",
    "obs_q_s4",

    "obs_m_s1",
    "obs_m_s2",
    "obs_m_s3a",
    "obs_m_s3b",
    "obs_m_s4",
]

ACTION_COL = "action"

TARGET_COLS = [
    "next_q_s1",
    "next_q_s2",
    "next_q_s3a",
    "next_q_s3b",
    "next_q_s4",

    "next_m_s1",
    "next_m_s2",
    "next_m_s3a",
    "next_m_s3b",
    "next_m_s4",
]

KPI_COLS = [
    "queue_saturation_score",
    "bottleneck_binary",
]


class QueueSequenceDataset(Dataset):

    def __init__(
        self,
        csv_path,
        seq_len=20,
        scaler=None,
        fit_scaler=False,
    ):

        self.df = pd.read_csv(csv_path)

        self.seq_len = seq_len

        features = self.df[OBS_COLS].values.astype(np.float32)

        if scaler is None:
            scaler = StandardScaler()

        if fit_scaler:
            features = scaler.fit_transform(features)
        else:
            features = scaler.transform(features)

        self.scaler = scaler

        self.features = features

        self.actions = self.df[ACTION_COL].values.astype(np.int64)

        self.targets = self.df[TARGET_COLS].values.astype(np.float32)

        self.kpis = self.df[KPI_COLS].values.astype(np.float32)

        self.valid_indices = self._make_valid_indices()

    def _make_valid_indices(self):

        valid = []

        episode_ids = self.df["episode_id"].values

        for i in range(self.seq_len, len(self.df)):

            same_episode = (
                episode_ids[i]
                ==
                episode_ids[i - self.seq_len]
            )

            if same_episode:
                valid.append(i)

        return valid

    def __len__(self):

        return len(self.valid_indices)

    def __getitem__(self, idx):

        end_idx = self.valid_indices[idx]

        start_idx = end_idx - self.seq_len

        x_obs = self.features[start_idx:end_idx]

        x_action = self.actions[start_idx:end_idx]

        y_next = self.targets[end_idx]

        y_kpi = self.kpis[end_idx]

        return {
            "x_obs": torch.tensor(x_obs, dtype=torch.float32),

            "x_action": torch.tensor(
                x_action,
                dtype=torch.long,
            ),

            "y_next": torch.tensor(
                y_next,
                dtype=torch.float32,
            ),

            "y_kpi": torch.tensor(
                y_kpi,
                dtype=torch.float32,
            ),
        }