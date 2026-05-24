import torch
import torch.nn as nn


class ActionConditionedNoEmbeddingLSTM(nn.Module):

    def __init__(
        self,
        obs_dim=10,
        action_dim=1,
        hidden_dim=64,
        num_layers=1,
        next_state_dim=10,
        kpi_dim=2,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=obs_dim + action_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.next_state_head = nn.Linear(hidden_dim, next_state_dim)
        self.kpi_head = nn.Linear(hidden_dim, kpi_dim)

    def forward(self, x_obs, x_action):
        x_action = x_action.float().unsqueeze(-1)

        x = torch.cat([x_obs, x_action], dim=-1)

        lstm_out, _ = self.lstm(x)

        last_hidden = lstm_out[:, -1, :]

        y_next_pred = self.next_state_head(last_hidden)
        y_kpi_pred = self.kpi_head(last_hidden)

        return y_next_pred, y_kpi_pred