import torch
import torch.nn as nn


class ObservationOnlyGRU(nn.Module):

    def __init__(
        self,
        obs_dim=10,
        hidden_dim=64,
        num_layers=1,
        next_state_dim=10,
        kpi_dim=2,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=obs_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )

        self.next_state_head = nn.Linear(hidden_dim, next_state_dim)
        self.kpi_head = nn.Linear(hidden_dim, kpi_dim)

    def forward(self, x_obs):
        gru_out, _ = self.gru(x_obs)

        last_hidden = gru_out[:, -1, :]

        y_next_pred = self.next_state_head(last_hidden)
        y_kpi_pred = self.kpi_head(last_hidden)

        return y_next_pred, y_kpi_pred