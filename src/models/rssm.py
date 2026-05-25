import torch
import torch.nn as nn
import torch.nn.functional as F


class RSSM(nn.Module):

    def __init__(
        self,
        obs_dim=10,
        action_dim=8,
        action_embed_dim=8,
        hidden_dim=128,
        latent_dim=32,
        next_state_dim=10,
        kpi_dim=2,
    ):
        super().__init__()

        self.action_embedding = nn.Embedding(
            num_embeddings=action_dim,
            embedding_dim=action_embed_dim,
        )

        self.gru = nn.GRUCell(
            input_size=latent_dim + action_embed_dim,
            hidden_size=hidden_dim,
        )

        self.posterior_net = nn.Sequential(
            nn.Linear(hidden_dim + obs_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )

        self.prior_net = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2),
        )

        self.state_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, next_state_dim),
        )

        self.kpi_decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, kpi_dim),
        )

    def sample_gaussian(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self,
        x_obs,
        x_action,
    ):
        batch_size, seq_len, _ = x_obs.shape

        hidden = torch.zeros(
            batch_size,
            128,
            device=x_obs.device,
        )

        latent = torch.zeros(
            batch_size,
            32,
            device=x_obs.device,
        )

        kl_loss = 0.0

        for t in range(seq_len):

            action_emb = self.action_embedding(
                x_action[:, t]
            )

            gru_input = torch.cat(
                [latent, action_emb],
                dim=-1,
            )

            hidden = self.gru(
                gru_input,
                hidden,
            )

            posterior_input = torch.cat(
                [hidden, x_obs[:, t]],
                dim=-1,
            )

            posterior_stats = self.posterior_net(
                posterior_input
            )

            post_mean, post_logvar = torch.chunk(
                posterior_stats,
                2,
                dim=-1,
            )

            prior_stats = self.prior_net(hidden)

            prior_mean, prior_logvar = torch.chunk(
                prior_stats,
                2,
                dim=-1,
            )

            latent = self.sample_gaussian(
                post_mean,
                post_logvar,
            )

            kl = -0.5 * torch.sum(
                1
                + post_logvar
                - prior_logvar
                - (
                    (
                        post_mean - prior_mean
                    ) ** 2
                    + torch.exp(post_logvar)
                )
                / torch.exp(prior_logvar),
                dim=-1,
            )

            kl_loss += kl.mean()

        decoder_input = torch.cat(
            [hidden, latent],
            dim=-1,
        )

        next_state_pred = self.state_decoder(
            decoder_input
        )

        kpi_pred = self.kpi_decoder(
            decoder_input
        )

        kl_loss = kl_loss / seq_len

        return (
            next_state_pred,
            kpi_pred,
            kl_loss,
        )