"""Feature-space latent action model.

This module follows the DreamDojo LAM pattern but predicts frozen visual
features instead of RGB patches:

    feature_t, feature_t+k -> z_action -> reconstruct feature_t+k
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class FeatureLatentActionConfig:
    feature_dim: int = 384
    model_dim: int = 512
    latent_dim: int = 32
    num_encoder_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_views: int = 4
    kl_weight: float = 1e-6


def _make_transformer_layer(config: FeatureLatentActionConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.model_dim,
        nhead=config.num_heads,
        dim_feedforward=int(config.model_dim * config.mlp_ratio),
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class FeatureLatentActionModel(nn.Module):
    """Continuous latent-action VAE for frozen patch-token features.

    Inputs are expected as `[B, V, N, D]`, where `V` is camera/view count,
    `N` is patch-token count per image, and `D` is the frozen feature dim.
    """

    def __init__(self, config: FeatureLatentActionConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.feature_dim, config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.feature_dim)
        self.action_prompt = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        self.time_embed = nn.Parameter(torch.zeros(2, config.model_dim))
        self.view_embed = nn.Parameter(torch.zeros(config.max_views, config.model_dim))

        self.encoder = nn.TransformerEncoder(
            _make_transformer_layer(config),
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.to_mu_logvar = nn.Linear(config.model_dim, 2 * config.latent_dim)

        self.latent_proj = nn.Linear(config.latent_dim, config.model_dim)
        self.decoder = nn.TransformerEncoder(
            _make_transformer_layer(config),
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.action_prompt, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.view_embed, std=0.02)

    def _add_view_time_embed(self, tokens: torch.Tensor, *, time_index: int) -> torch.Tensor:
        batch_size, num_views, num_tokens, _ = tokens.shape
        if num_views > self.config.max_views:
            raise ValueError(f"num_views={num_views} exceeds max_views={self.config.max_views}.")
        view_embed = self.view_embed[:num_views].view(1, num_views, 1, -1)
        time_embed = self.time_embed[time_index].view(1, 1, 1, -1)
        return tokens + view_embed + time_embed

    def encode(self, current_features: torch.Tensor, future_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        current = self._add_view_time_embed(self.input_proj(current_features), time_index=0)
        future = self._add_view_time_embed(self.input_proj(future_features), time_index=1)

        batch_size, num_views, _, _ = current.shape
        current = current.flatten(1, 2)
        future = future.flatten(1, 2)
        action_prompt = self.action_prompt.expand(batch_size, num_views, -1)
        action_prompt = action_prompt + self.view_embed[:num_views].view(1, num_views, -1)

        hidden = torch.cat([current, action_prompt, future], dim=1)
        hidden = self.encoder(hidden)
        prompt_start = current.shape[1]
        prompt_hidden = hidden[:, prompt_start : prompt_start + num_views].mean(dim=1)
        mu, logvar = self.to_mu_logvar(prompt_hidden).chunk(2, dim=-1)
        return mu, logvar

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not torch.is_grad_enabled():
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode(self, current_features: torch.Tensor, latent_action: torch.Tensor) -> torch.Tensor:
        hidden = self._add_view_time_embed(self.input_proj(current_features), time_index=1)
        latent = self.latent_proj(latent_action).view(latent_action.shape[0], 1, 1, -1)
        hidden = hidden + latent

        batch_size, num_views, num_tokens, _ = hidden.shape
        hidden = self.decoder(hidden.flatten(1, 2))
        pred = self.output_proj(hidden).view(batch_size, num_views, num_tokens, self.config.feature_dim)
        return pred

    def forward(self, current_features: torch.Tensor, future_features: torch.Tensor) -> dict[str, torch.Tensor]:
        mu, logvar = self.encode(current_features, future_features)
        z = self.reparameterize(mu, logvar)
        pred = self.decode(current_features, z)
        return {
            "pred": pred,
            "z": z,
            "mu": mu,
            "logvar": logvar,
        }

    def compute_loss(self, current_features: torch.Tensor, future_features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self(current_features, future_features)
        recon_loss = F.mse_loss(outputs["pred"], future_features)
        kl_loss = -0.5 * (1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()).mean()
        loss = recon_loss + self.config.kl_weight * kl_loss
        return {
            **outputs,
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
        }
