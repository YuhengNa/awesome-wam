"""Per-frame semantic feature VAE for visual tokenizer experiments.

The model compresses teacher feature tokens independently for each frame:

    [B, V, N, D] -> z [B, V, N, d] -> reconstruction [B, V, N, D]

Training scripts can apply it to videos by flattening the frame axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass
class SemanticFeatureVAEConfig:
    feature_dim: int = 384
    model_dim: int = 512
    latent_dim: int = 96
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_views: int = 4
    max_tokens_per_view: int = 1024
    kl_weight: float = 1e-6
    cosine_weight: float = 0.1


class SemanticFeatureVAE(nn.Module):
    """Frame-wise semantic feature compressor.

    This is intentionally a simple token-level VAE baseline. It does not model
    temporal dynamics; that job belongs to PV-VAE and DeltaTok variants.
    """

    def __init__(self, config: SemanticFeatureVAEConfig):
        super().__init__()
        self.config = config

        ff_dim = int(config.model_dim * config.mlp_ratio)
        self.input_proj = nn.Linear(config.feature_dim, config.model_dim)
        self.view_embed = nn.Parameter(torch.zeros(config.max_views, config.model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(config.max_tokens_per_view, config.model_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.num_encoder_layers)
        self.to_mu_logvar = nn.Linear(config.model_dim, config.latent_dim * 2)

        self.latent_proj = nn.Linear(config.latent_dim, config.model_dim)
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=config.model_dim,
            nhead=config.num_heads,
            dim_feedforward=ff_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=config.num_decoder_layers)
        self.output_proj = nn.Linear(config.model_dim, config.feature_dim)

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.trunc_normal_(self.view_embed, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.to_mu_logvar.weight)
        nn.init.zeros_(self.to_mu_logvar.bias)
        nn.init.xavier_uniform_(self.latent_proj.weight)
        nn.init.zeros_(self.latent_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _check_features(self, features: torch.Tensor) -> tuple[int, int, int, int]:
        if features.ndim != 4:
            raise ValueError(f"Expected features [B,V,N,D], got shape {tuple(features.shape)}")
        batch, views, tokens, dim = features.shape
        if dim != self.config.feature_dim:
            raise ValueError(f"Expected feature_dim={self.config.feature_dim}, got {dim}")
        if views > self.config.max_views:
            raise ValueError(f"views={views} exceeds max_views={self.config.max_views}")
        if tokens > self.config.max_tokens_per_view:
            raise ValueError(
                f"tokens_per_view={tokens} exceeds max_tokens_per_view={self.config.max_tokens_per_view}"
            )
        return batch, views, tokens, dim

    def encode(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, views, tokens, _ = self._check_features(features)
        x = self.input_proj(features)
        x = x + self.view_embed[:views].view(1, views, 1, -1)
        x = x + self.pos_embed[:tokens].view(1, 1, tokens, -1)
        x = x.reshape(batch, views * tokens, self.config.model_dim)
        x = self.encoder(x)
        stats = self.to_mu_logvar(x).reshape(batch, views, tokens, self.config.latent_dim * 2)
        mu, logvar = stats.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-20.0, max=10.0)
        if self.training and self.config.kl_weight > 0:
            std = torch.exp(0.5 * logvar)
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        return z, mu, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        if z.ndim != 4:
            raise ValueError(f"Expected latent tokens [B,V,N,d], got shape {tuple(z.shape)}")
        batch, views, tokens, dim = z.shape
        if dim != self.config.latent_dim:
            raise ValueError(f"Expected latent_dim={self.config.latent_dim}, got {dim}")
        if views > self.config.max_views or tokens > self.config.max_tokens_per_view:
            raise ValueError(f"Latent shape {tuple(z.shape)} exceeds configured view/token limits")

        x = self.latent_proj(z)
        x = x + self.view_embed[:views].view(1, views, 1, -1)
        x = x + self.pos_embed[:tokens].view(1, 1, tokens, -1)
        x = x.reshape(batch, views * tokens, self.config.model_dim)
        x = self.decoder(x)
        x = self.output_proj(x)
        return x.reshape(batch, views, tokens, self.config.feature_dim)

    def forward(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        z, mu, logvar = self.encode(features)
        pred = self.decode(z)
        return {"pred": pred, "z": z, "mu": mu, "logvar": logvar}

    def compute_loss(self, features: torch.Tensor) -> dict[str, torch.Tensor]:
        out = self(features)
        pred = out["pred"]
        recon_loss = F.mse_loss(pred, features)
        cosine_loss = 1.0 - F.cosine_similarity(pred, features, dim=-1).mean()
        mu = out["mu"]
        logvar = out["logvar"]
        kl_loss = -0.5 * (1.0 + logvar - mu.pow(2) - logvar.exp()).mean()
        loss = recon_loss + self.config.cosine_weight * cosine_loss + self.config.kl_weight * kl_loss
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "kl_loss": kl_loss.detach(),
            "latent_norm": out["z"].detach().norm(dim=-1).mean(),
            "target_norm": features.detach().norm(dim=-1).mean(),
            "pred": pred.detach(),
            "z": out["z"].detach(),
        }
