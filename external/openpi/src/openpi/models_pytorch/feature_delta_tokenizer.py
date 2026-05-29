"""Feature-space DeltaTok tokenizer for robot world-model experiments.

The tokenizer encodes a semantic transition instead of a full state:

    encode(x_t, x_t+k) -> z_delta [B, M, d]
    decode(x_t, z_delta) -> x_hat_t+k [B, V, N, D]

Here x_t and x_t+k are frozen teacher patch features, for example SVG-P,
DINOv3, or SigLIP features. Static scene information can flow through x_t, so
the bottleneck is encouraged to represent the change.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class FeatureDeltaTokenizerConfig:
    feature_dim: int = 384
    model_dim: int = 384
    token_dim: int = 0
    num_delta_tokens: int = 1
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_views: int = 4
    max_tokens_per_view: int = 1024
    decode_residual: bool = True
    cosine_weight: float = 0.0


def _make_transformer_layer(config: FeatureDeltaTokenizerConfig) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d_model=config.model_dim,
        nhead=config.num_heads,
        dim_feedforward=int(config.model_dim * config.mlp_ratio),
        dropout=config.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )


class FeatureDeltaTokenizer(nn.Module):
    """Multi-token deterministic transition autoencoder over patch features."""

    def __init__(self, config: FeatureDeltaTokenizerConfig):
        super().__init__()
        if config.num_delta_tokens < 1:
            raise ValueError("num_delta_tokens must be >= 1")
        self.config = config
        self.token_dim = config.token_dim if config.token_dim > 0 else config.model_dim

        self.input_proj = nn.Linear(config.feature_dim, config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.feature_dim)
        self.time_embed = nn.Parameter(torch.zeros(2, config.model_dim))
        self.view_embed = nn.Parameter(torch.zeros(config.max_views, config.model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(config.max_tokens_per_view, config.model_dim))
        self.delta_query = nn.Parameter(torch.zeros(config.num_delta_tokens, config.model_dim))
        self.delta_decode_embed = nn.Parameter(torch.zeros(config.num_delta_tokens, config.model_dim))

        self.encoder = nn.TransformerEncoder(
            _make_transformer_layer(config),
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.to_token = nn.Linear(config.model_dim, self.token_dim)
        self.token_norm = nn.LayerNorm(self.token_dim)
        self.token_to_model = nn.Linear(self.token_dim, config.model_dim)
        self.decoder = nn.TransformerEncoder(
            _make_transformer_layer(config),
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.view_embed, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        nn.init.normal_(self.delta_query, std=0.02)
        nn.init.normal_(self.delta_decode_embed, std=0.02)
        nn.init.xavier_uniform_(self.input_proj.weight)
        nn.init.zeros_(self.input_proj.bias)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def _check_features(self, features: torch.Tensor) -> tuple[int, int, int, int]:
        if features.ndim != 4:
            raise ValueError(f"Expected features [B,V,N,D], got {tuple(features.shape)}")
        batch, views, tokens, dim = features.shape
        if dim != self.config.feature_dim:
            raise ValueError(f"Expected feature_dim={self.config.feature_dim}, got {dim}")
        if views > self.config.max_views:
            raise ValueError(f"views={views} exceeds max_views={self.config.max_views}")
        if tokens > self.config.max_tokens_per_view:
            raise ValueError(f"tokens={tokens} exceeds max_tokens_per_view={self.config.max_tokens_per_view}")
        return batch, views, tokens, dim

    def _project_features(self, features: torch.Tensor, *, time_index: int) -> torch.Tensor:
        batch, views, tokens, _ = self._check_features(features)
        hidden = self.input_proj(features)
        hidden = hidden + self.time_embed[time_index].view(1, 1, 1, -1)
        hidden = hidden + self.view_embed[:views].view(1, views, 1, -1)
        hidden = hidden + self.pos_embed[:tokens].view(1, 1, tokens, -1)
        return hidden.reshape(batch, views * tokens, self.config.model_dim)

    def encode(self, current_features: torch.Tensor, future_features: torch.Tensor) -> torch.Tensor:
        current = self._project_features(current_features, time_index=0)
        future = self._project_features(future_features, time_index=1)
        query = self.delta_query.unsqueeze(0).expand(current.shape[0], -1, -1)
        hidden = torch.cat([query, current, future], dim=1)
        hidden = self.encoder(hidden)
        delta_tokens = self.to_token(hidden[:, : self.config.num_delta_tokens])
        return self.token_norm(delta_tokens)

    def decode(self, current_features: torch.Tensor, delta_tokens: torch.Tensor) -> torch.Tensor:
        if delta_tokens.ndim != 3:
            raise ValueError(f"Expected delta tokens [B,M,d], got {tuple(delta_tokens.shape)}")
        batch, num_delta_tokens, token_dim = delta_tokens.shape
        if num_delta_tokens != self.config.num_delta_tokens:
            raise ValueError(f"Expected M={self.config.num_delta_tokens}, got {num_delta_tokens}")
        if token_dim != self.token_dim:
            raise ValueError(f"Expected token_dim={self.token_dim}, got {token_dim}")

        current = self._project_features(current_features, time_index=0)
        delta = self.token_to_model(delta_tokens)
        delta = delta + self.delta_decode_embed.unsqueeze(0)
        hidden = torch.cat([delta, current], dim=1)
        hidden = self.decoder(hidden)[:, num_delta_tokens:]

        batch_size, views, tokens, _ = current_features.shape
        decoded = self.output_proj(hidden).view(batch_size, views, tokens, self.config.feature_dim)
        if self.config.decode_residual:
            return current_features + decoded
        return decoded

    def forward(self, current_features: torch.Tensor, future_features: torch.Tensor) -> dict[str, torch.Tensor]:
        z_delta = self.encode(current_features, future_features)
        pred = self.decode(current_features, z_delta)
        return {"pred": pred, "z_delta": z_delta}

    def compute_loss(self, current_features: torch.Tensor, future_features: torch.Tensor) -> dict[str, torch.Tensor]:
        outputs = self(current_features, future_features)
        pred = outputs["pred"]
        recon_loss = F.mse_loss(pred.float(), future_features.detach().float())
        cosine_loss = 1.0 - F.cosine_similarity(pred.float(), future_features.detach().float(), dim=-1).mean()
        loss = recon_loss + self.config.cosine_weight * cosine_loss
        with torch.no_grad():
            copy_mse = F.mse_loss(current_features.float(), future_features.float())
            target_delta = future_features.float() - current_features.float()
            pred_delta = pred.float() - current_features.float()
            delta_ratio = pred_delta.norm() / target_delta.norm().clamp_min(1e-6)
            token_norm = outputs["z_delta"].float().norm(dim=-1).mean()
            delta_norm = target_delta.norm(dim=-1).mean()
        return {
            "loss": loss,
            "recon_loss": recon_loss.detach(),
            "cosine_loss": cosine_loss.detach(),
            "copy_mse": copy_mse.detach(),
            "delta_ratio": delta_ratio.detach(),
            "token_norm": token_norm.detach(),
            "target_delta_norm": delta_norm.detach(),
            "pred": pred.detach(),
            "z_delta": outputs["z_delta"].detach(),
        }
