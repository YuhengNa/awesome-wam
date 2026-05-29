"""DeltaTok-style transition tokenizer for frozen visual features.

The model encodes a pair of feature maps into one continuous delta token and
decodes the next feature map from the current feature map plus that token:

    feature_t, feature_t+k -> z_delta -> reconstruct feature_t+k

Unlike the DreamDojo-style LAM in ``latent_action.py``, this module is
deterministic and defaults to keeping the delta token at the teacher feature
dimension.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn


@dataclasses.dataclass(frozen=True)
class FeatureDeltaTokenizerConfig:
    feature_dim: int = 384
    model_dim: int = 384
    token_dim: int = 0
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    num_heads: int = 8
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_views: int = 4
    max_tokens_per_view: int = 1024


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


class FeatureDeltaTokenizerModel(nn.Module):
    """One-token transition autoencoder over frozen patch-token features.

    Inputs are expected as ``[B, V, N, D]``: batch, views, patch tokens, feature
    dimension. The model emits one global delta token per sample.
    """

    def __init__(self, config: FeatureDeltaTokenizerConfig):
        super().__init__()
        self.config = config
        token_dim = config.token_dim if config.token_dim > 0 else config.model_dim
        self.token_dim = token_dim

        self.input_proj = nn.Linear(config.feature_dim, config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.feature_dim)
        self.z_embed = nn.Parameter(torch.zeros(1, 1, config.model_dim))
        self.time_embed = nn.Parameter(torch.zeros(2, config.model_dim))
        self.view_embed = nn.Parameter(torch.zeros(config.max_views, config.model_dim))
        self.pos_embed = nn.Parameter(torch.zeros(config.max_tokens_per_view, config.model_dim))

        self.encoder = nn.TransformerEncoder(
            _make_transformer_layer(config),
            num_layers=config.num_encoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )
        self.to_token = nn.Linear(config.model_dim, token_dim)
        self.token_norm = nn.LayerNorm(token_dim)
        self.token_to_model = nn.Linear(token_dim, config.model_dim)
        self.decoder = nn.TransformerEncoder(
            _make_transformer_layer(config),
            num_layers=config.num_decoder_layers,
            norm=nn.LayerNorm(config.model_dim),
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.z_embed, std=0.02)
        nn.init.normal_(self.time_embed, std=0.02)
        nn.init.normal_(self.view_embed, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)

    def _project_features(self, features: torch.Tensor, *, time_index: int) -> torch.Tensor:
        batch_size, num_views, num_tokens, _ = features.shape
        if num_views > self.config.max_views:
            raise ValueError(f"num_views={num_views} exceeds max_views={self.config.max_views}.")
        if num_tokens > self.config.max_tokens_per_view:
            raise ValueError(
                f"num_tokens={num_tokens} exceeds max_tokens_per_view={self.config.max_tokens_per_view}."
            )

        hidden = self.input_proj(features)
        hidden = hidden + self.time_embed[time_index].view(1, 1, 1, -1)
        hidden = hidden + self.view_embed[:num_views].view(1, num_views, 1, -1)
        hidden = hidden + self.pos_embed[:num_tokens].view(1, 1, num_tokens, -1)
        return hidden.flatten(1, 2)

    def encode(self, current_features: torch.Tensor, future_features: torch.Tensor) -> torch.Tensor:
        current = self._project_features(current_features, time_index=0)
        future = self._project_features(future_features, time_index=1)
        z = self.z_embed.expand(current.shape[0], -1, -1)
        hidden = torch.cat([z, current, future], dim=1)
        hidden = self.encoder(hidden)
        return self.token_norm(self.to_token(hidden[:, 0]))

    def decode(self, current_features: torch.Tensor, delta_token: torch.Tensor) -> torch.Tensor:
        current = self._project_features(current_features, time_index=0)
        z = self.token_to_model(delta_token).unsqueeze(1)
        hidden = torch.cat([z, current], dim=1)
        hidden = self.decoder(hidden)[:, 1:]

        batch_size, num_views, num_tokens, _ = current_features.shape
        pred = self.output_proj(hidden).view(batch_size, num_views, num_tokens, self.config.feature_dim)
        return pred

    def forward(self, current_features: torch.Tensor, future_features: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encode(current_features, future_features)
        pred = self.decode(current_features, z)
        return {"pred": pred, "z": z}
