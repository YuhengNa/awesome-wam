"""Predictive feature VAE for frozen DINO/SVG patch-token videos.

The model operates on feature clips with shape `[B, V, F, N, D]`, where
`V` is the number of camera views, `F = 1 + T_future` is the number of frames,
`N` is the spatial patch-token count, and `D` is the frozen teacher feature
dimension. The current frame is encoded as its own latent token grid, while
future frames are compressed in fixed-size temporal groups. The latent sequence
length is therefore `1 + T_future / temporal_compression`: the `1` is the
current frame, and only future frames are temporally compressed. The decoder
reconstructs a complete clip from an observed prefix plus learned pad latents.
"""

from __future__ import annotations

import dataclasses

import torch
from torch import nn
import torch.nn.functional as F


@dataclasses.dataclass(frozen=True)
class PredictiveFeatureVAEConfig:
    feature_dim: int = 384
    model_dim: int = 768
    latent_dim: int = 128
    temporal_compression: int = 4
    num_encoder_layers: int = 8
    num_decoder_layers: int = 8
    num_heads: int = 12
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    max_views: int = 4
    max_frames: int = 32
    max_tokens: int = 1024
    max_groups: int = 16
    kl_weight: float = 0.0
    cosine_weight: float = 0.1
    delta_weight: float = 0.5
    future_loss_weight: float = 1.0
    future_weight_ramp: float = 0.0


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class FactorizedSpatioTemporalBlock(nn.Module):
    """Spatial attention inside each frame, then temporal attention per patch."""

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        self.spatial_norm = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.temporal_norm = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.mlp_norm = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_views, num_frames, num_tokens, dim = x.shape

        spatial = x.reshape(batch_size * num_views * num_frames, num_tokens, dim)
        spatial_norm = self.spatial_norm(spatial)
        spatial = spatial + self.spatial_attn(spatial_norm, spatial_norm, spatial_norm, need_weights=False)[0]
        x = spatial.reshape(batch_size, num_views, num_frames, num_tokens, dim)

        temporal = x.permute(0, 1, 3, 2, 4).reshape(batch_size * num_views * num_tokens, num_frames, dim)
        temporal_norm = self.temporal_norm(temporal)
        temporal = temporal + self.temporal_attn(temporal_norm, temporal_norm, temporal_norm, need_weights=False)[0]
        x = temporal.reshape(batch_size, num_views, num_tokens, num_frames, dim).permute(0, 1, 3, 2, 4)

        x = x + self.mlp(self.mlp_norm(x))
        return x


class TemporalAttentionPool(nn.Module):
    """Pool a fixed-size temporal group into one latent token per spatial patch."""

    def __init__(self, dim: int, heads: int, dropout: float):
        super().__init__()
        self.query = nn.Parameter(torch.zeros(1, 1, dim))
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, V, G, S, N, H]
        batch_size, num_views, num_groups, group_size, num_tokens, dim = x.shape
        group_tokens = x.permute(0, 1, 2, 4, 3, 5).reshape(
            batch_size * num_views * num_groups * num_tokens,
            group_size,
            dim,
        )
        group_tokens = self.norm(group_tokens)
        query = self.query.expand(group_tokens.shape[0], -1, -1)
        pooled = self.attn(query, group_tokens, group_tokens, need_weights=False)[0][:, 0]
        return pooled.view(batch_size, num_views, num_groups, num_tokens, dim)


class PredictiveFeatureVAE(nn.Module):
    """Temporal predictive VAE over frozen visual feature grids."""

    def __init__(self, config: PredictiveFeatureVAEConfig):
        super().__init__()
        self.config = config

        self.input_proj = nn.Linear(config.feature_dim, config.model_dim)
        self.encoder_blocks = nn.ModuleList(
            [
                FactorizedSpatioTemporalBlock(
                    config.model_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.num_encoder_layers)
            ]
        )
        self.encoder_norm = nn.LayerNorm(config.model_dim)
        self.temporal_pool = TemporalAttentionPool(config.model_dim, config.num_heads, config.dropout)
        self.to_mu_logvar = nn.Linear(config.model_dim, 2 * config.latent_dim)

        self.latent_proj = nn.Linear(config.latent_dim, config.model_dim)
        self.pad_latent = nn.Parameter(torch.zeros(1, 1, 1, 1, config.latent_dim))
        self.decoder_blocks = nn.ModuleList(
            [
                FactorizedSpatioTemporalBlock(
                    config.model_dim,
                    config.num_heads,
                    config.mlp_ratio,
                    config.dropout,
                )
                for _ in range(config.num_decoder_layers)
            ]
        )
        self.decoder_norm = nn.LayerNorm(config.model_dim)
        self.temporal_expand = nn.Linear(config.model_dim, config.temporal_compression * config.model_dim)
        self.output_proj = nn.Linear(config.model_dim, config.feature_dim)

        self.view_embed = nn.Parameter(torch.zeros(config.max_views, config.model_dim))
        self.frame_embed = nn.Parameter(torch.zeros(config.max_frames, config.model_dim))
        self.group_embed = nn.Parameter(torch.zeros(config.max_groups, config.model_dim))
        self.spatial_embed = nn.Parameter(torch.zeros(config.max_tokens, config.model_dim))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        for param in (self.pad_latent, self.view_embed, self.frame_embed, self.group_embed, self.spatial_embed):
            nn.init.normal_(param, std=0.02)

    def _check_shape(self, features: torch.Tensor) -> tuple[int, int, int, int, int, int]:
        if features.ndim != 5:
            raise ValueError(f"Expected features [B,V,T,N,D], got {tuple(features.shape)}.")
        batch_size, num_views, num_frames, num_tokens, dim = features.shape
        group_size = self.config.temporal_compression
        if dim != self.config.feature_dim:
            raise ValueError(f"feature dim {dim} does not match config {self.config.feature_dim}.")
        num_future_frames = num_frames - 1
        if num_future_frames < 0 or num_future_frames % group_size != 0:
            raise ValueError(
                f"future frames={num_future_frames} must be divisible by "
                f"temporal_compression={group_size}."
            )
        if num_views > self.config.max_views:
            raise ValueError(f"num_views={num_views} exceeds max_views={self.config.max_views}.")
        if num_frames > self.config.max_frames:
            raise ValueError(f"num_frames={num_frames} exceeds max_frames={self.config.max_frames}.")
        if num_tokens > self.config.max_tokens:
            raise ValueError(f"num_tokens={num_tokens} exceeds max_tokens={self.config.max_tokens}.")
        num_future_groups = num_future_frames // group_size
        total_groups = 1 + num_future_groups
        if total_groups > self.config.max_groups:
            raise ValueError(f"total_groups={total_groups} exceeds max_groups={self.config.max_groups}.")
        return batch_size, num_views, num_frames, num_tokens, dim, total_groups

    def _add_frame_pos(self, hidden: torch.Tensor) -> torch.Tensor:
        _, num_views, num_frames, num_tokens, _ = hidden.shape
        view = self.view_embed[:num_views].view(1, num_views, 1, 1, -1)
        frame = self.frame_embed[:num_frames].view(1, 1, num_frames, 1, -1)
        spatial = self.spatial_embed[:num_tokens].view(1, 1, 1, num_tokens, -1)
        return hidden + view + frame + spatial

    def _add_group_pos(self, hidden: torch.Tensor) -> torch.Tensor:
        _, num_views, num_groups, num_tokens, _ = hidden.shape
        view = self.view_embed[:num_views].view(1, num_views, 1, 1, -1)
        group = self.group_embed[:num_groups].view(1, 1, num_groups, 1, -1)
        spatial = self.spatial_embed[:num_tokens].view(1, 1, 1, num_tokens, -1)
        return hidden + view + group + spatial

    def encode_observed(self, features: torch.Tensor, observed_groups: int) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_views, num_frames, num_tokens, _, total_groups = self._check_shape(features)
        group_size = self.config.temporal_compression
        if not 1 <= observed_groups <= total_groups:
            raise ValueError(f"observed_groups must be in [1, {total_groups}], got {observed_groups}.")

        observed_future_groups = observed_groups - 1
        observed_frames = 1 + observed_future_groups * group_size
        hidden = self.input_proj(features[:, :, :observed_frames])
        hidden = self._add_frame_pos(hidden)
        for block in self.encoder_blocks:
            hidden = block(hidden)
        hidden = self.encoder_norm(hidden)

        current_group = hidden[:, :, :1].view(batch_size, num_views, 1, 1, num_tokens, self.config.model_dim)
        pooled_groups = [self.temporal_pool(current_group)]
        if observed_future_groups > 0:
            future_hidden = hidden[:, :, 1:].view(
                batch_size,
                num_views,
                observed_future_groups,
                group_size,
                num_tokens,
                self.config.model_dim,
            )
            pooled_groups.append(self.temporal_pool(future_hidden))
        pooled = torch.cat(pooled_groups, dim=2)
        mu, logvar = self.to_mu_logvar(pooled).chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        if not self.training or self.config.kl_weight == 0:
            return mu
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def decode_full(self, z_observed: torch.Tensor, total_groups: int) -> torch.Tensor:
        batch_size, num_views, observed_groups, num_tokens, _ = z_observed.shape
        if not 1 <= observed_groups <= total_groups:
            raise ValueError(f"observed_groups must be in [1, {total_groups}], got {observed_groups}.")
        if total_groups > self.config.max_groups:
            raise ValueError(f"total_groups={total_groups} exceeds max_groups={self.config.max_groups}.")

        if observed_groups < total_groups:
            pad = self.pad_latent.expand(batch_size, num_views, total_groups - observed_groups, num_tokens, -1)
            z_full = torch.cat([z_observed, pad], dim=2)
        else:
            z_full = z_observed

        hidden = self.latent_proj(z_full)
        hidden = self._add_group_pos(hidden)
        for block in self.decoder_blocks:
            hidden = block(hidden)
        hidden = self.decoder_norm(hidden)

        current_hidden = hidden[:, :, :1]
        current_hidden = current_hidden + self.frame_embed[:1].view(1, 1, 1, 1, self.config.model_dim)
        current_pred = self.output_proj(current_hidden)

        future_group_hidden = hidden[:, :, 1:]
        if future_group_hidden.shape[2] == 0:
            return current_pred

        expanded = self.temporal_expand(future_group_hidden)
        group_size = self.config.temporal_compression
        num_future_groups = future_group_hidden.shape[2]
        expanded = expanded.view(
            batch_size,
            num_views,
            num_future_groups,
            num_tokens,
            group_size,
            self.config.model_dim,
        )
        expanded = expanded.permute(0, 1, 2, 4, 3, 5).reshape(
            batch_size,
            num_views,
            num_future_groups * group_size,
            num_tokens,
            self.config.model_dim,
        )
        expanded = expanded + self.frame_embed[1 : 1 + num_future_groups * group_size].view(
            1,
            1,
            -1,
            1,
            self.config.model_dim,
        )
        future_pred = self.output_proj(expanded)
        return torch.cat([current_pred, future_pred], dim=2)

    def forward(self, features: torch.Tensor, observed_groups: int) -> dict[str, torch.Tensor]:
        *_, total_groups = self._check_shape(features)
        mu, logvar = self.encode_observed(features, observed_groups)
        z_observed = self.reparameterize(mu, logvar)
        pred = self.decode_full(z_observed, total_groups)
        return {
            "pred": pred,
            "z": z_observed,
            "mu": mu,
            "logvar": logvar,
        }

    def compute_loss(self, features: torch.Tensor, observed_groups: int) -> dict[str, torch.Tensor]:
        *_, total_groups = self._check_shape(features)
        group_size = self.config.temporal_compression
        observed_frames = 1 + (observed_groups - 1) * group_size
        outputs = self(features, observed_groups)
        pred = outputs["pred"]
        frame_weights = torch.ones(features.shape[2], device=features.device, dtype=features.dtype)
        if observed_frames < features.shape[2]:
            future_len = features.shape[2] - observed_frames
            future_weights = torch.full(
                (future_len,),
                self.config.future_loss_weight,
                device=features.device,
                dtype=features.dtype,
            )
            if self.config.future_weight_ramp != 0 and future_len > 1:
                ramp = torch.linspace(0.0, 1.0, future_len, device=features.device, dtype=features.dtype)
                future_weights = future_weights * (1.0 + self.config.future_weight_ramp * ramp)
            frame_weights[observed_frames:] = future_weights
        frame_weights = frame_weights.view(1, 1, -1, 1, 1)

        squared_error = (pred - features).pow(2)
        recon_loss = (squared_error * frame_weights).sum() / frame_weights.expand_as(squared_error).sum().clamp_min(1.0)
        cosine_distance = 1.0 - F.cosine_similarity(pred.float(), features.float(), dim=-1)
        cosine_weights = frame_weights[..., 0].float()
        cosine_loss = (cosine_distance * cosine_weights).sum() / cosine_weights.expand_as(cosine_distance).sum().clamp_min(1.0)

        pred_delta = pred[:, :, 1:] - pred[:, :, :-1]
        target_delta = features[:, :, 1:] - features[:, :, :-1]
        delta_loss = F.mse_loss(pred_delta, target_delta)
        kl_loss = -0.5 * (1.0 + outputs["logvar"] - outputs["mu"].pow(2) - outputs["logvar"].exp()).mean()
        loss = (
            recon_loss
            + self.config.cosine_weight * cosine_loss
            + self.config.delta_weight * delta_loss
            + self.config.kl_weight * kl_loss
        )
        observed_mse = squared_error[:, :, :observed_frames].mean()
        if observed_frames < features.shape[2]:
            future_squared_error = squared_error[:, :, observed_frames:]
            future_mse = future_squared_error.mean()
            future_mse_by_frame = future_squared_error.mean(dim=(0, 1, 3, 4))
        else:
            future_mse = squared_error.new_zeros(())
            future_mse_by_frame = squared_error.new_zeros((0,))
        pred_delta_norm = pred_delta.float().norm(dim=-1).mean()
        target_delta_norm = target_delta.float().norm(dim=-1).mean()
        delta_ratio = pred_delta_norm / target_delta_norm.clamp_min(1e-6)
        pred_delta_norm_by_frame = pred_delta.float().norm(dim=-1).mean(dim=(0, 1, 3))
        target_delta_norm_by_frame = target_delta.float().norm(dim=-1).mean(dim=(0, 1, 3))
        delta_ratio_by_frame = pred_delta_norm_by_frame / target_delta_norm_by_frame.clamp_min(1e-6)
        if observed_frames < features.shape[2]:
            static_future = features[:, :, observed_frames - 1 : observed_frames].expand_as(features[:, :, observed_frames:])
            static_squared_error = (static_future - features[:, :, observed_frames:]).pow(2)
            static_future_mse = static_squared_error.mean()
            static_future_mse_by_frame = static_squared_error.mean(dim=(0, 1, 3, 4))
            future_copy_ratio_by_frame = future_mse_by_frame / static_future_mse_by_frame.clamp_min(1e-6)
        else:
            static_future_mse = squared_error.new_zeros(())
            static_future_mse_by_frame = squared_error.new_zeros((0,))
            future_copy_ratio_by_frame = squared_error.new_zeros((0,))
        return {
            **outputs,
            "loss": loss,
            "recon_loss": recon_loss,
            "cosine_loss": cosine_loss,
            "delta_loss": delta_loss,
            "kl_loss": kl_loss,
            "observed_mse": observed_mse,
            "future_mse": future_mse,
            "future_mse_by_frame": future_mse_by_frame,
            "pred_delta_norm": pred_delta_norm,
            "target_delta_norm": target_delta_norm,
            "delta_ratio": delta_ratio,
            "pred_delta_norm_by_frame": pred_delta_norm_by_frame,
            "target_delta_norm_by_frame": target_delta_norm_by_frame,
            "delta_ratio_by_frame": delta_ratio_by_frame,
            "static_future_mse": static_future_mse,
            "static_future_mse_by_frame": static_future_mse_by_frame,
            "future_copy_ratio_by_frame": future_copy_ratio_by_frame,
            "observed_frames": torch.as_tensor(observed_frames, device=features.device),
            "total_groups": torch.as_tensor(total_groups, device=features.device),
        }
