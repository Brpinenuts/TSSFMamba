import torch
import torch.nn as nn
from einops import rearrange
from mamba_ssm import Mamba


class SpatioSpectralMambaBlock(nn.Module):
    """空时谱融合Mamba模块"""

    def __init__(self, dim):
        super().__init__()
        # 空间处理分支
        self.spatial_mamba = Mamba(
            d_model=dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.spatial_norm = nn.LayerNorm(dim)

        # 光谱处理分支
        self.spectral_mamba = Mamba(
            d_model=dim,
            d_state=8,
            d_conv=3,
            expand=1
        )
        self.spectral_norm = nn.LayerNorm(dim)

        # 时序处理分支
        self.temporal_mamba = Mamba(
            d_model=dim,
            d_state=12,
            d_conv=3,
            expand=1
        )
        self.temporal_norm = nn.LayerNorm(dim)

        # 跨维度交互
        self.cross_attn = nn.MultiheadAttention(dim, num_heads=4, batch_first=True)

        # 动态门控融合
        self.fusion_gate = nn.Sequential(
            nn.Linear(3 * dim, dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        """输入: (B, T, C, H, W)"""
        B, T, C, H, W = x.shape

        # 空间特征处理
        spatial_feat = rearrange(x, 'b t c h w -> (b t c) h w')
        spatial_feat = self.spatial_norm(spatial_feat)
        spatial_feat = rearrange(spatial_feat, '(b t c) h w -> b (t c) (h w)', b=B, t=T, c=C)
        spatial_feat = self.spatial_mamba(spatial_feat)

        # 光谱特征处理
        spectral_feat = rearrange(x, 'b t c h w -> (b h w) t c')
        spectral_feat = self.spectral_norm(spectral_feat)
        spectral_feat = self.spectral_mamba(spectral_feat)
        spectral_feat = rearrange(spectral_feat, '(b h w) t c -> b t c h w', h=H, w=W)

        # 时序特征处理
        temporal_feat = rearrange(x, 'b t c h w -> (b c h w) t')
        temporal_feat = self.temporal_norm(temporal_feat)
        temporal_feat = self.temporal_mamba(temporal_feat)
        temporal_feat = rearrange(temporal_feat, '(b c h w) t -> b t c h w', c=C, h=H, w=W)

        # 跨维度交互
        combined = torch.stack([spatial_feat, spectral_feat, temporal_feat], dim=2)  # [B, T, 3, C, H, W]
        attn_output, _ = self.cross_attn(
            combined.view(B * T, 3, C * H * W),
            combined.view(B * T, 3, C * H * W),
            combined.view(B * T, 3, C * H * W)
        )
        attn_output = attn_output.view(B, T, 3, C, H, W)

        # 动态门控融合
        gate = self.fusion_gate(torch.cat([
            spatial_feat.mean(dim=[3, 4]),
            spectral_feat.mean(dim=[2, 3, 4]),
            temporal_feat.mean(dim=[1, 3, 4])
        ], dim=-1))

        output = gate.unsqueeze(-1).unsqueeze(-1) * attn_output[:, :, 0] + \
                 (1 - gate.unsqueeze(-1).unsqueeze(-1)) * attn_output[:, :, 1]

        return output

