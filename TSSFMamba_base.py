import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score, average_precision_score
from tqdm import tqdm  # 新增tqdm导入
import os
from cnnlstm import *
from RX import *
import matplotlib.pyplot as plt
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import Resize
import cv2
from config import config
from roc_plot1 import *
import time
import random
from models.videomamba import VisionMamba
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
def preprocess_data(data, time_steps=5, stride=3):
    """改进的时序数据预处理"""
    # 原始数据形状: (N_frames, C, H, W) = (500, 25, 216, 409)

    # Step 1: 调整每帧尺寸到256x256
    resized_data = []
    for frame in data:
        frame_tensor = torch.from_numpy(frame).unsqueeze(0)  # (1, C, H, W)
        torch_resize = Resize([256, 256])
        resized_frame = torch_resize(frame_tensor)
        # resized_frame = F.interpolate(frame_tensor, size=(256, 256),
        #                               mode='bilinear', align_corners=False)
        resized_data.append(resized_frame.squeeze(0).numpy())
    resized_data = np.array(resized_data)  # (500, 25, 256, 256)

    # Step 2: 构建时序样本（带步长的滑动窗口）
    num_frames = resized_data.shape[0]
    num_samples = (num_frames - time_steps) // stride + 1
    indices = np.arange(0, num_samples * stride, stride)

    time_series = np.stack([resized_data[i:i + time_steps]
                            for i in indices])  # (N_samples, T, C, H, W)

    # Step 3: 通道独立标准化
    # 计算每个通道的均值和标准差
    mean = np.mean(time_series, axis=(0, 1, 3, 4), keepdims=True)  # (1,1,C,1,1)
    std = np.std(time_series, axis=(0, 1, 3, 4), keepdims=True) + 1e-8


    norm_series = (time_series - mean) / std
    return norm_series, mean, std, indices


class TemporalDataset(Dataset):
    def __init__(self, time_series, indices):
        """
        time_series: (N_samples, T, C, H, W)
        indices: 每个样本对应的起始帧索引
        """
        self.data = time_series
        self.indices = indices

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.from_numpy(self.data[idx]),  # (T,C,H,W)
            self.indices[idx]  # 起始帧索引
        )


def inverse_transform(model, loader, device, original_shape, mean, std, original_data):
    """改进的逆变换函数，精确对齐光流场"""
    N_frames, C, H, W = original_shape
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_matrix = np.zeros(original_shape, dtype=np.float32)


    # 标准化参数转换
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, indices in tqdm(loader, desc="Inverse Transform"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # 前向传播
            outputs, _ = model(inputs)

            # === 重建数据处理 ===
            outputs = outputs * std_tensor + mean_tensor
            outputs = F.interpolate(
                outputs.view(-1, C, 256, 256),
                size=(H, W),
                mode='bilinear',
                align_corners=False
            ).view(batch_size, -1, C, H, W)

            # 累加重建结果
            batch_pred = outputs.cpu().numpy()
            batch_indices = indices.numpy()
            for i in range(batch_size):
                start = batch_indices[i]
                end = start + batch_pred.shape[1]
                reconstructed[start:end] += batch_pred[i]
                count_matrix[start:end] += 1

    data_tensor = torch.from_numpy((reconstructed - original_data)).float().to(device=device)
    num_frames, num_channels, height, width = data_tensor.shape
    scores_maps = []


    for i in tqdm(range(num_frames), desc="Post Processing"):
        frame = data_tensor[i]  # (C, H, W)
        pixels = frame.permute(1, 2, 0).reshape(-1, num_channels)  # (H*W, C)

        # 计算均值和协方差矩阵
        mu = torch.mean(pixels, dim=0)  # (C,)
        pixels_centered = pixels - mu  # (H*W, C)
        cov = torch.matmul(pixels_centered.T, pixels_centered) / (pixels.shape[0] - 1)  # (C, C)
        cov_reg = cov + torch.eye(num_channels, device=device) * 1e-6
        inv_cov = torch.linalg.inv(cov_reg)  # (C, C)

        # 计算马氏距离 (批量计算)
        diff = pixels_centered  # (H*W, C)
        scores = torch.einsum('ni,ij,nj->n', diff, inv_cov, diff)  # (H*W,)
        scores = scores.reshape(height, width).cpu().numpy()  # 转回CPU处理后续步骤

        # 后处理 (可选)
        from scipy.ndimage import gaussian_filter
        scores = gaussian_filter(scores, sigma=0.4)


        scores_maps.append(scores)
    arrays = [np.array(arr) for arr in scores_maps]
    data = np.stack(arrays, axis=0)

    return reconstructed, data


# VisionMamba模块工厂函数
def create_mamba(TIME_STEPS):
    return VisionMamba(
        group_type=config.group_type,
        k_group=config.k_group,
        embed_dim=config.embed_dim,
        dt_rank=config.dt_rank,
        d_inner=config.dim_inner,
        d_state=config.d_state,
        num_classes=config.num_classes,
        depth=config.depth,
        scan_type=config.scan_type,
        pos=config.pos,
        cls=config.cls,
        conv3D_channel=config.conv3D_channel,
        conv3D_kernel=config.conv3D_kernel,
        dim_patch=config.dim_patch,
        dim_linear=config.dim_linear,
        TIME_STEPS=TIME_STEPS
    )


class FastChannelAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.conv = nn.Conv1d(channels, channels // 8, 1)
        self.relu = nn.ReLU()
        self.gate = nn.Conv1d(channels // 8, channels, 1)

    def forward(self, x):
        y = self.pool(x)
        y = self.conv(y)
        return x * torch.sigmoid(self.gate(self.relu(y)))

class OptimizedSpatialTemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels=64, reduction=8):
        super().__init__()

        # 空间卷积：深度可分离+通道压缩
        self.space_conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=6,
                      stride=4, padding=1, groups=in_channels),  # 深度卷积
            nn.Conv2d(in_channels, out_channels // 2, 1),  # 逐点卷积
            nn.BatchNorm2d(out_channels // 2),
            nn.GELU()
        )

        # self.n1 = nn.Conv2d(in_channels, in_channels, kernel_size=6,
        #               stride=4, padding=1, groups=in_channels)
        # self.n2 = nn.Conv2d(in_channels, out_channels // 2, 1)
        # self.n3 =nn.BatchNorm2d(out_channels // 2)
        # self.n4 =nn.GELU()


        # 时间卷积：分组可分离+通道扩展
        self.time_conv = nn.Sequential(
            nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=5,
                      padding=2, groups=out_channels // 2),  # 分组深度卷积
            nn.Conv1d(out_channels // 2, out_channels, 1),  # 逐点扩展
            nn.BatchNorm1d(out_channels),
            # FastChannelAttention(out_channels)
        )

        # self.nn1 = nn.Conv1d(out_channels // 2, out_channels // 2, kernel_size=5,
        #               padding=2, groups=out_channels // 2)
        # self.nn2 = nn.Conv1d(out_channels // 2, out_channels, 1)
        # self.nn3 = nn.BatchNorm1d(out_channels)
        # self.nn4 = FastChannelAttention(out_channels)

    def forward(self, x):
        """输入输出维度: [B, C, T, H, W]"""
        B, C, T, H, W = x.shape

        # 空间维度处理 ----------------------------------------------
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(-1, C, H, W)  # [B*T, C, H, W]
        x = self.space_conv(x)  # [B*T, C//2, H', W']
        # x = self.n1(x)
        # x = self.n2(x)
        # x = self.n3(x)
        # x = self.n4(x)
        # 重组维度 ------------------------------------------------
        _, C_new, H_new, W_new = x.shape
        x = x.view(B, T, C_new, H_new, W_new)
        x = x.permute(0, 2, 1, 3, 4)  # [B, C//2, T, H', W']

        # 时间维度处理 ----------------------------------------------
        x = x.permute(0, 3, 4, 1, 2)  # [B, H', W', C//2, T]
        x = x.reshape(-1, C_new, T)  # [B*H'W', C//2, T]

        x = self.time_conv(x)  # [B*H'W', C, T]

        # x = self.nn1(x)
        # x = self.nn2(x)
        # x = self.nn3(x)
        # x = self.nn4(x)

        # 最终维度重组 ---------------------------------------------
        x = x.view(B, H_new, W_new, -1, T)  # [B, H', W', C, T]
        x = x.permute(0, 3, 4, 1, 2)


        return x  # [B, C, T, H', W']


class OptimizedDecomposedConv3D(nn.Module):
    def __init__(self, in_channels=64, out_channels=128, reduction_ratio=8):
        super().__init__()

        # 空间卷积：深度可分离卷积+通道缩减
        self.space_conv = nn.Sequential(
            # 深度卷积
            nn.Conv2d(in_channels, in_channels, kernel_size=3,
                      stride=2, padding=1, groups=in_channels),
            # 通道压缩
            nn.Conv2d(in_channels, out_channels // 4, kernel_size=1),
            nn.BatchNorm2d(out_channels // 4),
            nn.GELU()
        )

        # 时间卷积：分组可分离卷积
        self.time_conv = nn.Sequential(
            # 深度卷积
            nn.Conv1d(out_channels // 4, out_channels // 4, kernel_size=3,
                      padding=1, groups=out_channels // 4),
            # 通道扩展
            nn.Conv1d(out_channels // 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.GELU(),
            # FastChannelAttention(out_channels)
        )


    def forward(self, x):
        """输入维度: [B, C, T, H, W]"""
        B, C, T, H, W = x.shape

        # 空间维度处理 --------------------------------------------------
        x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        x = x.reshape(-1, C, H, W)  # [B*T, C, H, W]
        x = self.space_conv(x)  # [B*T, C//4, H', W']

        # 恢复维度
        _, C_new, H_new, W_new = x.shape
        x = x.view(B, T, C_new, H_new, W_new)  # [B, T, C//4, H', W']
        x = x.permute(0, 2, 1, 3, 4)  # [B, C//4, T, H', W']

        # 时间维度处理 --------------------------------------------------
        x = x.permute(0, 3, 4, 1, 2)  # [B, H', W', C//4, T]
        x = x.reshape(-1, C_new, T)  # [B*H'W', C//4, T]
        x = self.time_conv(x)  # [B*H'W', C, T]

        # 维度恢复与注意力增强 ------------------------------------------
        x = x.view(B, H_new, W_new, -1, T)  # [B, H', W', C, T]
        x = x.permute(0, 3, 4, 1, 2)  # [B, C, T, H', W']

        return x

class OptimizedSpatialTemporalDeconv(nn.Module):
    def __init__(self, in_channels=128, out_channels=64):
        super().__init__()

        # 时间反卷积：分组可分离+通道压缩
        self.time_deconv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3,
                               stride=1, padding=1, groups=in_channels),
            nn.Conv1d(in_channels, out_channels * 2, 1),
            nn.BatchNorm1d(out_channels * 2),
            nn.GELU(),
            # FastChannelAttention(out_channels * 2),

        )

        # 空间反卷积：深度可分离+通道扩展
        self.space_deconv = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(out_channels * 2, out_channels * 2, kernel_size=3,
                          padding=1, groups=out_channels * 2)
            ),
            # nn.ConvTranspose2d(out_channels * 2, out_channels * 2, kernel_size=3,
            #                    stride=2, padding=1, output_padding=1, groups=out_channels * 2),
            nn.Conv2d(out_channels * 2, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.GELU()
        )

        # self.SpectralTemporalAttention = SpectralTemporalAttention(out_channels)
    def forward(self, x):
        """输入输出维度: [B, C, T, H, W]"""
        B, C, T, H, W = x.shape

        # ========== 时间维度处理 ==========
        # 合并空间维度到batch维度 [B*H*W, C, T]
        x = x.permute(0, 3, 4, 1, 2).contiguous().view(-1, C, T)
        x = self.time_deconv(x)  # [B*H*W, C*2, T]

        # ========== 联合维度重组 ==========
        # 直接重组为4D张量 [B, C*2, T_new, H, W]
        x = x.contiguous().view(B, H, W, -1, T).permute(0, 3, 4, 1, 2)  # 单次维度变换
        C_new = x.shape[1]

        # ========== 空间维度处理 ==========
        # 合并时间维度到batch维度 [B*T_new, C_new, H, W]
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(-1, C_new, H, W)
        x = self.space_deconv(x)  # [B*T_new, C_out, H', W']

        # ========== 最终输出重组 ==========
        # 单次维度重组 [B, C_out, T_new, H', W']
        x = x.view(B, -1, x.shape[1], x.shape[2], x.shape[3]).permute(0, 2, 1, 3, 4)
        # x = self.SpectralTemporalAttention(x)
        return x


class FinalReconstructionDeconv(nn.Module):
    def __init__(self, in_channels=64, out_channels=25, reduction=8):
        super().__init__()

        # 时间维度处理
        self.time_deconv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels * 2, kernel_size=5,
                               stride=1, padding=2, groups=in_channels),
            nn.Conv1d(in_channels * 2, in_channels * 4, 1),
            nn.BatchNorm1d(in_channels * 4),
            nn.GELU(),
            # FastChannelAttention(in_channels * 4),
        )

        # 空间维度处理
        self.space_deconv = nn.Sequential(
            nn.Sequential(
                nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels * 4, in_channels * 4, kernel_size=5,
                          padding=2, groups=in_channels * 4)
            ),
            # nn.ConvTranspose2d(in_channels * 4, in_channels * 4, kernel_size=6,
            #                    stride=4, padding=1, output_padding=0, groups=in_channels * 4),
            nn.Conv2d(in_channels * 4, out_channels, 1),
            nn.Tanh()
        )


    def forward(self, x):
        """输入维度: [B, C, T, H, W]"""
        B, C, T, H, W = x.shape

        # 时间维度上采样 --------------------------------------------
        x = x.permute(0, 3, 4, 1, 2) # [B, H, W, C, T]
        x = x.reshape(-1, C, T)  # [B*H*W, C, T]
        x = self.time_deconv(x)  # [B*H*W, C*4, T]

        # 重组维度 ------------------------------------------------
        _, C_new, T_new = x.shape
        x = x.contiguous().view(B, H, W, C_new, T_new)
        x = x.permute(0, 3, 4, 1, 2)  # [B, C*4, T_new, H, W]

        # 空间维度上采样 --------------------------------------------
        x = x.permute(0, 2, 1, 3, 4)  # [B, T_new, C*4, H, W]
        x = x.reshape(-1, C_new, H, W)  # [B*T_new, C*4, H, W]
        x = self.space_deconv(x)  # [B*T_new, C_out, H', W']

        # 残差增强 ------------------------------------------------
        _, C_out, H_new, W_new = x.shape
        x = x.contiguous().view(B, T_new, C_out, H_new, W_new)
        x = x.permute(0, 2, 1, 3, 4)  # [B, C_out, T_new, H', W']

        return x


class WaveletConv(nn.Module):
    def __init__(self, in_channels, wavelet_type='haar', temporal_window=3):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = self._init_wavelet_kernel(wavelet_type)
        self.temporal_window = temporal_window

        # 可学习子带权重
        self.subband_weights = nn.Parameter(torch.tensor([1.2, 0.2, 0.2, 0.2], dtype=torch.float32))

        # 时空分离卷积
        self.ll_processor = nn.Sequential(
            # 空间特征提取
            nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels),
            # 通道注意力
            ChannelAttention(in_channels),
            # 时间特征聚合
            TemporalAggregation(in_channels, temporal_window)
        )

        # 高频抑制模块
        self.hf_suppress = HFSuppress(in_channels * 3)

        # 融合卷积
        # self.post_conv = nn.Sequential(
        #     nn.Conv2d(4 * in_channels, in_channels, 3, padding=1),
        #     nn.ReLU(),
        #     nn.Conv2d(2 * in_channels, in_channels, 3, padding=1)
        # )

        self.post_conv = nn.Conv2d(
            in_channels=in_channels * 4,  # 小波分解后通道数
            out_channels=in_channels,  # 输出通道数恢复为原始值
            kernel_size=3,
            padding=1,
            groups=in_channels  # 分组数等于输入通道数
        )

    def _init_wavelet_kernel(self, wavelet_type):
        if wavelet_type == 'haar':
            LL = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32) / 2.0
            LH = torch.tensor([[[[-1, -1], [1, 1]]]], dtype=torch.float32) / 2.0
            HL = torch.tensor([[[[-1, 1], [-1, 1]]]], dtype=torch.float32) / 2.0
            HH = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32) / 2.0
            return torch.cat([LL, LH, HL, HH], dim=0)  # shape [4,1,2,2]
        else:
            raise NotImplementedError

    def forward(self, x):
        b, c, h, w = x.shape

        # 小波分解
        wavelet_filters = self.wavelet.repeat(c, 1, 1, 1).to(x.device)
        coeffs = F.conv2d(x, wavelet_filters, stride=2, groups=c)  # [b,c*4,h//2,w//2]

        # 子带分离与加权
        coeffs = coeffs.view(b, c, 4, h // 2, w // 2)
        coeffs = coeffs * self.subband_weights.view(1, 1, 4, 1, 1)
        ll = coeffs[:, :, 0, :, :]  # 低频 [b,c,h//2,w//2]
        hf = coeffs[:, :, 1:, :, :].flatten(1, 2)  # 高频合并 [b,c*3,h//2,w//2]

        # 低频时空处理
        # ll_enhanced = self.ll_processor(ll)
        ll_enhanced = ll
        # 高频抑制
        hf_suppressed = self.hf_suppress(hf)

        # 特征重组
        coeffs = torch.cat([
            ll_enhanced.unsqueeze(2),
            hf_suppressed.view(b, c, 3, h // 2, w // 2)
        ], dim=2).flatten(1, 2)  # [b,c*4,h//2,w//2]

        # 上采样重建
        coeffs = F.interpolate(coeffs, size=(h, w), mode='bicubic', align_corners=False)

        return self.post_conv(coeffs)


# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * (avg_out + max_out).view(b, c, 1, 1)


# 时间聚合模块
class TemporalAggregation(nn.Module):
    def __init__(self, channels, window_size):
        super().__init__()
        self.window_size = window_size
        self.conv = nn.Conv1d(
            in_channels=channels,  # 输入通道数
            out_channels=channels,  # 输出通道数
            kernel_size=3,
            padding=1,
            groups=channels  # 保持通道独立性
        )
        self.attn = nn.Sequential(
            nn.Linear(window_size, window_size * 2),
            nn.ReLU(),
            nn.Linear(window_size * 2, window_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 输入x形状: [b, c, h, w]
        b, c, h, w = x.shape

        # 将空间维度合并为序列长度
        x_seq = x.view(b, c, -1)  # [b, c, h*w]

        # 时间维度滑动窗口处理
        pad = (self.window_size - 1) // 2
        padded = F.pad(x_seq, (0, 0, pad, pad), mode='replicate')  # 时间维度padding

        # 构建时间窗口序列
        windows = []
        for t in range(self.window_size):
            window = padded[:, :, t:t + b]  # [b, c, h*w]
            windows.append(window)
        temporal_seq = torch.stack(windows, dim=1)  # [b, window_size, c, h*w]

        # 时间卷积处理
        temporal_seq = temporal_seq.permute(0, 2, 1, 3)  # [b, c, window_size, h*w]
        temporal_feat = self.conv(temporal_seq.flatten(0, 1))  # [b*c, window_size, h*w]

        # 时间注意力
        attn_weights = self.attn(temporal_feat.mean(dim=-1))  # [b*c, window_size]
        output = (temporal_feat * attn_weights.unsqueeze(-1)).sum(dim=1)  # [b*c, h*w]

        # 恢复空间维度
        return output.view(b, c, h, w)  # [b, c, h, w]


# 高频抑制模块
class HFSuppress(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // 4),
            nn.ReLU(),
            nn.Linear(channels // 4, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        weights = self.fc(self.avg_pool(x).view(b, c)).view(b, c, 1, 1)
        return x * (1 - weights)  # 抑制高频


class DaubechiesWaveletConv(nn.Module):
    def __init__(self, in_channels, wavelet_order=4):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = self._init_db_kernel(wavelet_order)
        self.post_conv = nn.Conv2d(in_channels * 4, in_channels, 3, padding=1, groups=in_channels)

    def _init_db_kernel(self, order):
        # 修正后的滤波器生成逻辑
        if order == 4:  # db2小波
            low_pass = torch.tensor([
                0.4829629131445341,
                0.8365163037378077,
                0.2241438680420134,
                -0.1294095225512603
            ])
        elif order == 6:  # db3小波
            low_pass = torch.tensor([
                0.3326705529500825,
                0.8068915093110924,
                0.4598775021184914,
                -0.1350110200102546,
                -0.0854412738820267,
                0.0352262918857095
            ])
        else:
            raise NotImplementedError

        # 使用翻转函数替代切片操作
        flipped = torch.flip(low_pass, dims=[0])  # 关键修正点

        # 构建二维可分离滤波器
        LL = torch.outer(low_pass, low_pass).view(1, 1, *low_pass.shape, *low_pass.shape)
        LH = torch.outer(low_pass, (-1) ** torch.arange(len(low_pass)) * flipped)
        HL = LH.T
        HH = torch.outer((-1) ** torch.arange(len(low_pass)) * flipped,
                         (-1) ** torch.arange(len(low_pass)) * flipped)
        LL = LL.squeeze()
        # 调整滤波器尺寸适应卷积操作
        return torch.cat([
            LL.unsqueeze(0),
            LH.unsqueeze(0),
            HL.unsqueeze(0),
            HH.unsqueeze(0)
        ], dim=0).float()

    def forward(self, x):
        # 添加尺寸检查与padding处理
        b, c, h, w = x.shape
        kernel_size = self.wavelet.shape[-1]
        pad = (kernel_size - 2) // 2  # 自动计算padding

        filters = self.wavelet.repeat(c, 1, 1, 1).to(x.device)
        coeffs = F.conv2d(
            F.pad(x, (pad, pad, pad, pad), mode='reflect'),  # 反射填充避免边界效应
            filters,
            stride=2,
            groups=c
        )
        coeffs = F.interpolate(coeffs, scale_factor=2, mode='bilinear', align_corners=False)
        return self.post_conv(coeffs)


class WaveletConv1(nn.Module):
    def __init__(self, in_channels, wavelet_type='haar'):
        super().__init__()
        self.in_channels = in_channels
        self.wavelet = self._init_wavelet_kernel(wavelet_type)
        # self.subband_weights = nn.Parameter(torch.tensor([0.9, 0.1, 0.1, 0.1], dtype=torch.float32))
        self.subband_weights = torch.tensor([1, 1, 1, 1])
        self.post_conv = nn.Conv2d(
            in_channels=in_channels * 4,  # 小波分解后通道数
            out_channels=in_channels,  # 输出通道数恢复为原始值
            kernel_size=3,
            padding=1,
            groups=in_channels  # 分组数等于输入通道数
        )

    def _init_wavelet_kernel(self, wavelet_type):
        # 生成符合PyTorch卷积核格式的4D张量
        if wavelet_type == 'haar':
            LL = torch.tensor([[[[1, 1], [1, 1]]]], dtype=torch.float32) / 2.0
            LH = torch.tensor([[[[-1, -1], [1, 1]]]], dtype=torch.float32) / 2.0
            HL = torch.tensor([[[[-1, 1], [-1, 1]]]], dtype=torch.float32) / 2.0
            HH = torch.tensor([[[[1, -1], [-1, 1]]]], dtype=torch.float32) / 2.0
            return torch.cat([LL, LH, HL, HH], dim=0)  # shape [4,1,2,2]
        else:
            raise NotImplementedError

    def forward(self, x):
        b, c, h, w = x.shape
        assert c == self.in_channels, f"Input channels {c} != expected {self.in_channels}"

        # 扩展小波核匹配输入通道
        wavelet_filters = self.wavelet.repeat(c, 1, 1, 1).to(x.device)  # [c*4,1,2,2]

        # 执行多通道小波分解
        coeffs = F.conv2d(
            input=x,
            weight=wavelet_filters,
            stride=2,
            groups=c  # 关键：每个通道独立处理
        )
        # coeffs shape: [b, c*4, h//2, w//2]

        # 子带加权增强低频
        coeffs = coeffs.view(b, c, 4, h // 2, w // 2)
        coeffs = coeffs * self.subband_weights.view(1, 1, 4, 1, 1).to(x.device)
        coeffs = coeffs.view(b, c * 4, h // 2, w // 2)

        # 上采样恢复空间分辨率
        coeffs = F.interpolate(coeffs, scale_factor=2, mode='nearest')
        # coeffs shape: [b, c*4, h, w]

        # 跨子带特征融合
        return self.post_conv(coeffs)


class HybridDecomposition(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 空间域分解
        self.spatial_conv = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1)

        # 频域分解
        self.freq_conv = nn.Conv2d(in_channels, in_channels * 2, 3, padding=1, dilation=2)

        # 跨域交互
        self.cross_attention = nn.MultiheadAttention(in_channels, 4)
        self.norm = nn.LayerNorm(in_channels)

    def forward(self, x):
        # 空间分支
        spatial = self.spatial_conv(x)

        # 频域分支
        freq = torch.fft.rfft2(x, norm='ortho')
        freq = self.freq_conv(freq.real) + 1j * self.freq_conv(freq.imag)
        freq = torch.fft.irfft2(freq, s=x.shape[-2:], norm='ortho')

        # 跨域交互
        b, c, h, w = spatial.shape
        combined = (spatial + freq).view(b, c, h * w).permute(2, 0, 1)  # [h*w, b, c]
        attn_out, _ = self.cross_attention(combined, combined, combined)
        return self.norm(attn_out.permute(1, 2, 0).view(b, c, h, w))

class SpectralSpatialWavelet(nn.Module):
    def __init__(self, in_bands):
        super().__init__()
        self.wave_conv = WaveletConv1(in_bands)
        # self.wave_conv = DaubechiesWaveletConv(in_bands)
        self.SpectralTemporalAttention = SpectralTemporalAttention(in_bands)

    def forward(self, x):
        # x: [B, C, T, H, W]
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        x = x.flatten(0, 1)
        spatial_feat = self.wave_conv(x)
        spatial_feat = spatial_feat.view(B, T, C, H, W).permute(0, 2, 1, 3, 4)
        spatial_feat = self.SpectralTemporalAttention(spatial_feat)
        return spatial_feat

class MemoryModule(nn.Module):
    """记忆增强模块"""

    def __init__(self, mem_dim, feat_dim):
        super().__init__()
        self.mem_dim = mem_dim
        self.memory = nn.Parameter(torch.randn(mem_dim, feat_dim))  # 可学习记忆库
        self.feat_dim = feat_dim

    def forward(self, z):
        B, C, T, H, W = z.shape

        # x = x.permute(0, 2, 1, 3, 4)  # [B, T, C, H, W]
        # x = x.reshape(-1, C, H, W)
        z_flat = z.reshape(B, C, -1).permute(0, 2, 1)  # (B, N, C)

        # 计算相似度
        sim = torch.matmul(z_flat, self.memory.T)  # (B, N, M)
        att = F.softmax(sim, dim=-1)  # (B, N, M)

        # 记忆读取
        z_hat = torch.matmul(att, self.memory)  # (B, N, C)
        z_hat = z_hat.permute(0, 2, 1).view(B, C, T, H, W)

        # 残差连接
        return z + 0.3 * z_hat

# class SpectralSpatialAE(nn.Module):
#     """空-谱自编码器（无监督背景重建）"""
#
#     def __init__(self, in_channels=25, latent_dim=256, T=8):
#         super().__init__()
#
#         self.SpatialTemporalConv = OptimizedSpatialTemporalConv(in_channels, out_channels=64)
#         self.SpatialTemporalConv1 = OptimizedDecomposedConv3D(in_channels=64, out_channels=128)
#         self.SpatialTemporalConv2 = OptimizedDecomposedConv3D(in_channels=128, out_channels=latent_dim)
#
#         self.DecomposedTransposeConv3D1 = OptimizedSpatialTemporalDeconv(in_channels=latent_dim, out_channels=128)
#         self.DecomposedTransposeConv3D = OptimizedSpatialTemporalDeconv(in_channels=128, out_channels=64)
#         self.SpatioTemporalTransposeConv = FinalReconstructionDeconv(in_channels=64, out_channels=in_channels)
#
#         self.wavelet = nn.ModuleList([
#             SpectralSpatialWavelet(in_bands=128),
#             SpectralSpatialWavelet(in_bands=64),
#             SpectralSpatialWavelet(in_bands=in_channels)
#         ])
#
#         # self.memory = nn.ModuleList([
#         #     MemoryModule(mem_dim=64, feat_dim=128),
#         #     MemoryModule(mem_dim=64, feat_dim=64),
#         #     MemoryModule(mem_dim=64, feat_dim=in_channels)
#         # ])
#
#         # 编码器组件
#         self.encoders = nn.ModuleList([
#                 nn.Sequential(
#                     nn.Conv3d(in_channels, 64, (5, 6, 6), stride=(1, 4, 4), padding=(2, 1, 1)),
#                     # self.SpatialTemporalConv,
#                     nn.BatchNorm3d(64),
#                     nn.LeakyReLU(0.2)
#                 ),
#                 nn.Sequential(
#                     nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
#                     # self.SpatialTemporalConv1,
#                     nn.BatchNorm3d(128),
#                     nn.LeakyReLU(0.2)
#                 ),
#                 nn.Sequential(
#                     nn.Conv3d(128, 256, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
#                     # self.SpatialTemporalConv2,
#                     nn.BatchNorm3d(256),
#                     nn.LeakyReLU(0.2)
#                 )
#         ])
#
#         # 解码器组件
#         self.decoders = nn.ModuleList([
#             nn.Sequential(
#                 nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
#                 # self.DecomposedTransposeConv3D1,
#                 nn.BatchNorm3d(128),
#                 nn.LeakyReLU(0.2)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose3d(128, 64, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
#                 # self.DecomposedTransposeConv3D,
#                 nn.BatchNorm3d(64),
#                 nn.LeakyReLU(0.2)
#             ),
#             nn.Sequential(
#                 # self.SpatioTemporalTransposeConv,
#                 nn.ConvTranspose3d(64, in_channels, (5, 6, 6), stride=(1, 4, 4), padding=(2, 1, 1)),
#                 nn.Tanh()
#             )
#         ])
#
#         # Mamba中间层
#         self.mamba_layers = nn.ModuleList([create_mamba(T) for _ in range(5)])
#
#     def forward(self, x):
#         # 输入维度调整 [B,T,C,H,W] -> [B,C,T,H,W]
#         x = x.permute(0, 2, 1, 3, 4)
#         residual0 = x
#         # 编码过程
#         latent = x
#         residuals = []
#         for i in range(3):
#             latent = self.encoders[i](latent)
#             # latent = self.mamba_layers[i](latent)  # 使用前3个Mamba层
#             residuals.append(latent)
#         medi = latent
#         # 解码过程
#         for j in range(2):
#
#             # latent = self.mamba_layers[3 + j](latent) + residuals[:-1][1 - j]
#
#             latent = self.decoders[j](latent)
#             # latent = self.mamba_layers[3 + j](latent) + self.wavelet[j](residuals[:-1][1-j])  # 使用后2个Mamba层
#
#
#
#             # latent = self.decoders[1](latent)
#             # # latent = latent + self.wavelet[1](residuals[:-1][0])
#             # latent = self.mamba_layers[3 + j](latent) + self.wavelet[1](residuals[:-1][0])
#
#
#             # latent = self.mamba_layers[3 + j](latent) + residuals[:-1][0]
#
#
#             # latent = self.mamba_layers[3 + j](latent) + residuals[:-1][0]
#             # latent = self.mamba_layers[3 + j](latent) + self.memory[1](self.wavelet[1](residuals[:-1][0]))
#
#
#         recon = self.decoders[-1](latent)
#         # recon = recon + self.wavelet[-1](residual0)
#
#         # recon = recon + self.memory[-1](self.wavelet[-1](residual0))
#         # recon = recon + residual0
#         # 维度恢复 [B,C,T,H,W] -> [B,T,C,H,W]
#         return recon.permute(0, 2, 1, 3, 4), medi
# class SpectralSpatialAE(nn.Module):
#     """空-谱自编码器（无监督背景重建）"""
#
#     def __init__(self, in_channels=25, latent_dim=128, T=8):
#         super().__init__()
#
#         # 编码器
#         self.encoders = nn.ModuleList([
#             nn.Sequential(
#                 # ConvLSTM3DNetwork(in_channels, 32, (5, 3, 3), 1),
#                 nn.Conv3d(in_channels, 64, (5, 6, 6), stride=(1, 4, 4), padding=(2, 1, 1)),
#                 # self.SpatialTemporalConv,
#                 nn.BatchNorm3d(64),
#                 nn.LeakyReLU(0.2)
#             ),
#             nn.Sequential(
#                 nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
#                 # self.SpatialTemporalConv1,
#                 nn.BatchNorm3d(128),
#                 nn.LeakyReLU(0.2)
#             ),
#             nn.Sequential(
#                 nn.Conv3d(128, 256, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
#                 # self.SpatialTemporalConv2,
#                 nn.BatchNorm3d(256),
#                 nn.LeakyReLU(0.2)
#             )
#         ])
#
#         # 解码器组件
#         self.decoders = nn.ModuleList([
#             nn.Sequential(
#                 nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
#                 # self.DecomposedTransposeConv3D1,
#                 nn.BatchNorm3d(128),
#                 nn.LeakyReLU(0.2)
#             ),
#             nn.Sequential(
#                 nn.ConvTranspose3d(128, 64, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
#                 # self.DecomposedTransposeConv3D,
#                 nn.BatchNorm3d(64),
#                 nn.LeakyReLU(0.2)
#             ),
#             nn.Sequential(
#                 # self.SpatioTemporalTransposeConv,
#                 nn.ConvTranspose3d(64, in_channels, (5, 6, 6), stride=(1, 4, 4), padding=(2, 1, 1)),
#                 nn.BatchNorm3d(in_channels),
#                 nn.LeakyReLU(0.2),
#                 # nn.Tanh(),
#                 # ConvLSTM3DNetwork(in_channels, 32, (5, 3, 3), 1)
#             )
#         ])
#         # self.wavelet = nn.ModuleList([
#         #     SpectralSpatialWavelet(in_bands=128),
#         #     SpectralSpatialWavelet(in_bands=64),
#         #     SpectralSpatialWavelet(in_bands=in_channels)
#         # ])
#
#         # Mamba中间层
#         # self.mamba_layers = nn.ModuleList([create_mamba(T) for _ in range(5)])
#
#         # 运动特征提取
#         # self.flow_net = optical_flow.Raft_Large(pretrained=True)
#         # self.flow_net = raft_large(progress=False, TIME_STEPS=in_channels)
#         # for param in self.flow_net.parameters():
#         #     param.requires_grad = False
#
#     def forward(self, x):
#         # x: [B,T,C,H,W]
#         B, T, C, H, W = x.shape
#         x = x.permute(0, 2, 1, 3, 4)
#         residual0 = x
#         # 编码过程
#         latent = x
#         residuals = []
#         for i in range(3):
#             latent = self.encoders[i](latent)
#             residuals.append(latent)
#         medi = latent
#         # 解码过程
#         for j in range(2):
#
#             latent = self.decoders[j](latent)
#             # latent = latent + self.wavelet[j](residuals[:-1][1 - j])  # 使用后2个Mamba层
#             # latent = self.mamba_layers[3 + j](latent) + self.wavelet[j](residuals[:-1][1-j])  # 使用后2个Mamba层
#
#             # latent = self.decoders[1](latent)
#             # latent = latent + self.wavelet[1](residuals[:-1][0])
#             # latent = self.mamba_layers[3 + j](latent) + self.wavelet[1](residuals[:-1][0])
#
#
#         recon = self.decoders[-1](latent)
#         # recon = recon + self.wavelet[-1](residual0)
#
#         recon = recon.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]
#         #
#         # # 光流运动特征  X:[B,C,T,H,W]
#         # flow_maps = []
#         # y = x.permute(0, 2, 1, 3, 4)
#         # for t in range(T - 1):
#         #     # flow = self.flow_net(x[:, t, :, :, :].squeeze(1), x[:, t + 1, :, :, :].squeeze(1))[0]
#         #     flow = self.flow_net(y[:, t], y[:, t + 1])[0]
#         #     flow_maps.append(flow)
#         # flow_stack = torch.stack(flow_maps, dim=1)  # [B,T-1,2,H,W]
#
#         return recon, medi
class SpectralSpatialAE(nn.Module):
    """空-谱自编码器（无监督背景重建）"""

    def __init__(self, in_channels=25, latent_dim=128, T=8):
        super().__init__()

        self.SpatialTemporalConv = OptimizedSpatialTemporalConv(in_channels, out_channels=64)
        self.SpatialTemporalConv1 = OptimizedDecomposedConv3D(in_channels=64, out_channels=128)
        self.SpatialTemporalConv2 = OptimizedDecomposedConv3D(in_channels=128, out_channels=latent_dim)

        self.DecomposedTransposeConv3D1 = OptimizedSpatialTemporalDeconv(in_channels=latent_dim, out_channels=128)
        self.DecomposedTransposeConv3D = OptimizedSpatialTemporalDeconv(in_channels=128, out_channels=64)
        self.SpatioTemporalTransposeConv = FinalReconstructionDeconv(in_channels=64, out_channels=in_channels)


        # 编码器
        self.encoders = nn.ModuleList([
            nn.Sequential(
                # nn.Conv3d(in_channels, 64, (5, 6, 6), stride=(1, 4, 4), padding=(2, 1, 1)),
                self.SpatialTemporalConv,
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                # nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                self.SpatialTemporalConv1,
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                # nn.Conv3d(128, 256, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
                self.SpatialTemporalConv2,
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.2)
            )
        ])

        # 解码器组件
        self.decoders = nn.ModuleList([
            nn.Sequential(
                # nn.ConvTranspose3d(256, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
                self.DecomposedTransposeConv3D1,
                nn.BatchNorm3d(128),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                # nn.ConvTranspose3d(128, 64, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
                self.DecomposedTransposeConv3D,
                nn.BatchNorm3d(64),
                nn.LeakyReLU(0.2)
            ),
            nn.Sequential(
                self.SpatioTemporalTransposeConv,
                # nn.ConvTranspose3d(64, in_channels, (5, 6, 6), stride=(1, 4, 4), padding=(2, 1, 1)),
                nn.Tanh()
            )
        ])

        # Mamba中间层
        # self.mamba_layers = nn.ModuleList([create_mamba(T) for _ in range(5)])
        # self.wavelet = nn.ModuleList([
        #     SpectralSpatialWavelet(in_bands=128),
        #     SpectralSpatialWavelet(in_bands=64),
        #     SpectralSpatialWavelet(in_bands=in_channels)
        # ])

        # Mamba中间层
        # self.mamba_layers = nn.ModuleList([create_mamba(T) for _ in range(5)])

        # 运动特征提取
        # self.flow_net = optical_flow.Raft_Large(pretrained=True)
        # self.flow_net = raft_large(progress=False, TIME_STEPS=in_channels)
        # for param in self.flow_net.parameters():
        #     param.requires_grad = False

    def forward(self, x):
        # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)
        residual0 = x
        # 编码过程
        latent = x
        residuals = []
        for i in range(2):
            latent = self.encoders[i](latent)
            # latent = self.mamba_layers[i](latent)  # 使用前3个Mamba层
            residuals.append(latent)
        medi = latent
        # 解码过程
        for j in range(1):
            # latent = self.mamba_layers[3 + j](latent) + residuals[:-1][1 - j]

            # latent = self.decoders[j](latent)
            # # latent = latent + self.wavelet[j](residuals[:-1][1 - j])  # 使用后2个Mamba层
            # latent = self.mamba_layers[3 + j](latent) + self.wavelet[j](residuals[:-1][1-j])  # 使用后2个Mamba层

            latent = self.decoders[1](latent)
            # latent = latent + self.wavelet[1](residuals[:-1][0])
            # latent = self.mamba_layers[3 + j](latent) + self.wavelet[1](residuals[:-1][0])


        recon = self.decoders[-1](latent)
        # recon = recon + self.wavelet[-1](residual0)

        recon = recon.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]

        # 光流运动特征  X:[B,C,T,H,W]
        # flow_maps = []
        # y = x.permute(0, 2, 1, 3, 4)
        # for t in range(T - 1):
        #     # flow = self.flow_net(x[:, t, :, :, :].squeeze(1), x[:, t + 1, :, :, :].squeeze(1))[0]
        #     flow = self.flow_net(y[:, t], y[:, t + 1])[0]
        #     flow_maps.append(flow)
        # flow_stack = torch.stack(flow_maps, dim=1)  # [B,T-1,2,H,W]

        return recon, medi


class SpectralTemporalAttention(nn.Module):
    """光谱-时空联合注意力机制"""

    def __init__(self, in_channels):
        super().__init__()
        self.spectral_att = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=(1, 1, 1)),
            nn.Sigmoid()
        )
        self.temporal_att = nn.Sequential(
            nn.Conv3d(in_channels, 1, kernel_size=(3, 1, 1), padding=(1, 0, 0)),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B,C,T,H,W)
        spectral_weights = self.spectral_att(x)  # 光谱维度压缩
        temporal_weights = self.temporal_att(x)  # 时间维度关注
        return x * (spectral_weights + temporal_weights)


class UnsupervisedLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.mse = nn.MSELoss()
        # self.mse = nn.L1Loss()
        # self.mse = nn.SmoothL1Loss()

    def spectral_consistency_loss(self, feats):
        """时序光谱一致性损失"""
        B, C, T, H, W = feats.shape
        feats = feats.permute(0, 2, 1, 3, 4)
        feats = feats.view(B, T, C, H * W)
        loss = 0
        for t in range(T - 1):
            # 计算相邻帧光谱特征相似度
            cos_sim = F.cosine_similarity(feats[:, t], feats[:, t + 1], dim=2)  # (B,H*W)
            loss += (1 - cos_sim.mean())  # 最大化相似度
            #
            # diff = feats[:, t] - feats[:, t + 1]
            # loss += torch.mean(torch.abs(diff))

        return loss / (T - 1)

    def forward(self, recon, inputs,  latent_feats):
        # 原损失计算
        recon_loss = self.mse(recon, inputs)

        # 新增光谱一致性损失
        spec_loss = self.spectral_consistency_loss(latent_feats)
        # return recon_loss
        return (self.alpha * recon_loss +
                self.beta * spec_loss)


def train(model, loader, device,  epochs=50, save_interval=10, dataname= None):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = UnsupervisedLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)
    print(dataname)
    # 创建检查点保存目录
    save_path = "checkpoints_Tbase0/" + dataname
    os.makedirs(save_path, exist_ok=True)

    model.train()
    epoch_pbar = tqdm(range(epochs), desc="Total Training Progress")

    for epoch in epoch_pbar:
        total_loss = 0.0
        batch_pbar = tqdm(enumerate(loader), total=len(loader),
                          desc=f'Epoch {epoch + 1}/{epochs}', leave=False)

        for batch_idx, (inputs, originals) in batch_pbar:
            inputs, originals = inputs.to(device), originals.to(device)

            optimizer.zero_grad()

            recon, latent = model(inputs)
            # if epoch % 3 == 0 and epoch != 0:
            # residual_img = (recon-inputs)
            # r_max = residual_img.max()
            # residual_img = r_max - residual_img
            # r_min, r_max = residual_img.min(), residual_img.max()
            # residual_img = (residual_img - r_min) / (r_max - r_min)
            #
            # loss = criterion(residual_img * recon, residual_img * inputs, latent)  # 传入潜在特征
            # else:
            loss = criterion(recon, inputs, latent)
            loss.backward()

            optimizer.step()



            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        # 每save_interval个epoch保存模型
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = (save_path+"/model_epoch_"+str(epoch + 1)+".pth")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")

        epoch_pbar.set_postfix({'epoch_loss': f"{avg_loss:.4f}"})


def post_process(anomaly_scores, flow_maps):
    """
    多阶段后处理流程
    :param anomaly_scores: 原始异常得分 (N, H, W)
    :param flow_maps: 光流场序列 (N-1, 2, H, W)
    :param time_window: 时间滑动窗口大小
    :param spatial_window: 空间滑动窗口大小
    :return: 优化后的异常得分 (N, H, W)
    """

    # 运动轨迹分析
    motion_enhanced = motion_trajectory_analysis(anomaly_scores, flow_maps)
    # return motion_enhanced


    return motion_enhanced


def warp_flow_with_direction(img, flow, direction_threshold=0.5):
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))

    # 计算光流方向一致性
    flow_magnitude = np.sqrt(flow[0] ** 2 + flow[1] ** 2)
    flow_direction_x = flow[0] / (flow_magnitude + 1e-8)
    flow_direction_y = flow[1] / (flow_magnitude + 1e-8)

    # 计算邻域方向一致性
    kernel = np.ones((3, 3), dtype=np.float32) / 9
    neighbor_consistency_x = cv2.filter2D(flow_direction_x, -1, kernel)
    neighbor_consistency_y = cv2.filter2D(flow_direction_y, -1, kernel)

    # 方向一致性掩码
    direction_mask = (
            (np.abs(flow_direction_x - neighbor_consistency_x) < direction_threshold) &
            (np.abs(flow_direction_y - neighbor_consistency_y) < direction_threshold)
    )

    # 传播像素位置
    new_x = (x + flow[0]).clip(0, w - 1)
    new_y = (y + flow[1]).clip(0, h - 1)

    # 应用方向一致性掩码
    new_x[~direction_mask] = x[~direction_mask]
    new_y[~direction_mask] = y[~direction_mask]

    return cv2.remap(img, new_x.astype(np.float32), new_y.astype(np.float32), cv2.INTER_LINEAR)


def motion_trajectory_analysis(scores, flow_maps, decay_factor=0.7, direction_threshold=0.5):
    """基于光流的运动轨迹分析，加入方向一致性约束"""
    enhanced_scores = np.zeros_like(scores)
    trajectory_map = np.zeros_like(scores[0])

    for t in range(scores.shape[0]):
        # 更新轨迹能量图
        trajectory_map = trajectory_map * decay_factor

        if t > 0:
            # 根据光流场传播轨迹（加入方向一致性约束）
            flow = flow_maps[t - 1]
            propagated_map = warp_flow_with_direction(
                trajectory_map, flow, direction_threshold
            )
            trajectory_map = np.maximum(propagated_map, scores[t])
        else:
            trajectory_map = scores[t]

        # 融合当前帧得分
        enhanced_scores[t] = trajectory_map * scores[t]

    return enhanced_scores


# 修改后的评估流程
def full_evaluation(model,  train_time, model_path, loader, device, TIME_STEPS, original_data, mean, std, true_labels, dataname):
    """
    完整评估流程
    original_data: 原始未标准化数据 (N,C,H,W)
    true_labels: 真实异常标签 (N,H,W)
    """
    checkpoint = torch.load(model_path)

    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取重建数据和光流场
    start = time.time()

    reconstructed, score_maps = inverse_transform(
        model, loader, device, original_data.shape, mean, std, original_data
    )
    data_dir = './results/'
    np.save(data_dir + ('TSSFMamba_base0_' + dataname + '.npy'), score_maps)

    # GT_flat = (score_maps[15, :, :] - score_maps[15, :, :].min()) / (
    #             score_maps[15, :, :].max() - score_maps[15, :, :].min() + 1e-10)
    # plt.imshow(GT_flat,
    #            cmap=plt.cm.hot,
    #            alpha=1  # 透明度调节
    #            )
    # plt.show()

    end = time.time()
    print("runtime of algorithm0：", end - start)
    # 计算评估指标

    test_time = end - start
    np.save(data_dir + ('train_time_TSSFMamba_base0_' + dataname + '.npy'), train_time)
    np.save(data_dir + ('test_time_TSSFMamba_base0_' + dataname + '.npy'), test_time)
    frame_metrics, (avg_auc, avg_f1), AUC = evaluate_system(score_maps,
        original_data, reconstructed, true_labels, None, dataname=dataname
    )

    return frame_metrics, (avg_auc, avg_f1), AUC


# 新增评估函数
def evaluate_system(score_maps, original_data, reconstructed_data, true_labels, flow_maps=None, dataname=None):
    """
    完整评估系统
    original_data: 原始数据 (N,C,H,W)
    reconstructed_data: 重建数据 (N,C,H,W)
    true_labels: 真实标签 (N,H,W)
    """
    # 计算异常得分
    anomaly_scores = np.mean((original_data - reconstructed_data) ** 2, axis=1)  # (N,H,W)
    print(dataname)
    data_dir = './results/'
    # anomaly_scores = get_auc(original_data, reconstructed_data)

    start = time.time()
    # scores_maps, gt_maps, mean_rx_auc = rx_anomaly_detection((original_data - reconstructed_data), true_labels)

    # scores_maps, gt_maps, mean_rx_auc = rx_anomaly_detection_gpu((original_data - reconstructed_data), true_labels, device='cuda')

    TSSNet = data_dir + 'TSSFMamba_base0_' + dataname + '.npy'
    # TSSNet = data_dir + 'MTSSNet_' + dataname + '.npy'
    TSSNet = np.load(TSSNet)
    scores_maps, gt_maps, mean_rx_auc, mean_snpr_maps, mean_od_maps, mean_dt_maps, mean_ft_maps = anomaly_detection(TSSNet, true_labels)


    Auto = data_dir + 'Auto_' + dataname + '.npy'
    Auto = np.load(Auto)
    scores_maps1, gt_maps, mean_rx_auc1, mean_snpr_maps1, mean_od_maps1, mean_dt_maps1, mean_ft_maps1  = anomaly_detection(Auto, true_labels)
    #
    MSNet = data_dir + 'MSNet_' + dataname + '.npy'
    MSNet = np.load(MSNet)
    scores_maps2, gt_maps, mean_rx_auc2, mean_snpr_maps2, mean_od_maps2, mean_dt_maps2, mean_ft_maps2  = anomaly_detection(MSNet, true_labels)
    #
    BockNet = data_dir + 'BockNet_' + dataname + '.npy'
    BockNet = np.load(BockNet)
    scores_maps3, gt_maps, mean_rx_auc3, mean_snpr_maps3, mean_od_maps3, mean_dt_maps3, mean_ft_maps3  = anomaly_detection(BockNet, true_labels)
    #
    GTHAD = data_dir + 'GTHAD_' + dataname + '.npy'
    GTHAD = np.load(GTHAD)
    scores_maps4, gt_maps, mean_rx_auc4, mean_snpr_maps4, mean_od_maps4, mean_dt_maps4, mean_ft_maps4 = anomaly_detection(GTHAD, true_labels)

    SSHAD = data_dir + 'SSHAD_' + dataname + '.npy'
    SSHAD = np.load(SSHAD)
    scores_maps5, gt_maps, mean_rx_auc5, mean_snpr_maps5, mean_od_maps5, mean_dt_maps5, mean_ft_maps5 = anomaly_detection(SSHAD, true_labels)

    PCA = data_dir + 'TLRA-MSL_' + dataname + '.npy'
    PCA = np.load(PCA)
    PCA = PCA.real
    scores_maps6, gt_maps, mean_rx_auc6, mean_snpr_maps6, mean_od_maps6, mean_dt_maps6, mean_ft_maps6 = anomaly_detection(PCA, true_labels)

    SS = data_dir + '2S-GLRT_' + dataname + '.npy'
    SS = np.load(SS)
    scores_maps7, gt_map7, mean_rx_auc7, mean_snpr_maps7, mean_od_maps7, mean_dt_maps7, mean_ft_maps7 = anomaly_detection(SS, true_labels)

    RX = data_dir + 'RX_' + dataname + '.npy'
    RX = np.load(RX)
    scores_maps8, gt_map8, mean_rx_auc8, mean_snpr_maps8, mean_od_maps8, mean_dt_maps8, mean_ft_maps8 = anomaly_detection(
        RX, true_labels)

    KIFD = data_dir + 'KIFD_' + dataname + '.npy'
    KIFD = np.load(KIFD)
    scores_maps9, gt_map9, mean_rx_auc9, mean_snpr_maps9, mean_od_maps9, mean_dt_maps9, mean_ft_maps9 = anomaly_detection(
        KIFD, true_labels)

    VF = data_dir + 'VT_' + dataname + '.npy'
    VF = np.load(VF)
    scores_maps10, gt_map10, mean_rx_auc10, mean_snpr_maps10, mean_od_maps10, mean_dt_maps10, mean_ft_maps10 = anomaly_detection(
        VF, true_labels)

    STH = data_dir + 'STH_' + dataname + '.npy'
    STH = np.load(STH)
    scores_maps11, gt_map11, mean_rx_auc11, mean_snpr_maps11, mean_od_maps11, mean_dt_maps11, mean_ft_maps11 = anomaly_detection(
        STH, true_labels)

    STF = data_dir + 'STF_' + dataname + '.npy'
    STF = np.load(STF)
    scores_maps12, gt_map12, mean_rx_auc12, mean_snpr_maps12, mean_od_maps12, mean_dt_maps12, mean_ft_maps12 = anomaly_detection(
        STF, true_labels)
    #
    print('-------------------------------------------------')
    print(f"Our算法平均AUC值: {mean_rx_auc:.4f}")
    print(f"Our算法平均SNPR值: {mean_snpr_maps:.4f}")
    print(f"Our算法平均OD值: {mean_od_maps:.4f}")
    print(f"Our算法平均DT值: {mean_dt_maps:.4f}")
    print(f"Our算法平均FT值: {mean_ft_maps:.4f}")
    our = []
    our.append(mean_rx_auc)
    our.append(mean_snpr_maps)
    our.append(mean_od_maps)
    our.append(mean_dt_maps)
    our.append(mean_ft_maps)
    print('-------------------------------------------------')
    print(f"Auto-AD算法平均AUC值: {mean_rx_auc1:.4f}")
    print(f"Auto-AD算法平均SNPR值: {mean_snpr_maps1:.4f}")
    print(f"Auto-AD算法平均OD值: {mean_od_maps1:.4f}")
    print(f"Auto-AD算法平均DT值: {mean_dt_maps1:.4f}")
    print(f"Auto-AD算法平均FT值: {mean_ft_maps1:.4f}")
    Auto = []
    Auto.append(mean_rx_auc1)
    Auto.append(mean_snpr_maps1)
    Auto.append(mean_od_maps1)
    Auto.append(mean_dt_maps1)
    Auto.append(mean_ft_maps1)
    print('-------------------------------------------------')
    print(f"MSNet算法平均AUC值: {mean_rx_auc2:.4f}")
    print(f"MSNet算法平均SNPR值: {mean_snpr_maps2:.4f}")
    print(f"MSNet算法平均OD值: {mean_od_maps2:.4f}")
    print(f"MSNet算法平均DT值: {mean_dt_maps2:.4f}")
    print(f"MSNet算法平均FT值: {mean_ft_maps2:.4f}")
    MSNet = []
    MSNet.append(mean_rx_auc2)
    MSNet.append(mean_snpr_maps2)
    MSNet.append(mean_od_maps2)
    MSNet.append(mean_dt_maps2)
    MSNet.append(mean_ft_maps2)
    print('-------------------------------------------------')
    print(f"BockNet算法平均AUC值: {mean_rx_auc3:.4f}")
    print(f"BockNet算法平均SNPR值: {mean_snpr_maps3:.4f}")
    print(f"BockNet算法平均OD值: {mean_od_maps3:.4f}")
    print(f"BockNet算法平均DT值: {mean_dt_maps3:.4f}")
    print(f"BockNet算法平均FT值: {mean_ft_maps3:.4f}")
    BockNet = []
    BockNet.append(mean_rx_auc3)
    BockNet.append(mean_snpr_maps3)
    BockNet.append(mean_od_maps3)
    BockNet.append(mean_dt_maps3)
    BockNet.append(mean_ft_maps3)
    print('-------------------------------------------------')
    print(f"GTHAD算法平均AUC值: {mean_rx_auc4:.4f}")
    print(f"GTHAD算法平均SNPR值: {mean_snpr_maps4:.4f}")
    print(f"GTHAD算法平均OD值: {mean_od_maps4:.4f}")
    print(f"GTHAD算法平均DT值: {mean_dt_maps4:.4f}")
    print(f"GTHAD算法平均FT值: {mean_ft_maps4:.4f}")
    GTHAD = []
    GTHAD.append(mean_rx_auc4)
    GTHAD.append(mean_snpr_maps4)
    GTHAD.append(mean_od_maps4)
    GTHAD.append(mean_dt_maps4)
    GTHAD.append(mean_ft_maps4)
    print('-------------------------------------------------')
    print(f"SSHAD算法平均AUC值: {mean_rx_auc5:.4f}")
    print(f"SSHAD算法平均SNPR值: {mean_snpr_maps5:.4f}")
    print(f"SSHAD算法平均OD值: {mean_od_maps5:.4f}")
    print(f"SSHAD算法平均DT值: {mean_dt_maps5:.4f}")
    print(f"SSHAD算法平均FT值: {mean_ft_maps5:.4f}")
    print('-------------------------------------------------')
    SSHAD = []
    SSHAD.append(mean_rx_auc5)
    SSHAD.append(mean_snpr_maps5)
    SSHAD.append(mean_od_maps5)
    SSHAD.append(mean_dt_maps5)
    SSHAD.append(mean_ft_maps5)
    print('-------------------------------------------------')
    print(f"TLRA-MSL算法平均AUC值: {mean_rx_auc6:.4f}")
    print(f"TLRA-MSL算法平均SNPR值: {mean_snpr_maps6:.4f}")
    print(f"TLRA-MSL算法平均OD值: {mean_od_maps6:.4f}")
    print(f"TLRA-MSL算法平均DT值: {mean_dt_maps6:.4f}")
    print(f"TLRA-MSL算法平均FT值: {mean_ft_maps6:.4f}")
    print('-------------------------------------------------')
    PCA = []
    PCA.append(mean_rx_auc6)
    PCA.append(mean_snpr_maps6)
    PCA.append(mean_od_maps6)
    PCA.append(mean_dt_maps6)
    PCA.append(mean_ft_maps6)
    print('-------------------------------------------------')
    print(f"2S-GLRT算法平均AUC值: {mean_rx_auc7:.4f}")
    print(f"2S-GLRT算法平均SNPR值: {mean_snpr_maps7:.4f}")
    print(f"2S-GLRT算法平均OD值: {mean_od_maps7:.4f}")
    print(f"2S-GLRT算法平均DT值: {mean_dt_maps7:.4f}")
    print(f"2S-GLRT算法平均FT值: {mean_ft_maps7:.4f}")
    print('-------------------------------------------------')
    SS = []
    SS.append(mean_rx_auc7)
    SS.append(mean_snpr_maps7)
    SS.append(mean_od_maps7)
    SS.append(mean_dt_maps7)
    SS.append(mean_ft_maps7)
    print('-------------------------------------------------')
    print(f"RX算法平均AUC值: {mean_rx_auc8:.4f}")
    print(f"RX算法平均SNPR值: {mean_snpr_maps8:.4f}")
    print(f"RX算法平均OD值: {mean_od_maps8:.4f}")
    print(f"RX算法平均DT值: {mean_dt_maps8:.4f}")
    print(f"RX算法平均FT值: {mean_ft_maps8:.4f}")
    print('-------------------------------------------------')
    RX= []
    RX.append(mean_rx_auc8)
    RX.append(mean_snpr_maps8)
    RX.append(mean_od_maps8)
    RX.append(mean_dt_maps8)
    RX.append(mean_ft_maps8)

    print('-------------------------------------------------')
    print(f"KIFD算法平均AUC值: {mean_rx_auc9:.4f}")
    print(f"KIFD算法平均SNPR值: {mean_snpr_maps9:.4f}")
    print(f"KIFD算法平均OD值: {mean_od_maps9:.4f}")
    print(f"KIFD算法平均DT值: {mean_dt_maps9:.4f}")
    print(f"KIFD算法平均FT值: {mean_ft_maps9:.4f}")
    print('-------------------------------------------------')
    KIFD= []
    KIFD.append(mean_rx_auc9)
    KIFD.append(mean_snpr_maps9)
    KIFD.append(mean_od_maps9)
    KIFD.append(mean_dt_maps9)
    KIFD.append(mean_ft_maps9)

    print('-------------------------------------------------')
    print(f"VF算法平均AUC值: {mean_rx_auc10:.4f}")
    print(f"VF算法平均SNPR值: {mean_snpr_maps10:.4f}")
    print(f"VF算法平均OD值: {mean_od_maps10:.4f}")
    print(f"VF算法平均DT值: {mean_dt_maps10:.4f}")
    print(f"VF算法平均FT值: {mean_ft_maps10:.4f}")
    print('-------------------------------------------------')
    VF = []
    VF.append(mean_rx_auc10)
    VF.append(mean_snpr_maps10)
    VF.append(mean_od_maps10)
    VF.append(mean_dt_maps10)
    VF.append(mean_ft_maps10)

    print('-------------------------------------------------')
    print(f"STH算法平均AUC值: {mean_rx_auc11:.4f}")
    print(f"STH算法平均SNPR值: {mean_snpr_maps11:.4f}")
    print(f"STH算法平均OD值: {mean_od_maps11:.4f}")
    print(f"STH算法平均DT值: {mean_dt_maps11:.4f}")
    print(f"STH算法平均FT值: {mean_ft_maps11:.4f}")
    print('-------------------------------------------------')
    STH = []
    STH.append(mean_rx_auc11)
    STH.append(mean_snpr_maps11)
    STH.append(mean_od_maps11)
    STH.append(mean_dt_maps11)
    STH.append(mean_ft_maps11)

    print('-------------------------------------------------')
    print(f"STF算法平均AUC值: {mean_rx_auc12:.4f}")
    print(f"STF算法平均SNPR值: {mean_snpr_maps12:.4f}")
    print(f"STF算法平均OD值: {mean_od_maps12:.4f}")
    print(f"STF算法平均DT值: {mean_dt_maps12:.4f}")
    print(f"STF算法平均FT值: {mean_ft_maps12:.4f}")
    print('-------------------------------------------------')
    STF = []
    STF.append(mean_rx_auc12)
    STF.append(mean_snpr_maps12)
    STF.append(mean_od_maps12)
    STF.append(mean_dt_maps12)
    STF.append(mean_ft_maps12)

    AUC = []
    AUC.append(our)
    AUC.append(GTHAD)
    AUC.append(MSNet)
    AUC.append(BockNet)
    AUC.append(Auto)
    AUC.append(SSHAD)
    AUC.append(PCA)
    AUC.append(SS)
    AUC.append(KIFD)
    AUC.append(RX)
    AUC.append(VF)
    AUC.append(STH)
    AUC.append(STF)
    # #
    # #
    combined_list = [np.concatenate((a, b, c, d, e, f, g, h, i, j, m, n, o), axis=1) for a, b, c, d, e, f, g, h, i, j, m, n, o in zip(scores_maps, scores_maps4, scores_maps2,
                                                scores_maps3, scores_maps1, scores_maps5, scores_maps6, scores_maps7, scores_maps9, scores_maps8, scores_maps10, scores_maps11, scores_maps12)]
    # Plot_3DROC_M_optimized(det_maps=combined_list, GTs=gt_maps, detec_label=['Ours', "GT-HAD", "MSNet", 'BockNet', "Auto-AD",  "SS-HAD",
    #                                                                          "TLRA-MSL",  "2S-GLRT",  "KIFD",  "FrFE-RX",  "VT",  "STH",  "STF"], datanames=dataname, save_prefix='output')


    PDPF = np.array((mean_rx_auc, mean_rx_auc4, mean_rx_auc2,
                    mean_rx_auc3, mean_rx_auc1, mean_rx_auc5, mean_rx_auc6, mean_rx_auc7, mean_rx_auc9, mean_rx_auc8, mean_rx_auc10, mean_rx_auc11, mean_rx_auc12))

    PDT = np.array((mean_dt_maps, mean_dt_maps4, mean_dt_maps2,
                    mean_dt_maps3, mean_dt_maps1, mean_dt_maps5, mean_dt_maps6, mean_dt_maps7, mean_dt_maps9, mean_dt_maps8, mean_dt_maps10, mean_dt_maps11, mean_dt_maps12))

    PFT = np.array((mean_ft_maps, mean_ft_maps4, mean_ft_maps2,
                    mean_ft_maps3, mean_ft_maps1, mean_ft_maps5, mean_ft_maps6, mean_ft_maps7, mean_ft_maps9, mean_ft_maps8, mean_ft_maps10, mean_ft_maps11, mean_ft_maps12))

    OD = np.array((mean_od_maps, mean_od_maps4, mean_od_maps2,
                    mean_od_maps3, mean_od_maps1, mean_od_maps5, mean_od_maps6, mean_od_maps7, mean_od_maps9, mean_od_maps8, mean_od_maps10, mean_od_maps11, mean_od_maps12))

    SNPR = np.array((mean_snpr_maps, mean_snpr_maps4, mean_snpr_maps2,
                    mean_snpr_maps3, mean_snpr_maps1, mean_snpr_maps5, mean_snpr_maps6, mean_snpr_maps7, mean_snpr_maps9, mean_snpr_maps8, mean_snpr_maps10, mean_snpr_maps11, mean_snpr_maps12))
    metric = np.vstack((PDPF, PDT, PFT, OD, SNPR))


    #
    # (n, h, w) = gt_maps.shape
    # combined_list1 = [a.reshape(h, w) for a in combined_list[0].T]
    # combined_list2 = combined_list1[::-1]
    # combined_list2.append(gt_maps)
    # algorithms = ['GT', 'Ours', "GT-HAD", "PDBSNet", "Auto-AD", 'RGAE', "LREN",
    #               "GTVLRR", "CRDBPSW", "EAS-RX", "CRD"][::-1]
    # from roc_plot2 import visualize_anomaly_comparison
    # visualize_anomaly_comparison(
    #     results=combined_list2,
    #     algorithms=algorithms,
    #     save_path='./output1/' + dataname + '_detection_comparison.pdf',
    #     cmap=plt.cm.hot,
    #     title=''
    # )


    #
    # plot_box_multiframe1(gt_maps, combined_list, dataname, ['Ours', "GT-HAD", "MSNet", 'BockNet', "Auto-AD",  "SS-HAD", "TLRA-MSL",  "2S-GLRT",  "KIFD",  "FrFE-RX",  "VT",  "STH",  "STF"], n_jobs=-1)


    # end = time.time()
    print("runtime of algorithm1：", end - start)
    # 打印结果
    def standardize_labels(labels):
        unique = np.unique(labels)
        if len(unique) != 2:
            raise ValueError("Labels must be binary")

        # 将较大值映射为1，较小值映射为0
        return np.where(labels == max(unique), 1, 0).astype(np.uint8)

    true_labels = standardize_labels(true_labels)

    # print(np.array(scores_maps).shape)
    scores_maps = np.array(scores_maps)
    if flow_maps is not None:
        anomaly_scores = post_process(scores_maps, flow_maps)
        # anomaly_scores = post_process(
        #     anomaly_scores, flow_maps, original_data, spectral_window=3
        # )
    # 逐帧计算指标
    frame_metrics = []
    for frame_idx in tqdm(range(anomaly_scores.shape[0]), desc="Evaluating Frames"):
        scores = anomaly_scores[frame_idx].flatten()
        labels = true_labels[frame_idx].flatten()

        # 计算AUC
        auc = roc_auc_score(labels, scores)

        # 计算最佳F1
        precision, recall, thresholds = precision_recall_curve(labels, scores, pos_label=1)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_f1 = np.max(f1_scores)
        # best_f1 = 1
        frame_metrics.append((auc, best_f1))

        # 计算AP
        # ap = average_precision_score(labels, scores, pos_label=1)
        # print(f"AP: {ap:.2f}")

        # # 绘制PR曲线
        # plt.plot(recall, precision, marker='.', label='Logistic Regression (AP = {:.2f})'.format(ap))
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.title('Precision-Recall Curve')
        # plt.legend()
        # plt.show()


    # 计算平均指标
    avg_auc = np.mean([m[0] for m in frame_metrics])
    avg_f1 = np.mean([m[1] for m in frame_metrics])

    return frame_metrics, (avg_auc, avg_f1), metric


if __name__ == "__main__":
    # 数据预处理

    # print(device)
    seed = 3407

    setup_seed(seed)
    # data_list = ['SanDiego100']
    T_dic = {'boat3': 8, 'boat7': 11, 'airplane2': 11, 'airplane13': 8, 'airplane15': 8, 'hsi_video': 10,
             'Terrain100': 11,
             'SHARE2012100': 8, 'SanDiego100': 11}
    data_list = ['boat3', 'boat7',  'airplane2', 'airplane13', 'airplane15', 'hsi_video' ,'SanDiego100', 'SHARE2012100', 'Terrain100']
    # data_list = ['boat3', 'boat7', 'boat12', 'airplane2', 'airplane9', 'airplane12', 'airplane13', 'airplane15',
    #              'hsi_video']
    # data_list = ['boat3', 'boat7',  'airplane2', 'airplane13', 'airplane15', 'hsi_video'], 'SanDiego100', 'SHARE2012100', 'Terrain100'
    # 70 200 70 153 204
    data_dir = './data/'
    AUCs = []
    nums = 0
    datasets = data_list  # 必须6个元素
    algorithms = ['Ours', "GT-HAD", "MSNet", 'BockNet', "Auto-AD",  "SS-HAD",
                "TLRA-MSL",  "2S-GLRT",  "KIFD",  "FrFE-RX",  "VT",  "STH",  "STF"]  # 必须11个算法名称
    metrics = ['PDPF', 'PDT', 'PFT', 'OD', 'SNPR']  # 必须5个指标名称
    results = np.zeros((len(data_list), len(algorithms), len(metrics)))   # 替换为真实数据，形状(6,11,5)
    for data in data_list: #['gulfport', 'Sandiego', 'pavia', 'cat-island',  'texas-goast', 'abu-urban-3']
        image_file = data_dir + data + '.npy'
        raw_data = np.load(image_file)
        num, channel, _, _ = raw_data.shape
        if data == 'hsi_video':
            gt_file = data_dir + 'ground_truth' + '.npy'
        else:
            gt_file = data_dir + data + '_gt' + '.npy'
        true_labels = np.load(gt_file)
        raw_data = raw_data.astype(np.float32)
        true_labels = true_labels.astype(np.float32)
        print('Data Preparation---------------------------------Data Preparation')
        TIME_STEPS = T_dic[data]# 15:222 17:111 13:244

        STRIDE = TIME_STEPS-2  # TIME_STEPS-2
        # TIME_STEPS = 1
        # STRIDE = 1
        EPOCH = 10

        # 数据预处理
        norm_series, mean, std, indices = preprocess_data(
            raw_data,
            time_steps=TIME_STEPS,
            stride=STRIDE
        )

        # 创建数据集
        dataset = TemporalDataset(norm_series, indices)
        train_loader = DataLoader(dataset, batch_size=2, shuffle=True)
        # 初始化模型
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        memory_before = torch.cuda.memory_allocated()
        os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(device)

        memory_before = torch.cuda.memory_allocated()
        model = SpectralSpatialAE(in_channels=channel, T=TIME_STEPS)
        model = nn.DataParallel(model, device_ids=[0, 1])
        model.to(device)

        print('Training---------------------------------Starting')
        # 训练模型
        # # Parameter calculate
        from thop import profile
        dummy_input = torch.randn(1, 1, 25, 256, 256)
        flops, params = profile(model.module, (dummy_input.cuda(device=device),))
        print('flops: ', flops, 'params: ', params)
        print('flops: %.2f M, params: %.2f M' % (flops / (1000000.0), params / 1000000.0))
        # #
        start = time.time()

        train(model, train_loader, device, epochs=EPOCH, save_interval=EPOCH, dataname=data)
        #
        end = time.time()
        print("runtime of running：", end - start)
        memory_after = torch.cuda.memory_allocated()

        memory_usage = memory_after - memory_before

        print(f"模型占用的显存大小约为：{memory_usage} bytes")
        # 使用最新检查点进行评估

        save_path = "checkpoints_Tbase0/" + data
        latest_checkpoint = save_path + "/model_epoch_" + str(EPOCH) + ".pth"
        # 评估模型
        print('Testing---------------------------------Starting')
        loader_eval = DataLoader(dataset, batch_size=4, shuffle=False)

        train_time = end - start
        # full_evaluation(model, train_time,
        #                 latest_checkpoint,
        #                 loader_eval,
        #                 device,
        #                 TIME_STEPS,
        #                 raw_data,  # 原始未标准化数据
        #                 mean,
        #                 std,
        #                 true_labels,
        #                 data
        #                 )
        frame_metrics, (avg_auc, avg_f1), metricss = full_evaluation(model,train_time,
            latest_checkpoint,
            loader_eval,
            device,
            TIME_STEPS,
            raw_data,  # 原始未标准化数据
            mean,
            std,
            true_labels,
            data
        )

        results[nums] = metricss.T
        nums = nums + 1

    data = {
        datasets[i]: pd.DataFrame(
            results[i],  # 提取第i个数据集的数据
            index=algorithms,
            columns=metrics
        ).round(4)  # 保留两位小数
        for i in range(results.shape[0])
    }
    import pandas as pd
    from pandas import ExcelWriter


    # 写入Excel文件
    def create_decimal_format(writer):
        """创建保留两位小数的单元格格式"""
        return writer.book.add_format({'num_format': '0.0000'})


    with ExcelWriter('Algorithm_Comparison_TSSFMamba_base0.xlsx', engine='xlsxwriter') as writer:
        for dataset, df in data.items():
            sheet_name = dataset[:31]  # Excel工作表名最长31字符

            # 写入数据到sheet（从第2行开始）
            df.to_excel(writer, sheet_name=sheet_name, startrow=1)

            # 获取工作表对象
            worksheet = writer.sheets[sheet_name]
            dec_format = create_decimal_format(writer)
            # 添加数据集标题（合并单元格）
            header_format = writer.book.add_format({
                'bold': True,
                'font_size': 14,
                'align': 'center',
                'valign': 'vcenter'
            })
            worksheet.merge_range('A1:F1', f'Dataset: {dataset}', header_format)
            # 设置数值列为两位小数格式
            worksheet.set_column('B:F', None, dec_format)  # B-F列应用格式
            # 设置列宽自适应
            for col_num, col_name in enumerate(df.columns):
                max_len = max(
                    len(str(col_name)),  # 列名长度
                    df[col_name].astype(str).str.len().max()  # 数据最大长度
                )
                worksheet.set_column(col_num + 1, col_num + 1, max_len + 2)  # +2字符缓冲

            # 冻结首行首列（算法名称列）
            worksheet.freeze_panes(2, 1)  # 行2，列1

        # 新增：算法平均表现汇总页 ------------------------------------------------
        # 计算每个算法在所有数据集上的指标均值（形状：11算法 × 5指标）
        algorithm_means = np.mean(results, axis=0)  # axis=0 沿数据集维度聚合

        # 构建汇总数据框架
        summary_df = pd.DataFrame(
            algorithm_means,
            index=algorithms,
            columns=[f'Avg {m}' for m in metrics]
        ).round(4)

        # 添加排名列
        summary_df['Overall Rank'] = summary_df.mean(axis=1).rank(ascending=False).astype(int)

        # 写入Excel并设置格式
        summary_df.to_excel(writer, sheet_name='Summary', startrow=2)
        summary_sheet = writer.sheets['Summary']

        # 合并标题行
        summary_header_format = writer.book.add_format({
            'bold': True, 'font_size': 16, 'align': 'center', 'valign': 'vcenter'
        })
        summary_sheet.set_column('B:F', None, dec_format)  # 数值列两位小数
        summary_sheet.merge_range('A1:G2', 'Algorithm Performance Summary', summary_header_format)

    print("数据已成功保存至 Algorithm_Comparison_move_base.xlsx")
    ##################################################################
