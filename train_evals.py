import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, f1_score
from tqdm import tqdm  # 新增tqdm导入
import os
from cnnlstm import *
from RX import *
from torchvision.models.optical_flow import raft_large
from torchvision.transforms import Resize
import cv2
from scipy.ndimage import median_filter

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

    # min_vals = time_series.min(axis=(2, 3, 4), keepdims=True)  # 计算每帧最小值
    # max_vals = time_series.max(axis=(2, 3, 4), keepdims=True)  # 计算每帧最大值
    # norm_series = (time_series - min_vals) / (max_vals - min_vals)


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


# 改进的逆变换函数
# def inverse_transform(model, loader, device, original_shape, mean, std):
#     """实时逆变换避免内存溢出"""
#     N_frames, C, H, W = original_shape
#     reconstructed = np.zeros(original_shape, dtype=np.float32)
#     count_matrix = np.zeros(original_shape, dtype=np.float32)
#     # torch_resize1 = Resize([H, W])
#     # 将标准化参数转换为Tensor
#     mean_tensor = torch.from_numpy(mean).to(device)
#     std_tensor = torch.from_numpy(std).to(device)
#     # 用于存储光流场数据
#     flow_maps = []  # 存储每对连续帧的光流场 (N_frames-1, 2, H, W)
#     model.eval()
#     with torch.no_grad():
#         for inputs, indices in tqdm(loader, desc="Inverse Transform"):
#             inputs = inputs.to(device)
#
#             outputs, flow, _ = model(inputs)
#             # 反标准化（在GPU执行）
#             outputs = outputs * std_tensor + mean_tensor
#
#             # 调整到原始空间尺寸
#             # outputs = torch_resize1(outputs)
#             outputs = F.interpolate(
#                 outputs.view(-1, *outputs.shape[2:]),  # (B*T,C,H,W)
#                 size=(H, W),
#                 mode='bilinear',
#                 align_corners=False
#             ).view(-1, outputs.shape[1], C, H, W)  # (B,T,C,H,W)
#
#             # 转换为CPU numpy
#             batch_pred = outputs.cpu().numpy()
#             batch_indices = indices.numpy()
#
#             # 逐样本处理
#             for i in range(len(batch_pred)):
#                 start_idx = batch_indices[i]
#                 end_idx = start_idx + batch_pred.shape[1]
#
#                 # 累加重建结果
#                 reconstructed[start_idx:end_idx] += batch_pred[i]
#                 count_matrix[start_idx:end_idx] += 1
#
#             flow_maps.append(flow.cpu().numpy())  # (B, T-1, 2, H, W)
#
#     # 处理未覆盖帧
#     count_matrix[count_matrix == 0] = 1
#     reconstructed = reconstructed / count_matrix
#     # 将光流场数据拼接为完整序列
#     flow_maps = np.concatenate(flow_maps, axis=0)  # (N_samples*(T-1), 2, H, W)
#     flow_maps = flow_maps.reshape(-1, 2, H, W)  # (N_frames-1, 2, H, W)
#     return reconstructed, flow_maps


def inverse_transform(model, loader, device, original_shape, mean, std):
    """改进的逆变换函数，精确对齐光流场"""
    N_frames, C, H, W = original_shape
    reconstructed = np.zeros(original_shape, dtype=np.float32)
    count_matrix = np.zeros(original_shape, dtype=np.float32)

    # 初始化光流场存储结构
    flow_accumulator = np.zeros((N_frames - 1, 2, H, W), dtype=np.float32)
    flow_counter = np.zeros(N_frames - 1, dtype=np.int32)

    # 获取模型输出的光流尺寸
    sample_input = next(iter(loader))[0][:1].to(device)
    with torch.no_grad():
        _, test_flow, _ = model(sample_input)
    _, _, _, H_flow, W_flow = test_flow.shape

    # 创建调整光流尺寸的转换器
    flow_resizer = Resize((H, W)) if H != H_flow or W != W_flow else None

    # 标准化参数转换
    mean_tensor = torch.from_numpy(mean).to(device)
    std_tensor = torch.from_numpy(std).to(device)

    model.eval()
    with torch.no_grad():
        for inputs, indices in tqdm(loader, desc="Inverse Transform"):
            inputs = inputs.to(device)
            batch_size = inputs.size(0)

            # 前向传播
            outputs, flow, _ = model(inputs)

            # === 重建数据处理 ===
            outputs = outputs * std_tensor + mean_tensor
            outputs = F.interpolate(
                outputs.view(-1, C, H_flow, W_flow),
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

            # === 光流数据处理 ===
            flow = flow.cpu().numpy()  # (B, T-1, 2, H_flow, W_flow)

            # 调整光流尺寸
            if flow_resizer is not None:
                resized_flows = []
                for b in range(flow.shape[0]):
                    frame_flows = []
                    for t in range(flow.shape[1]):
                        flow_tensor = torch.from_numpy(flow[b, t]).unsqueeze(0)
                        resized = flow_resizer(flow_tensor).squeeze(0).numpy()
                        frame_flows.append(resized)
                    resized_flows.append(np.stack(frame_flows))
                flow = np.stack(resized_flows)  # (B, T-1, 2, H, W)

            # 准确对齐光流场
            for b in range(batch_size):
                window_start = indices[b].item()
                for t in range(flow.shape[1]):
                    global_idx = window_start + t
                    if global_idx < N_frames - 1:
                        flow_accumulator[global_idx] += flow[b, t]
                        flow_counter[global_idx] += 1

    # 最终处理
    valid_flows = flow_counter > 0
    flow_accumulator[valid_flows] /= flow_counter[valid_flows, None, None, None]
    reconstructed = reconstructed / np.where(count_matrix > 0, count_matrix, 1)

    return reconstructed, flow_accumulator


class STContextAggregation(nn.Module):
    """时空上下文聚合模块"""

    def __init__(self, in_channels):
        super().__init__()
        self.spatial_conv = nn.Conv3d(in_channels, in_channels, (1, 3, 3), padding=(0, 1, 1))
        self.temporal_conv = nn.Conv3d(in_channels, in_channels, (3, 1, 1), padding=(1, 0, 0))
        self.attention = nn.Sequential(
            nn.Conv3d(2 * in_channels, in_channels // 2, 1),
            nn.ReLU(),
            nn.Conv3d(in_channels // 2, 2, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        spatial_feat = self.spatial_conv(x)
        temporal_feat = self.temporal_conv(x)
        combined = torch.cat([spatial_feat, temporal_feat], dim=1)
        att_weights = self.attention(combined)
        return att_weights[:, 0:1] * spatial_feat + att_weights[:, 1:2] * temporal_feat

class CascadeRefiner(nn.Module):
    """级联精修网络"""

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            nn.Conv3d(25, 64, (5, 3, 3), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.stage2 = nn.Sequential(
            nn.Conv3d(64 + 25, 128, (3, 3, 3), padding=(1, 1, 1)),
            SpectralTemporalAttention(128),
            nn.Conv3d(128, 64, 1)
        )
        self.stage3 = nn.Sequential(
            nn.Conv3d(64 + 25, 64, 3, padding=1),
            STContextAggregation(64)
        )

    def forward(self, x):
        s1 = self.stage1(x)
        s2 = self.stage2(torch.cat([s1, x], dim=1))
        s3 = self.stage3(torch.cat([s2, x], dim=1))
        return s1 + s2 + s3

class SpectralSpatialAE(nn.Module):
    """空-谱自编码器（无监督背景重建）"""

    def __init__(self, in_channels=25, latent_dim=128, T=8):
        super().__init__()
        # self.backbone = CascadeRefiner()
        # 编码器
        self.encoder = nn.Sequential(
            ConvLSTM3DNetwork(in_channels, 32, (5, 3, 3), 1),
            # CascadeRefiner(),
            nn.Conv3d(in_channels, 64, (5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            # SpectralAttention3D(64),
            SpectralTemporalAttention(64),
            nn.Conv3d(64, 128, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1)),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2)
        )
        # self.flat = nn.Flatten(start_dim=2)
        self.cn = nn.Conv2d(128 * T, latent_dim, 1)  # T=5帧时间窗
        self.nc = nn.Conv2d(latent_dim, 25 * T, 1)

        # 解码器
        self.decoder = nn.Sequential(
            # nn.Unflatten(1, (128, 5)),
            nn.ConvTranspose3d(25, 64, (3, 3, 3), stride=(1, 2, 2), padding=(1, 1, 1), output_padding=(0, 1, 1)),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(0.2),
            # SpectralAttention3D(64),
            SpectralTemporalAttention(64),
            nn.ConvTranspose3d(64, in_channels, (5, 3, 3), stride=(1, 2, 2), padding=(2, 1, 1),output_padding=(0, 1, 1)),
            nn.Tanh()
            # ConvLSTM3DNetwork(in_channels, 32, (5, 3, 3), 1)
        )

        # 运动特征提取
        # self.flow_net = optical_flow.Raft_Large(pretrained=True)
        self.flow_net = raft_large(progress=False, TIME_STEPS=T)
        for param in self.flow_net.parameters():
            param.requires_grad = False

    def forward(self, x):
        # x: [B,T,C,H,W]
        B, T, C, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4)  # [B,C,T,H,W]

        # 编码
        latent = self.encoder(x)  # [B,latent_dim,H//4,W//4]
        latent = latent.view(B,-1, H//4,W//4) # (4,128,5,64,64)==>(4,128,20480)
        latent = self.cn(latent)
        latent = self.nc(latent)
        latent = latent.view(B,C,T, H//4,W//4)
        # 解码重建
        recon = self.decoder(latent)  # [B,C,T,H,W]
        recon = recon.permute(0, 2, 1, 3, 4)  # [B,T,C,H,W]

        # 光流运动特征
        flow_maps = []
        for t in range(T - 1):
            # flow = self.flow_net(x[:, t, :, :, :].squeeze(1), x[:, t + 1, :, :, :].squeeze(1))[0]
            flow = self.flow_net(x[:, t], x[:, t + 1])[0]
            flow_maps.append(flow)
        flow_stack = torch.stack(flow_maps, dim=1)  # [B,T-1,2,H,W]

        return recon, flow_stack, latent


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
class SpectralAttention3D(nn.Module):
    """三维光谱注意力"""

    def __init__(self, channels):
        super().__init__()
        self.att = nn.Sequential(
            nn.Conv3d(channels, channels // 4, 1),
            nn.LeakyReLU(0.2),
            nn.Conv3d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        att = self.att(x.mean(dim=(3, 4), keepdim=True))
        return x * att



class UnsupervisedLoss(nn.Module):
    def __init__(self, alpha=0.7):
        super().__init__()
        self.alpha = alpha
        self.mse = nn.MSELoss()

    def forward(self, recon, inputs, flow_pred):
        # 重建损失
        recon_loss = self.mse(recon, inputs)

        # 运动平滑约束
        flow_loss = F.huber_loss(flow_pred[:, :-1], flow_pred[:, 1:])

        # 特征紧凑性约束
        compact_loss = torch.norm(flow_pred, p=2, dim=1).mean()

        return self.alpha * recon_loss + (1 - self.alpha) * (flow_loss + 0.1 * compact_loss)

        # return recon_loss

def train(model, loader, device,  epochs=50, save_interval=10):
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = UnsupervisedLoss()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

    # 创建检查点保存目录
    os.makedirs("checkpoints", exist_ok=True)

    model.train()
    epoch_pbar = tqdm(range(epochs), desc="Total Training Progress")

    for epoch in epoch_pbar:
        total_loss = 0.0
        batch_pbar = tqdm(enumerate(loader), total=len(loader),
                          desc=f'Epoch {epoch + 1}/{epochs}', leave=False)

        for batch_idx, (inputs, originals) in batch_pbar:
            inputs, originals = inputs.to(device), originals.to(device)

            optimizer.zero_grad()
            # outputs = model(inputs)

            recon, flow, _ = model(inputs)
            loss = criterion(recon, inputs, flow)

            # 反标准化并恢复原始尺寸
            # outputs = outputs * std + mean
            # outputs = F.interpolate(outputs, size=(216, 409), mode='bilinear', align_corners=False)
            # loss = criterion(outputs, originals)

            loss.backward()
            optimizer.step()



            total_loss += loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            batch_pbar.set_postfix({'loss': f"{loss.item():.4f}", 'lr': f"{current_lr:.2e}"})

        avg_loss = total_loss / len(loader)
        scheduler.step(avg_loss)

        # 每save_interval个epoch保存模型
        if (epoch + 1) % save_interval == 0 or epoch == epochs - 1:
            checkpoint_path = f"checkpoints/model_epoch_{epoch + 1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint to {checkpoint_path}")

        epoch_pbar.set_postfix({'epoch_loss': f"{avg_loss:.4f}"})


def post_process(anomaly_scores, flow_maps, time_window=5, spatial_window=5):
    """
    多阶段后处理流程
    :param anomaly_scores: 原始异常得分 (N, H, W)
    :param flow_maps: 光流场序列 (N-1, 2, H, W)
    :param time_window: 时间滑动窗口大小
    :param spatial_window: 空间滑动窗口大小
    :return: 优化后的异常得分 (N, H, W)
    """
    # 时间维度处理
    # temp_smoothed = temporal_consistency_filter(anomaly_scores, window_size=time_window)

    # 运动轨迹分析
    motion_enhanced = motion_trajectory_analysis(anomaly_scores, flow_maps)
    # return motion_enhanced
    # 空间维度处理
    # spatial_refined = spatial_context_filter(motion_enhanced, kernel_size=spatial_window)

    return motion_enhanced


def temporal_consistency_filter(scores, window_size=5):
    """时间连续性滤波"""
    # 使用中值滤波保持边缘
    filtered = np.zeros_like(scores)
    for i in range(scores.shape[0]):
        start = max(0, i - window_size // 2)
        end = min(scores.shape[0], i + window_size // 2 + 1)
        filtered[i] = np.median(scores[start:end], axis=0)
    return filtered


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
# def motion_trajectory_analysis(scores, flow_maps, decay_factor=0.7):
#     """基于光流的运动轨迹分析"""
#     enhanced_scores = np.zeros_like(scores)
#     trajectory_map = np.zeros_like(scores[0])
#
#     for t in range(scores.shape[0]):
#         # 更新轨迹能量图
#         trajectory_map = trajectory_map * decay_factor
#
#         if t > 0:
#             # 根据光流场传播轨迹
#             flow = flow_maps[t - 1]
#             propagated_map = warp_flow(trajectory_map, flow)
#             trajectory_map = np.maximum(propagated_map, scores[t])
#         else:
#             trajectory_map = scores[t]
#
#         # 融合当前帧得分
#         enhanced_scores[t] = trajectory_map * scores[t]
#
#     return enhanced_scores


def warp_flow(img, flow):
    """基于光流场进行像素位置传播"""
    h, w = img.shape
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    new_x = (x + flow[0]).clip(0, w - 1)
    new_y = (y + flow[1]).clip(0, h - 1)
    return cv2.remap(img, new_x.astype(np.float32), new_y.astype(np.float32), cv2.INTER_LINEAR)


def spatial_context_filter(scores, kernel_size=5):
    """空间上下文滤波"""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filtered = np.zeros_like(scores)

    for t in range(scores.shape[0]):
        # 各向异性扩散滤波
        filtered[t] = cv2.medianBlur(scores[t].astype(np.float32), kernel_size)
        filtered[t] = cv2.morphologyEx(filtered[t], cv2.MORPH_CLOSE, kernel)

    return filtered


# 修改后的评估流程
def full_evaluation(model_path, loader, device, TIME_STEPS, original_data, mean, std, true_labels):
    """
    完整评估流程
    original_data: 原始未标准化数据 (N,C,H,W)
    true_labels: 真实异常标签 (N,H,W)
    """
    checkpoint = torch.load(model_path)
    model = SpectralSpatialAE(in_channels=25, T=TIME_STEPS).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 获取重建数据和光流场
    reconstructed, flow_maps = inverse_transform(
        model, loader, device, original_data.shape, mean, std
    )

    # 计算评估指标
    frame_metrics, (avg_auc, avg_f1) = evaluate_system(
        original_data, reconstructed, true_labels, flow_maps
    )

    return frame_metrics, (avg_auc, avg_f1)


    # reconstructed = inverse_transform(model, loader, device,
    #                                   original_data.shape, mean, std)
    #
    # # 计算评估指标
    # return evaluate_system(original_data, reconstructed, true_labels)


# 新增评估函数
def evaluate_system(original_data, reconstructed_data, true_labels, flow_maps=None):
    """
    完整评估系统
    original_data: 原始数据 (N,C,H,W)
    reconstructed_data: 重建数据 (N,C,H,W)
    true_labels: 真实标签 (N,H,W)
    """
    # 计算异常得分
    anomaly_scores = np.mean((original_data - reconstructed_data) ** 2, axis=1)  # (N,H,W)
    scores_maps, mean_rx_auc = rx_anomaly_detection((original_data - reconstructed_data), true_labels)

    # 打印结果

    print(f"RX算法平均AUC值: {mean_rx_auc:.4f}")
    # print(np.array(scores_maps).shape)
    scores_maps = np.array(scores_maps)
    if flow_maps is not None:
        anomaly_scores = post_process(scores_maps, flow_maps)

    def standardize_labels(labels):
        unique = np.unique(labels)
        if len(unique) != 2:
            raise ValueError("Labels must be binary")

        # 将较大值映射为1，较小值映射为0
        return np.where(labels == max(unique), 1, 0).astype(np.uint8)

    true_labels = standardize_labels(true_labels)

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

    # 计算平均指标
    avg_auc = np.mean([m[0] for m in frame_metrics])
    avg_f1 = np.mean([m[1] for m in frame_metrics])

    return frame_metrics, (avg_auc, avg_f1)


if __name__ == "__main__":
    # 数据预处理
    # 参数设置
    print('Data Preparation---------------------------------Data Preparation')
    TIME_STEPS = 5
    STRIDE = 3
    # raw_data = np.load('hsi_video.npy')
    # true_labels = np.load('ground_truth.npy')

    raw_data = np.load('./data/boat3.npy')  # 形状 (N_frames, C, H, W)
    true_labels = np.load('./data/boat3_gt.npy')  # 形状 (N_frames, H, W)
    # 模拟数据
    # raw_data = np.random.randn(500, 25, 216, 409).astype(np.float32)
    # true_labels = np.random.randint(0, 2, size=(500, 216, 409))  # 替换真实标签

    # 数据预处理
    norm_series, mean, std, indices = preprocess_data(
        raw_data,
        time_steps=TIME_STEPS,
        stride=STRIDE
    )

    # 创建数据集
    dataset = TemporalDataset(norm_series, indices)
    train_loader = DataLoader(dataset, batch_size=16, shuffle=True)
    # 初始化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectralSpatialAE(in_channels=25, T=TIME_STEPS).to(device)

    print('Training---------------------------------Starting')
    # 训练模型
    train(model, train_loader, device, epochs=100, save_interval=10)


    # 使用最新检查点进行评估
    checkpoint_files = sorted([f for f in os.listdir("checkpoints") if f.endswith(".pth")])
    latest_checkpoint = os.path.join("checkpoints", checkpoint_files[-1])

    # 评估模型
    print('Testing---------------------------------Starting')
    loader_eval = DataLoader(dataset, batch_size=1, shuffle=False)
    frame_metrics, (avg_auc, avg_f1) = full_evaluation(
        latest_checkpoint,
        loader_eval,
        device,
        TIME_STEPS,
        raw_data,  # 原始未标准化数据
        mean,
        std,
        true_labels
    )
    print('Evaling---------------------------------Starting')
    # 打印结果
    print(f"\nAverage AUC: {avg_auc:.4f}")
    print(f"Average F1: {avg_f1:.4f}")

    # 保存逐帧结果
    np.savez("evaluation_results.npz",
             auc_scores=[m[0] for m in frame_metrics],
             f1_scores=[m[1] for m in frame_metrics])
    print('Running---------------------------------Ending')