import numpy as np
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import cv2
import time
from roc_plot import *
from roc_plot1 import *
def rx_anomaly_detection(data, labels):
    """
    对高光谱序列图像应用RX算法进行异常检测，并计算每帧的AUC值及平均AUC
    :param data: 原始高光谱数据 (N_frames, C, H, W)
    :param labels: 真实异常标签 (N_frames, H, W)
    :return: 每帧的AUC值列表, 平均AUC值
    """
    rx_auc_scores = []
    auc_scores = []
    scores_maps = []
    sco_maps = []
    gt_maps = []
    num_frames, num_channels, height, width = data.shape

    for i in tqdm(range(num_frames), desc="RX Processing"):
        frame = data[i]  # 当前帧数据 (C, H, W)
        pixels = frame.reshape(num_channels, -1).T  # 转换为 (H*W, C)

        # 计算当前帧的统计量
        mu = np.mean(pixels, axis=0)  # 均值 (C,)
        cov = np.cov(pixels, rowvar=False)  # 协方差矩阵 (C, C)
        cov_reg = cov + np.eye(num_channels) * 1e-6  # 正则化协方差矩阵
        inv_cov = np.linalg.inv(cov_reg)  # 协方差矩阵的逆

        # 计算RX得分（马氏距离）
        diff = pixels - mu  # 差值 (H*W, C)
        scores = np.sum(diff @ inv_cov * diff, axis=1)  # RX得分 (H*W,)
        scores = scores.reshape(height, width)  # 恢复为图像尺寸 (H, W)
        from scipy.ndimage import gaussian_filter
        from skimage.morphology import opening, closing, disk
        scores = gaussian_filter(scores, sigma=1)

        # kernel = disk(radius=1)  # 定义形态学核
        # scores = opening(scores, kernel)
        # scores = closing(scores, kernel)
        # 计算AUC
        scores_maps.append(scores)
        auc = roc_auc_score(labels[i].flatten(), scores.flatten())

        # mask_flat = labels[i].reshape((1, height*width), order='F').ravel()
        # ano_score = scores.flatten()
        # auc1 = roc_auc_score(labels[i].flatten(), ano_score)
        # print(auc1)

        sco = scores.flatten()[:, np.newaxis]
        # gt = labels[i].reshape((1, height*width), order='F').ravel()
        sco_maps.append(sco)
        gt_maps.append(labels[i])
        AUC, AUCnor, AUCod, AUCsnpr = cal_AUC_optimized(sco_maps[i], labels[i], mode_tau=1, mode_eq=1)
        # auc_scores.append(AUCsnpr[0])
        # auc_scores.append(AUCod[0])
        # auc_scores.append(AUC['tauPD'][0])
        # auc_scores.append(AUC['tauPF'][0])
        auc_scores.append(AUC['PFPD'][0])
        rx_auc_scores.append(auc)

    # Plot_3DROC_M(det_maps=sco_maps, GTs=gt_maps, detec_label=['RX'], datanames='boat3', save_prefix='output')
    # Plot_3DROC_M_optimized(det_maps=sco_maps, GTs=gt_maps, detec_label=['RX'], datanames='boat3', save_prefix='output')

    # print(f"PFPD: {AUC['PFPD'][0]}:.4f")
    # print(f"tauPD: {AUC['tauPD'][0]}:.4f")
    # print(f"tauPF: {AUC['tauPF'][0]}:.4f")
    # print(f"AUCod: {AUCod[0]}:.4f")
    # print(f"AUCsnpr: {AUCsnpr[0]}:.4f")

    mean_auc_scores = np.mean(auc_scores)
    print(f"算法平均AUC值: {mean_auc_scores:.4f}")
    mean_rx_auc = np.mean(rx_auc_scores)
    return scores_maps, gt_maps, mean_auc_scores

import torch
# import numpy as np
# from tqdm import tqdm
# from sklearn.metrics import roc_auc_score

def rx_anomaly_detection_gpu(data, labels, device='cuda'):
    """
    GPU加速版RX异常检测 (使用PyTorch)
    :param data: 高光谱数据 (N_frames, C, H, W) [numpy数组]
    :param labels: 标签 (N_frames, H, W) [numpy数组]
    :param device: 设备 ('cuda' 或 'cpu')
    :return: 同原函数
    """
    # 将数据转换为PyTorch Tensor并移至GPU
    data_tensor = torch.from_numpy(data).float().to(device)
    num_frames, num_channels, height, width = data_tensor.shape
    rx_auc_scores = []
    auc_scores = []
    scores_maps = []
    sco_maps = []
    gt_maps = []

    for i in tqdm(range(num_frames), desc="RX GPU Processing"):
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
        scores = gaussian_filter(scores, sigma=1)

        # 计算AUC
        gt = labels[i].flatten()
        auc = roc_auc_score(gt, scores.flatten())
        rx_auc_scores.append(auc)
        sco_maps.append(scores.flatten()[:, np.newaxis])
        gt_maps.append(labels[i])
        # 其他自定义AUC计算...
        AUC, AUCnor, AUCod, AUCsnpr = cal_AUC_optimized(sco_maps[i], labels[i], mode_tau=1, mode_eq=1)
        # auc_scores.append(AUCsnpr[0])
        # auc_scores.append(AUCod[0])
        # auc_scores.append(AUC['tauPD'][0])
        # auc_scores.append(AUC['tauPF'][0])
        auc_scores.append(AUC['PFPD'][0])

    mean_rx_auc = np.mean(auc_scores)
    return sco_maps, gt_maps, mean_rx_auc
def anomaly_detection(data, labels):
    """
    对高光谱序列图像应用RX算法进行异常检测，并计算每帧的AUC值及平均AUC
    :param data: 原始高光谱数据 (N_frames, C, H, W)
    :param labels: 真实异常标签 (N_frames, H, W)
    :return: 每帧的AUC值列表, 平均AUC值
    """
    # rx_auc_scores = []
    auc_scores = []
    scores_maps = []
    sco_maps = []
    gt_maps = []
    snpr_maps = []
    od_maps = []
    dt_maps = []
    ft_maps = []
    num_frames, height, width = data.shape

    for i in tqdm(range(num_frames), desc="Evaling Processing"):


        # 计算AUC
        scores_maps.append(data[i])
        # auc = roc_auc_score(labels[i].flatten(), data[i].flatten())
        # from scipy.ndimage import gaussian_filter
        # data[i] = gaussian_filter(data[i], sigma=1)
        sco = data[i].flatten()[:, np.newaxis]

        sco_maps.append(sco)
        gt_maps.append(labels[i])
        AUC, AUCnor, AUCod, AUCsnpr = cal_AUC_optimized(sco_maps[i], labels[i], mode_tau=1, mode_eq=1)
        # auc_scores.append(AUCsnpr[0])
        od_maps.append(AUCod[0])
        dt_maps.append(AUC['tauPD'][0])
        ft_maps.append(AUC['tauPF'][0])
        auc_scores.append(AUC['PFPD'][0])
        snpr_maps.append(AUCsnpr[0])

        # print(f"PFPD: {AUC['PFPD'][0]}:.4f")
        # print(f"tauPD: {AUC['tauPD'][0]}:.4f")
        # print(f"tauPF: {AUC['tauPF'][0]}:.4f")
        # print(f"AUCod: {AUCod[0]}:.4f")
        # print(f"AUCsnpr: {AUCsnpr[0]}:.4f")



    mean_dt_maps = np.mean(dt_maps)
    mean_ft_maps = np.mean(ft_maps)
    mean_auc_scores = np.mean(auc_scores)
    if mean_dt_maps == 0:
        mean_dt_maps = 0.0001
    # mean_snpr_maps = np.mean(snpr_maps)
    # mean_od_maps = np.mean(od_maps)

    mean_snpr_maps = mean_dt_maps / mean_ft_maps
    mean_od_maps = mean_dt_maps + mean_auc_scores-mean_ft_maps
    print(f"算法平均AUCod值: {mean_auc_scores:.4f}")

    return sco_maps, gt_maps, mean_auc_scores, mean_snpr_maps, mean_od_maps, mean_dt_maps, mean_ft_maps

# if __name__ == "__main__":
#     # 加载数据
#
#     # raw_data = np.load('./data/hsi_video.npy')  # 形状 (N_frames, C, H, W)
#     # true_labels = np.load('./data/ground_truth.npy')  # 形状 (N_frames, H, W)
#
#     raw_data = np.load('./data/boat3.npy')  # 形状 (N_frames, C, H, W)
#     true_labels = np.load('./data/boat3_gt.npy')  # 形状 (N_frames, H, W)
#     # 应用RX算法进行异常检测
#     start = time.time()
#
#     rx_auc_scores, mean_rx_auc = rx_anomaly_detection(raw_data, true_labels)
#     # rx_auc_scores, mean_rx_auc = rx_anomaly_detection(raw_data[0:1, :, :, :], true_labels[0:1, :, :])
#     end = time.time()
#     print("runtime of algorithm：", end - start)
#
#
#     # 打印结果
#     # print(f"RX算法每帧AUC值: {rx_auc_scores}")
#     print(f"RX算法平均AUC值: {mean_rx_auc:.4f}")
#
#     # 保存结果
#     np.savez("rx_anomaly_results.npz",
#              rx_auc_scores=rx_auc_scores,
#              mean_rx_auc=mean_rx_auc)


