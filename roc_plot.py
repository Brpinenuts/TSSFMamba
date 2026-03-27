import numpy as np

from scipy import integrate
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import rcParams
from sklearn.preprocessing import MinMaxScaler
import os
import seaborn as sns
import pandas as pd
from matplotlib.patches import PathPatch
def cal_pdpf(result, GT_flat):
    BKG = np.where(GT_flat == 0)[0]
    g_result = np.where(GT_flat == 1)[0]
    d_result = np.where(result == 1)[0]

    PD = len(np.intersect1d(d_result, g_result)) / len(g_result) if len(g_result) > 0 else 0.0
    PF = len(np.intersect1d(d_result, BKG)) / len(BKG) if len(BKG) > 0 else 0.0
    return PD, PF


def cal_AUC(det_map, GT, mode_tau=1, mode_eq=1):
    num_map = det_map.shape[1]
    GT_flat = GT.flatten()
    GT_flat = (GT_flat - GT_flat.min()) / (GT_flat.max() - GT_flat.min())
    # 归一化每一列
    for i in range(num_map):
        min_val = np.min(det_map[:, i])
        max_val = np.max(det_map[:, i])
        if max_val - min_val != 0:
            det_map[:, i] = (det_map[:, i] - min_val) / (max_val - min_val)
        else:
            det_map[:, i] = 0.0

    # 生成tau
    if mode_tau == 1:
        tau = np.sort(det_map, axis=0)[::-1]  # 按列降序排列
    else:
        tau_uniform = np.arange(0, 1.01, 0.01)
        tau = np.tile(tau_uniform.reshape(-1, 1), (1, num_map))
        tau = np.sort(tau, axis=0)[::-1]  # 按列降序排列

    num_tau = tau.shape[0]
    PD = np.zeros((num_tau, num_map))
    PF = np.zeros((num_tau, num_map))

    for k in range(num_map):
        for i in range(num_tau):
            current_tau = tau[i, k]
            AD_bw = np.zeros_like(det_map[:, k])
            if mode_eq == 1:
                AD_bw[det_map[:, k] >= current_tau] = 1
            else:
                AD_bw[det_map[:, k] > current_tau] = 1
            pd, pf = cal_pdpf(AD_bw, GT_flat)
            PD[i, k] = pd
            PF[i, k] = pf



    # 计算归一化参数
    a1 = np.min(PD[0, :])
    a0 = np.max(PD)
    b1 = np.min(PF[0, :])
    b0 = np.max(PF)

    AUC = {
        'PFPD': np.zeros(num_map),
        'tauPD': np.zeros(num_map),
        'tauPF': np.zeros(num_map)
    }

    AUCnor = {
        'PFPD': np.zeros(num_map),
        'tauPD': np.zeros(num_map),
        'tauPF': np.zeros(num_map),
        'a1': a1, 'a0': a0, 'b1': b1, 'b0': b0
    }

    # 计算各项AUC
    for i in range(num_map):
        # PFPD
        auc_pfpd = integrate.trapezoid(PD[:, i], PF[:, i])
        AUC['PFPD'][i] = np.round(auc_pfpd, 6)
        AUCnor['PFPD'][i] = np.round((auc_pfpd - a1) / ((a0 - a1) * (b0 - b1)), 6)

        # tauPD
        auc_taupd = -integrate.trapezoid(PD[:, i], tau[:, i])
        AUC['tauPD'][i] = np.round(auc_taupd, 6)
        AUCnor['tauPD'][i] = np.round((auc_taupd - a1) / (a0 - a1), 6)

        # tauPF
        auc_taupf = np.abs(integrate.trapezoid(PF[:, i], tau[:, i]))
        AUC['tauPF'][i] = np.round(auc_taupf, 6)
        AUCnor['tauPF'][i] = np.round((auc_taupf - b1) / (b0 - b1), 6)
    AUCod = AUCnor['PFPD'] + AUCnor['tauPD'] - AUCnor['tauPF']
    AUCsnpr = AUCnor['tauPD'] / AUCnor['tauPF']
    return AUC, AUCnor, AUCod, AUCsnpr


def plot_box(hsi_gt, methods_results, datanames, method_names):
    """
    修正版箱线图绘制函数，解决'set_ydata'错误问题

    参数说明：
    hsi_gt : array-like (N,)
        一维真实标签（0：背景，1：目标）
    methods_results : list of arrays
        检测结果列表，每个元素为(N,)数组
    datanames : str
        数据集名称（用于保存文件）
    method_names : list of str
        方法名称列表
    """
    # 数据预处理
    hsi_gt = np.asarray(hsi_gt).flatten()
    num_methods = len(methods_results)

    # 获取索引
    target_idx = np.where(hsi_gt == 1)[0]
    background_idx = np.where(hsi_gt == 0)[0]

    # 准备绘图数据
    data = []
    positions = []
    colors = []

    # 生成数据布局
    for i, result in enumerate(methods_results):
        result = np.asarray(result).flatten()

        # 目标数据（左偏）
        data.append(result[target_idx])
        positions.append(i - 0.15)
        colors.append('firebrick')

        # 背景数据（右偏）
        data.append(result[background_idx])
        positions.append(i + 0.15)
        colors.append('forestgreen')

    # 创建图形
    plt.figure(figsize=(8, 6))
    ax = plt.gca()

    # 绘制箱线图
    boxplot = ax.boxplot(
        data,
        positions=positions,
        widths=0.2,
        patch_artist=True,
        showfliers=False,
        whis=(0, 100),  # 须线扩展到全数据范围
        labels=[n for pair in zip(method_names, method_names) for n in pair]  # 临时标签
    )

    # 设置箱体颜色
    for box, color in zip(boxplot['boxes'], colors):
        box.set_facecolor(color)
        box.set_alpha(0.7)
        box.set_linewidth(1.2)

    # 关键修正：手动调整箱体高度至10%-90%分位数
    for i, box in enumerate(boxplot['boxes']):
        # 获取对应数据
        current_data = data[i]

        # 计算分位数
        q10 = np.percentile(current_data, 10)
        q90 = np.percentile(current_data, 90)

        # 获取当前箱体的路径顶点
        path = box.get_path()
        vertices = path.vertices

        # 原始顶点结构：
        # [左下, 左上, 右上, 右下, 左下, 顶点0]
        # 修改Y坐标为新的分位数
        vertices[0:4, 1] = [q10, q10, q90, q90]  # 调整箱体高度

        # 更新路径
        box.set_path(path)

    # 设置其他样式
    plt.setp(boxplot['whiskers'], color='k', linewidth=1.2)
    plt.setp(boxplot['caps'], color='k', linewidth=1.2)
    plt.setp(boxplot['medians'], color='k', linewidth=1.5)

    # 设置坐标轴
    ax.set_ylabel('Detection Test Statistic Range', fontsize=12)
    ax.set_xticks(np.arange(num_methods))
    ax.set_xticklabels(method_names, rotation=20, ha='right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.6)

    # 创建图例
    legend_elements = [
        Patch(facecolor='firebrick', alpha=0.7, label='Anomaly'),
        Patch(facecolor='forestgreen', alpha=0.7, label='Background')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    # 调整布局并保存
    plt.tight_layout()
    plt.savefig(f'{datanames}_box.pdf', bbox_inches='tight')
    plt.show()


def Plot_3DROC_M(det_maps, GTs, detec_label, mode_eq=1, datanames=None, save_prefix=None):
    """核心绘图函数"""
    # 输入处理
    PDnors = []
    PFnors = []
    # tau = np.linspace(1, 0, 5000)
    for k in range(len(det_maps)):
        det_map = det_maps[k]
        GT = GTs[k]
        num_map = det_map.shape[1]
        GT_flat = GT.flatten()
        N = GT_flat.size
        GT_flat = (GT_flat - GT_flat.min()) / (GT_flat.max() - GT_flat.min())
        # 归一化检测结果
        for i in range(num_map):
            min_val = det_map[:, i].min()
            max_val = det_map[:, i].max()

            if max_val == min_val:
                print("警告：特征图所有值相同，归一化后全为0或常数值！")
            det_map[:, i] = (det_map[:, i] - min_val) / (max_val - min_val)

        # 生成阈值tau
        tau = np.linspace(1, 0, 5000)
        num_tau = len(tau)

        # 预分配内存
        PD = np.zeros((num_tau, num_map))
        PF = np.zeros((num_tau, num_map))

        # 计算PD和PF
        for k in range(num_map):
            detector_data = det_map[:, k]
            for i, current_tau in enumerate(tau):
                if mode_eq == 1:
                    AD_bw = (detector_data >= current_tau).astype(int)
                else:
                    AD_bw = (detector_data > current_tau).astype(int)
                PD[i, k], PF[i, k] = cal_pdpf(AD_bw, GT_flat)

        # 归一化处理
        a21, a20 = PD[0, :].min(), PD.max()
        b21, b20 = PF[0, :].min(), PF.max()


        PDnor = (PD - a21) / (a20 - a21)
        PDnors.append(PDnor)
        PFnor = (PF - b21) / (b20 - b21)
        PFnors.append(PFnor)

    PDnors = np.array(PDnors)  # 转换为形状为(200, 5000)的数组
    PDnor = PDnors.mean(axis=0)  # 按列求均值，得到长度为5000的数组

    PFnors = np.array(PFnors)  # 转换为形状为(200, 5000)的数组
    PFnor = PFnors.mean(axis=0)  # 按列求均值，得到长度为5000的数组


    colors = ['b', 'y', (0.15, 0.5, 0.15), 'm', 'k',
              (0.15, 0.15, 0.5), (0.5, 0.5, 1), 'g', 'c', 'r']


    # ================== 3D ROC绘图 ==================
    # fig = plt.figure(figsize=(10, 8))
    # ax = fig.add_subplot(111, projection='3d')

    # 预定义颜色和线型

    # for k in range(num_map):
    #     ax.plot3D(PFnor[:, k], tau, PDnor[:, k],
    #               color=colors[k % 10], linewidth=2, label=detec_label[k])
    #
    # # 坐标轴设置
    # ax.set_xlabel('False alarm rate', labelpad=15, fontsize=12)
    # ax.set_ylabel(r'$\mu$', labelpad=15, fontsize=12)
    # ax.set_zlabel('Probability of detection', labelpad=15, fontsize=12)
    # ax.set_xscale('log')
    # ax.xaxis.set_major_locator(LogLocator(numticks=4))
    # ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0e}"))
    # ax.set_xticks([1e-4, 1e-3, 1e-2, 1e-1, 1])
    # ax.set_yticks(np.linspace(0, 1, 6))
    # ax.set_zticks(np.linspace(0, 1, 6))
    # ax.view_init(elev=20, azim=-45)
    #
    # plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1), fontsize=10)
    # if save_prefix:
    #     plt.savefig(f"{save_prefix}_3DROC.pdf", format='pdf', bbox_inches='tight')
    # plt.show()


    # ================== 2D PF-PD绘图 ==================
    plt.figure(figsize=(8, 6))
    for k in range(num_map):
        plt.plot(PFnor[:, k], PDnor[:, k], color=colors[k % 10], marker="o", markersize=4, linewidth=2, label=detec_label[k])

    plt.ylabel(r'Detection probability rate ', fontsize=15)
    plt.xlabel('False alarm rate', fontsize=15)
    plt.xscale('log')
    plt.xticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ['1e-4', '1e-3', '1e-2', '1e-1', '1'])
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid(True)
    plt.rcParams.update({'font.size': 15})
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_PF_PD.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # ================== 2D PD-tau绘图 ==================
    plt.figure(figsize=(8, 6))
    for k in range(num_map):
        plt.plot(tau, PDnor[:, k], color=colors[k % 10], linewidth=2, label=detec_label[k])

    plt.xlabel(r'$\mu$', fontsize=12)
    plt.ylabel('Normalized PD', fontsize=12)
    plt.xticks(np.linspace(0, 1, 11))
    plt.yticks(np.linspace(0, 1, 11))
    plt.grid(True)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_PD_tau.pdf", format='pdf', bbox_inches='tight')
    plt.show()

    # ================== 2D PF-tau绘图 ==================
    plt.figure(figsize=(8, 6))
    for k in range(num_map):
        plt.plot(tau, PFnor[:, k], color=colors[k % 10], linewidth=2, label=detec_label[k])

    plt.xlabel(r'$\mu$', fontsize=12)
    plt.ylabel('False alarm rate', fontsize=12)
    plt.yscale('log')
    plt.yticks([1e-4, 1e-3, 1e-2, 1e-1, 1], ['1e-4', '1e-3', '1e-2', '1e-1', '1'])
    plt.grid(True)
    plt.legend()
    if save_prefix:
        plt.savefig(f"{save_prefix}_PF_tau.pdf", format='pdf', bbox_inches='tight')
    plt.show()







