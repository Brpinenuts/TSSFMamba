import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import label, center_of_mass
import pandas as pd
import warnings
import time
from pylab import mpl

from matplotlib.font_manager import FontProperties
# 设置显示中文字体
mpl.rcParams['font.sans-serif'] = ['Noto Sans CJK']
plt.rcParams['axes.unicode_minus'] = False
data_list = ['boat3', 'boat7',  'airplane2', 'airplane13', 'airplane15', 'hsi_video', 'Terrain100', 'SHARE2012100', 'SanDiego100']
# data_list = ['SHARE2012100']
data_dir = './data/'

# 忽略特定警告
warnings.filterwarnings("ignore", category=RuntimeWarning)

def calculate_centroids(binary_sequence):
    """
    计算多帧二值图像序列中每帧目标的质心位置
    添加边界检查以确保索引安全

    参数:
    binary_sequence: 3D NumPy数组, shape=(帧数, 高度, 宽度)

    返回:
    centroids_list: 每帧中每个目标的质心列表
    num_objects_list: 每帧中的目标数量列表
    """
    centroids_list = []
    num_objects_list = []

    for i, frame in enumerate(binary_sequence):
        try:
            # 连通区域标记
            labeled_frame, num_objects = label(frame)
            num_objects_list.append(num_objects)

            if num_objects > 0:
                # 计算每个目标的质心
                centroids = center_of_mass(frame, labeled_frame, range(1, num_objects + 1))
                centroids_list.append(centroids)
            else:
                centroids_list.append([])
        except Exception as e:
            print(f"计算第{i}帧质心时发生错误: {str(e)}")
            centroids_list.append([])
            num_objects_list.append(0)

    return centroids_list, num_objects_list


def calculate_displacements(centroids_list, max_displacement_threshold=20.0):
    """
    计算相邻帧之间质心的位移
    添加位移约束，当位移超过阈值时视为异常并剔除

    参数:
    centroids_list: 每帧中目标的质心列表
    max_displacement_threshold: 最大允许位移阈值（像素）

    返回:
    displacements: 相邻帧之间每个目标的位移列表（已过滤超过阈值的位移）
    displacement_stats: 包含所有位移统计信息的字典
    valid_displacements: 有效位移列表（未超过阈值）
    removed_displacements: 被剔除的位移列表（超过阈值）
    """
    displacements = []
    valid_displacements = []  # 有效位移
    removed_displacements = []  # 被剔除的位移
    all_distances = []  # 所有有效位移距离

    for i in range(1, len(centroids_list)):
        frame1_centroids = centroids_list[i - 1]
        frame2_centroids = centroids_list[i]

        frame_displacements = []

        # 尝试匹配目标 - 这里简化处理，假设目标数量不变且位置相邻
        min_objs = min(len(frame1_centroids), len(frame2_centroids))

        for j in range(min_objs):
            try:
                # 计算欧氏距离
                dx = frame2_centroids[j][1] - frame1_centroids[j][1]
                dy = frame2_centroids[j][0] - frame1_centroids[j][0]
                distance = np.sqrt(dx ** 2 + dy ** 2)

                # 检查是否超过位移阈值
                if distance > max_displacement_threshold:
                    # 标记为异常位移
                    displacement_info = {
                        'from_frame': i - 1,
                        'to_frame': i,
                        'object_id': j,
                        'dx': dx,
                        'dy': dy,
                        'distance': distance,
                        'centroid_from': frame1_centroids[j],
                        'centroid_to': frame2_centroids[j],
                        'status': 'removed (over threshold)'
                    }
                    removed_displacements.append(displacement_info)
                    continue

                # 有效位移
                displacement_info = {
                    'from_frame': i - 1,
                    'to_frame': i,
                    'object_id': j,
                    'dx': dx,
                    'dy': dy,
                    'distance': distance,
                    'centroid_from': frame1_centroids[j],
                    'centroid_to': frame2_centroids[j],
                    'status': 'valid'
                }

                frame_displacements.append(displacement_info)
                valid_displacements.append(displacement_info)
                all_distances.append(distance)

            except IndexError:
                print(f"索引错误: 帧{i}目标{j}")
            except Exception as e:
                print(f"计算帧{i}目标{j}位移时发生错误: {str(e)}")

        displacements.append(frame_displacements)

    # 计算位移统计信息
    displacement_stats = calculate_displacement_stats(displacements, all_distances, removed_displacements)

    return displacements, displacement_stats, valid_displacements, removed_displacements


def calculate_displacement_stats(displacements, all_distances, removed_displacements):
    """
    计算位移的统计信息，包括最大值、最小值等
    添加被剔除位移的统计

    参数:
    displacements: 位移数据列表
    all_distances: 所有有效位移距离列表
    removed_displacements: 被剔除的位移列表

    返回:
    包含各种位移统计信息的字典
    """
    stats = {
        'max_displacement': 0,
        'min_displacement': 0,
        'max_frame_pair': (0, 0),
        'min_frame_pair': (0, 0),
        'max_object_id': -1,
        'min_object_id': -1,
        'max_movement_vector': (0, 0),
        'min_movement_vector': (0, 0),
        'avg_displacement': 0,
        'std_displacement': 0,
        'total_movements': 0,
        'removed_count': len(removed_displacements),
        'removed_max_distance': 0,
        'removed_min_distance': 0,
        'max_displacement_threshold': 0
    }

    if not all_distances or len(all_distances) == 0:
        return stats

    # 有效位移的统计
    stats['total_movements'] = len(all_distances)
    stats['max_displacement'] = max(all_distances) if all_distances else 0
    stats['min_displacement'] = min(all_distances) if all_distances else 0
    stats['avg_displacement'] = np.mean(all_distances) if all_distances else 0
    stats['std_displacement'] = np.std(all_distances) if all_distances and len(all_distances) > 1 else 0

    # 被剔除位移的统计
    if removed_displacements:
        removed_distances = [r['distance'] for r in removed_displacements]
        stats['removed_max_distance'] = max(removed_distances) if removed_distances else 0
        stats['removed_min_distance'] = min(removed_distances) if removed_distances else 0
        stats['max_displacement_threshold'] = max(removed_distances) if removed_distances else 0

    # 查找最大和最小位移的具体信息
    max_info = None
    min_info = None

    for frame_displacements in displacements:
        for displacement in frame_displacements:
            if displacement['status'] == 'valid':
                if max_info is None or displacement['distance'] > max_info['distance']:
                    max_info = displacement
                if min_info is None or displacement['distance'] < min_info['distance']:
                    min_info = displacement

    if max_info:
        stats['max_frame_pair'] = (max_info['from_frame'], max_info['to_frame'])
        stats['max_object_id'] = max_info['object_id']
        stats['max_movement_vector'] = (max_info['dx'], max_info['dy'])

    if min_info:
        stats['min_frame_pair'] = (min_info['from_frame'], min_info['to_frame'])
        stats['min_object_id'] = min_info['object_id']
        stats['min_movement_vector'] = (min_info['dx'], min_info['dy'])

    return stats


def plot_frame_with_centroids(frame, centroids, frame_idx, output_dir):
    """
    绘制单帧图像并标记质心位置

    参数:
    frame: 单帧二值图像
    centroids: 当前帧的质心列表
    frame_idx: 帧索引
    output_dir: 输出目录
    """
    try:
        plt.figure(figsize=(8, 8))
        plt.imshow(frame, cmap='gray')
        plt.title(f'Frame {frame_idx} - Targets: {len(centroids)}')

        # 标记质心
        for i, (y, x) in enumerate(centroids):
            plt.plot(x, y, 'ro')  # 标记质心位置
            plt.text(x + 1, y + 1, f'Obj{i}\n({x:.1f},{y:.1f})', color='red', fontsize=8)

        # 保存图像
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, f'frame_{frame_idx:03d}.png'))
        plt.close()
    except Exception as e:
        print(f"绘制第{frame_idx}帧时发生错误: {str(e)}")


def save_results(centroids_list, displacements, displacement_stats, valid_displacements, removed_displacements,
                 output_dir):
    """
    保存分析结果到CSV文件
    添加被剔除位移的保存

    参数:
    centroids_list: 每帧的质心列表
    displacements: 所有位移数据
    displacement_stats: 位移统计信息
    valid_displacements: 有效位移
    removed_displacements: 被剔除的位移
    output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)

    # 保存质心数据
    centroids_data = []
    for frame_idx, centroids in enumerate(centroids_list):
        for obj_idx, centroid in enumerate(centroids):
            if centroid:
                try:
                    y, x = centroid
                    centroids_data.append({
                        'frame': frame_idx,
                        'object_id': obj_idx,
                        'x': x,
                        'y': y
                    })
                except Exception:
                    continue

    pd.DataFrame(centroids_data).to_csv(os.path.join(output_dir, 'centroids.csv'), index=False)

    # 保存所有位移数据（包含状态）
    displacements_data = []
    for frame_displacements in displacements:
        for displacement in frame_displacements:
            displacements_data.append({
                'from_frame': displacement['from_frame'],
                'to_frame': displacement['to_frame'],
                'object_id': displacement['object_id'],
                'dx': displacement['dx'],
                'dy': displacement['dy'],
                'distance': displacement['distance'],
                'status': displacement['status'],
                'centroid_from_x': displacement['centroid_from'][1] if 'centroid_from' in displacement else 0,
                'centroid_from_y': displacement['centroid_from'][0] if 'centroid_from' in displacement else 0,
                'centroid_to_x': displacement['centroid_to'][1] if 'centroid_to' in displacement else 0,
                'centroid_to_y': displacement['centroid_to'][0] if 'centroid_to' in displacement else 0
            })

    pd.DataFrame(displacements_data).to_csv(os.path.join(output_dir, 'all_displacements.csv'), index=False)

    # 保存有效位移数据
    valid_data = []
    for displacement in valid_displacements:
        valid_data.append({
            'from_frame': displacement['from_frame'],
            'to_frame': displacement['to_frame'],
            'object_id': displacement['object_id'],
            'dx': displacement['dx'],
            'dy': displacement['dy'],
            'distance': displacement['distance'],
            'centroid_from_x': displacement['centroid_from'][1],
            'centroid_from_y': displacement['centroid_from'][0],
            'centroid_to_x': displacement['centroid_to'][1],
            'centroid_to_y': displacement['centroid_to'][0]
        })

    pd.DataFrame(valid_data).to_csv(os.path.join(output_dir, 'valid_displacements.csv'), index=False)

    # 保存被剔除位移数据
    removed_data = []
    for displacement in removed_displacements:
        removed_data.append({
            'from_frame': displacement['from_frame'],
            'to_frame': displacement['to_frame'],
            'object_id': displacement['object_id'],
            'dx': displacement['dx'],
            'dy': displacement['dy'],
            'distance': displacement['distance'],
            'centroid_from_x': displacement['centroid_from'][1],
            'centroid_from_y': displacement['centroid_from'][0],
            'centroid_to_x': displacement['centroid_to'][1],
            'centroid_to_y': displacement['centroid_to'][0]
        })

    pd.DataFrame(removed_data).to_csv(os.path.join(output_dir, 'removed_displacements.csv'), index=False)

    # 保存位移统计信息到文本文件
    with open(os.path.join(output_dir, 'displacement_stats.txt'), 'w') as f:
        f.write("位移统计摘要\n")
        f.write("================\n\n")
        f.write(
            f"最大允许位移阈值: {20.0 if 'max_displacement_threshold' not in displacement_stats else displacement_stats['max_displacement_threshold']:.2f} 像素\n")
        f.write(f"总位移次数: {displacement_stats['total_movements']}\n")
        f.write(
            f"有效位移平均距离: {displacement_stats['avg_displacement']:.2f} ± {displacement_stats['std_displacement']:.2f} 像素\n")
        f.write(f"被剔除的位移次数: {displacement_stats['removed_count']}\n")

        if displacement_stats['removed_count'] > 0:
            f.write(f"被剔除位移中的最大距离: {displacement_stats['removed_max_distance']:.2f} 像素\n")
            f.write(f"被剔除位移中的最小距离: {displacement_stats['removed_min_distance']:.2f} 像素\n\n")

        f.write("\n最大位移信息:\n")
        f.write(f"  距离: {displacement_stats['max_displacement']:.2f} 像素\n")
        f.write(
            f"  发生位置: 帧 {displacement_stats['max_frame_pair'][0]} 到帧 {displacement_stats['max_frame_pair'][1]}\n")
        f.write(f"  目标ID: {displacement_stats['max_object_id']}\n")
        f.write(
            f"  位移向量: (dx={displacement_stats['max_movement_vector'][0]:.2f}, dy={displacement_stats['max_movement_vector'][1]:.2f})\n\n")

        f.write("最小位移信息:\n")
        f.write(f"  距离: {displacement_stats['min_displacement']:.2f} 像素\n")
        f.write(
            f"  发生位置: 帧 {displacement_stats['min_frame_pair'][0]} 到帧 {displacement_stats['min_frame_pair'][1]}\n")
        f.write(f"  目标ID: {displacement_stats['min_object_id']}\n")
        f.write(
            f"  位移向量: (dx={displacement_stats['min_movement_vector'][0]:.2f}, dy={displacement_stats['min_movement_vector'][1]:.2f})\n")

        # 添加位移分布直方图
        distances = [d['distance'] for d in valid_displacements]
        if distances:
            plt.figure(figsize=(10, 6))
            plt.hist(distances, bins=20, color='skyblue', edgecolor='black')
            plt.axvline(x=20, color='r', linestyle='--', label='位移阈值 (20像素)')
            plt.xlabel('位移距离 (像素)')
            plt.ylabel('出现次数')
            plt.title('有效位移距离分布直方图')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'displacement_histogram.png'))
            plt.close()

            f.write("\n位移距离分布直方图已保存为 displacement_histogram.png")


def visualize_movement(centroids_list, displacements, displacement_stats, output_dir):
    """
    可视化目标的运动轨迹
    用不同颜色区分正常位移和被剔除位移

    参数:
    centroids_list: 每帧的质心列表
    displacements: 位移数据
    displacement_stats: 位移统计信息
    output_dir: 输出目录
    """
    try:
        # 创建一个新的图形用于绘制所有轨迹
        plt.figure(figsize=(14, 12))

        # 为每个目标绘制运动轨迹
        max_objs = max(len(centroids) for centroids in centroids_list) if centroids_list else 0
        colors = plt.cm.viridis(np.linspace(0, 1, max_objs)) if max_objs > 0 else []

        # 绘制所有有效位移
        valid_arrows = []
        removed_arrows = []

        for obj_idx in range(max_objs):
            x_vals = []
            y_vals = []
            frames = []

            for frame_idx, centroids in enumerate(centroids_list):
                if centroids and obj_idx < len(centroids):
                    try:
                        y, x = centroids[obj_idx]
                        x_vals.append(x)
                        y_vals.append(y)
                        frames.append(frame_idx)
                    except Exception:
                        continue

            if x_vals:  # 确保有数据
                # 绘制轨迹
                plt.plot(x_vals, y_vals, '-o', color=colors[obj_idx], alpha=0.7,
                         label=f'Object {obj_idx}' if max_objs <= 20 else "")

                # 标记起点和终点
                plt.scatter(x_vals[0], y_vals[0], marker='s', s=100,
                            color=colors[obj_idx], edgecolors='white')
                plt.scatter(x_vals[-1], y_vals[-1], marker='*', s=200,
                            color=colors[obj_idx], edgecolors='white')

        # 添加位移箭头
        for frame_displacements in displacements:
            for displacement in frame_displacements:
                try:
                    if 'centroid_from' in displacement and 'centroid_to' in displacement:
                        x1, y1 = displacement['centroid_from'][1], displacement['centroid_from'][0]
                        x2, y2 = displacement['centroid_to'][1], displacement['centroid_to'][0]
                        dx, dy = displacement['dx'], displacement['dy']

                        # 根据状态选择颜色
                        if displacement['status'] == 'valid':
                            arrow = plt.arrow(x1, y1, dx, dy,
                                              head_width=0.5, head_length=1,
                                              color=colors[displacement['object_id']],
                                              length_includes_head=True, alpha=0.7)
                            valid_arrows.append(arrow)
                        else:
                            arrow = plt.arrow(x1, y1, dx, dy,
                                              head_width=0.5, head_length=1,
                                              color='red', linestyle='dashed', alpha=0.7,
                                              length_includes_head=True)
                            removed_arrows.append(arrow)
                except Exception:
                    continue

        # 添加图例
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', markersize=10, label='起点'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='gray', markersize=15, label='终点'),
            Line2D([0], [0], color='blue', lw=2, label='有效位移'),
            Line2D([0], [0], color='red', linestyle='dashed', lw=2, label='剔除位移 (>20像素)')
        ]
        plt.legend(handles=legend_elements, loc='best')

        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        plt.title('目标运动轨迹 (剔除超过20像素的位移)')
        plt.grid(True, alpha=0.3)

        # 保存轨迹图
        plt.savefig(os.path.join(output_dir, 'trajectories.png'))
        plt.close()
    except Exception as e:
        print(f"绘制轨迹图时发生错误: {str(e)}")


def plot_displacement_timeseries(valid_displacements, removed_displacements, displacement_stats, output_dir):
    """
    绘制位移距离的时间序列
    区分显示有效位移和被剔除位移

    参数:
    valid_displacements: 有效位移数据
    removed_displacements: 被剔除位移数据
    displacement_stats: 位移统计信息
    output_dir: 输出目录
    """
    try:
        plt.figure(figsize=(14, 8))

        # 创建时间序列数据
        valid_frame_pairs = [(d['from_frame'], d['to_frame']) for d in valid_displacements]
        valid_distances = [d['distance'] for d in valid_displacements]
        valid_object_ids = [d['object_id'] for d in valid_displacements]

        # 为每个目标分配颜色
        unique_object_ids = list(set(valid_object_ids))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_object_ids)))
        color_map = {obj_id: colors[i % len(colors)] for i, obj_id in enumerate(unique_object_ids)}

        # 绘制有效位移
        for i, (obj_id, distance) in enumerate(zip(valid_object_ids, valid_distances)):
            plt.scatter(i, distance, color=color_map[obj_id], s=60, alpha=0.8, label=f'目标 {obj_id}' if i < 20 else "")

        # 绘制被剔除位移
        if removed_displacements:
            removed_indices = list(range(len(valid_distances), len(valid_distances) + len(removed_displacements)))
            removed_distances = [d['distance'] for d in removed_displacements]
            plt.scatter(removed_indices, removed_distances, color='red', marker='x', s=100, label='被剔除位移')

        # 标记平均值线
        if valid_distances:
            plt.axhline(y=displacement_stats['avg_displacement'], color='gray', linestyle='-',
                        label=f'平均位移 ({displacement_stats["avg_displacement"]:.2f}像素)')

        # 标记阈值线
        plt.axhline(y=20, color='r', linestyle='--', label='位移阈值 (20像素)')

        # 设置图表属性
        plt.xlabel('位移事件索引')
        plt.ylabel('位移距离 (像素)')
        plt.title('位移时间序列 (剔除了超过20像素的位移)')
        plt.grid(True, alpha=0.3)
        plt.legend()

        # 保存时间序列图
        plt.savefig(os.path.join(output_dir, 'displacement_timeseries.png'))
        plt.close()
    except Exception as e:
        print(f"绘制位移时间序列时发生错误: {str(e)}")


def analyze_image_sequence(binary_sequence, output_dir, max_displacement_threshold=20.0):
    """
    分析图像序列的主要函数
    包含计时和错误处理

    参数:
    binary_sequence: 3D NumPy数组, shape=(帧数, 高度, 宽度)
    output_dir: 输出目录
    max_displacement_threshold: 位移阈值，超过此值的位移将被剔除
    """
    start_time = time.time()

    try:
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        print(f"开始分析序列，共{len(binary_sequence)}帧...")

        # 1. 计算质心
        centroids_list, num_objects_list = calculate_centroids(binary_sequence)

        # 2. 计算位移（应用约束）
        displacements, displacement_stats, valid_displacements, removed_displacements = calculate_displacements(
            centroids_list,
            max_displacement_threshold=max_displacement_threshold
        )

        # 3. 保存结果
        save_results(
            centroids_list,
            displacements,
            displacement_stats,
            valid_displacements,
            removed_displacements,
            output_dir
        )

        # 4. 可视化结果
        # 为每一帧生成带有质心标记的图像
        print("生成帧图像...")
        for frame_idx, (frame, centroids) in enumerate(zip(binary_sequence, centroids_list)):
            plot_frame_with_centroids(
                frame, centroids, frame_idx,
                os.path.join(output_dir, "frames")
            )

        # 生成运动轨迹图
        print("生成轨迹图...")
        visualize_movement(
            centroids_list,
            displacements,
            displacement_stats,
            output_dir
        )

        # 生成位移时间序列图
        print("生成时间序列图...")
        plot_displacement_timeseries(
            valid_displacements,
            removed_displacements,
            displacement_stats,
            output_dir
        )

        # 5. 生成摘要报告
        analysis_time = time.time() - start_time

        with open(os.path.join(output_dir, 'analysis_report.txt'), 'w') as f:
            f.write("目标追踪分析报告\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"分析时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"分析用时: {analysis_time:.2f}秒\n")
            f.write(f"图像帧数: {len(binary_sequence)}\n")
            f.write(f"图像尺寸: {binary_sequence.shape[1]}x{binary_sequence.shape[2]}\n")
            f.write(f"平均每帧目标数: {np.mean(num_objects_list):.2f}\n\n")

            f.write("位移分析摘要:\n")
            f.write(f"- 位移阈值: {max_displacement_threshold}像素\n")
            f.write(f"- 有效位移次数: {displacement_stats['total_movements']}\n")
            f.write(f"- 被剔除位移次数: {displacement_stats['removed_count']}\n")

            if valid_displacements:
                f.write(
                    f"- 最大有效位移: {displacement_stats['max_displacement']:.2f}像素 (目标{displacement_stats['max_object_id']}, 帧{displacement_stats['max_frame_pair'][0]}-{displacement_stats['max_frame_pair'][1]})\n")
                f.write(
                    f"- 最小有效位移: {displacement_stats['min_displacement']:.2f}像素 (目标{displacement_stats['min_object_id']}, 帧{displacement_stats['min_frame_pair'][0]}-{displacement_stats['min_frame_pair'][1]})\n")
                f.write(
                    f"- 平均有效位移: {displacement_stats['avg_displacement']:.2f}±{displacement_stats['std_displacement']:.2f}像素\n")

            if removed_displacements:
                f.write(f"- 最大被剔除位移: {max([d['distance'] for d in removed_displacements]):.2f}像素\n")
                f.write(f"- 最小被剔除位移: {min([d['distance'] for d in removed_displacements]):.2f}像素\n")

            f.write("\n输出文件:\n")
            f.write("- centroids.csv: 所有质心位置\n")
            f.write("- all_displacements.csv: 所有位移数据（含状态）\n")
            f.write("- valid_displacements.csv: 有效位移数据\n")
            f.write("- removed_displacements.csv: 被剔除位移数据\n")
            f.write("- displacement_stats.txt: 位移统计摘要\n")
            f.write("- trajectories.png: 目标运动轨迹图\n")
            f.write("- displacement_timeseries.png: 位移时间序列图\n")
            f.write("- displacement_histogram.png: 位移直方图\n")
            f.write("- frames/ 目录: 带质心标记的帧图像\n")

        # 打印摘要信息
        print(f"\n分析完成! 结果保存在: {output_dir}")
        print(f"分析用时: {analysis_time:.2f}秒")
        print(f"图像帧数: {len(binary_sequence)}")
        print(f"平均每帧目标数: {np.mean(num_objects_list):.2f}")
        print(f"有效位移次数: {displacement_stats['total_movements']}")
        print(f"被剔除位移次数: {displacement_stats['removed_count']}")

        if displacement_stats['removed_count'] > 0:
            max_removed = max([d['distance'] for d in removed_displacements])
            min_removed = min([d['distance'] for d in removed_displacements])
            print(f"被剔除位移范围: {min_removed:.2f} - {max_removed:.2f}像素")

        if displacement_stats['total_movements'] > 0:
            print(
                f"最大有效位移: {displacement_stats['max_displacement']:.2f}像素 (目标{displacement_stats['max_object_id']})")
            print(
                f"最小有效位移: {displacement_stats['min_displacement']:.2f}像素 (目标{displacement_stats['min_object_id']})")
            print(
                f"平均有效位移: {displacement_stats['avg_displacement']:.2f}±{displacement_stats['std_displacement']:.2f}像素")

        # 显示轨迹图和直方图
        fig, ax = plt.subplots(1, 2, figsize=(18, 8))

        try:
            img1 = plt.imread(os.path.join(output_dir, 'trajectories.png'))
            ax[0].imshow(img1)
            ax[0].axis('off')
            ax[0].set_title('目标运动轨迹')
        except:
            ax[0].text(0.5, 0.5, '轨迹图不可用', ha='center', va='center')

        try:
            img2 = plt.imread(os.path.join(output_dir, 'displacement_histogram.png'))
            ax[1].imshow(img2)
            ax[1].axis('off')
            ax[1].set_title('位移分布直方图')
        except:
            ax[1].text(0.5, 0.5, '直方图不可用', ha='center', va='center')

        plt.tight_layout()
        # plt.show()

    except Exception as e:
        print(f"分析过程中发生严重错误: {str(e)}")
        import traceback
        traceback.print_exc()

def gui(anomaly_map):
    num = anomaly_map.shape[0]
    anomaly_maps = np.zeros_like(anomaly_map)
    for i in range(num):
        anomaly_maps[i, :, :] = ((anomaly_map[i, :, :] - anomaly_map[i, :, :].min()) / (anomaly_map[i, :, :].max() - anomaly_map[i, :, :].min()))
    return anomaly_maps
for data in data_list:
    image_file = data_dir + data + '.npy'
    raw_data = np.load(image_file)
    num, channel, h, w = raw_data.shape
    if data == 'hsi_video':
        gt_file = data_dir + 'ground_truth' + '.npy'
    else:
        gt_file = data_dir + data + '_gt' + '.npy'
    true_labels = np.load(gt_file)
    raw_data = raw_data.astype(np.float32)
    true_labels = true_labels.astype(np.float32)
    print(data)
    print(f"\nHSIS frame: {num:.4f}")
    print(f"\nband number: {channel:.4f}")
    print(f"\nheight: {h:.4f}")
    print(f"\nwidth: {w:.4f}")

    # print(num)
    rate = 0
    for i in range(num):
        rate = rate + np.count_nonzero(true_labels[i, :, :])
    rates = rate / (num * h * w) * 100
    print(f"\nanomaly rate (%): {rates:.4f}")
    print(f"\nmean pixels (%): {rate/num:.4f}")

    frames = num
    binary_sequence = gui(true_labels)
    # 3. 设置输出目录
    output_dir = "object_tracking_results/"
    os.makedirs(output_dir, exist_ok=True)

    analyze_image_sequence(
        binary_sequence,
        output_dir,
        max_displacement_threshold=20.0
    )
