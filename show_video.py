import numpy as np
import cv2
from tqdm import tqdm  # 可选，用于显示进度条


def visualize_hyperspectral_video(data, output_path='output.mp4', bands=[0, 1, 2], fps=30):
    """
    将高光谱视频数据可视化为RGB视频

    参数：
    data (numpy.ndarray): 输入数据，形状为[T, C, H, W]
    output_path (str): 输出视频路径
    bands (list): 选择的波段索引列表，默认[0, 1, 2]
    fps (int): 输出视频的帧率
    """

    # 参数校验
    assert len(data.shape) == 4, "输入数据必须是4维数组[T, C, H, W]"
    T, C, H, W = data.shape
    assert len(bands) == 3, "必须选择3个波段"
    assert max(bands) < C, "波段索引超出有效范围"

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编解码器
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H))

    # 处理每一帧
    for t in tqdm(range(T), desc="Processing frames"):
        # 选择三个波段 [C, H, W] -> [H, W, 3]
        frame = data[t, bands, :, :].transpose(1, 2, 0)

        # 归一化处理
        normalized = np.zeros_like(frame, dtype=np.float32)
        for c in range(3):
            channel = frame[..., c]
            min_val = np.min(channel)
            max_val = np.max(channel)
            if max_val > min_val:  # 防止除以零
                normalized[..., c] = (channel - min_val) / (max_val - min_val)
            else:
                normalized[..., c] = channel

        # 转换为8-bit图像并调整通道顺序（OpenCV使用BGR）
        rgb_frame = (normalized * 255).astype(np.uint8)
        bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # 写入视频
        video_writer.write(bgr_frame)

    # 释放资源
    video_writer.release()
    print(f"视频已保存至：{output_path}")


def visualize_ground_truth(dataname, ground_truth, output_path='ground_truth.mp4', fps=30):
    """
    可视化并保存二值化真值视频，同时统计目标像素数

    参数：
    ground_truth (numpy.ndarray): 二值化真值数据，形状为[T, H, W]
    output_path (str): 输出视频路径
    fps (int): 输出视频的帧率
    """
    # 参数校验
    assert len(ground_truth.shape) == 3, "输入数据必须是3维数组[T, H, W]"
    T, H, W = ground_truth.shape

    # 创建视频写入对象（强制使用三通道）
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (W, H), isColor=True)

    # 初始化统计结果
    pixel_counts = []

    # 处理每一帧
    for t in tqdm(range(T), desc="Processing ground truth"):
        # 获取当前帧并转换为uint8

        gt_frame = ground_truth[t].astype(np.uint8)

        # 统计目标像素数（假设真值为0/1）
        if dataname == 'hsi_video':
            pixel_count = np.count_nonzero(gt_frame == 1)
            gt_frame = np.clip(gt_frame * 255, 0, 255).astype(np.uint8)  # 二值化扩展至0-255
        else:
            pixel_count = np.count_nonzero(gt_frame == 255)
        pixel_counts.append(pixel_count)

        # 转换为三通道BGR格式
        # gt_frame = np.clip(gt_frame * 255, 0, 255).astype(np.uint8)  # 二值化扩展至0-255
        gt_frame = gt_frame.astype(np.uint8)
        bgr_frame = cv2.cvtColor(gt_frame, cv2.COLOR_GRAY2BGR)

        # 写入视频
        video_writer.write(bgr_frame)

    # 释放资源
    video_writer.release()
    print(f"真值视频已保存至：{output_path}")
    print("\n目标像素统计结果:")
    [print(f"帧 {t}: {count} 像素") for t, count in enumerate(pixel_counts)]

    return pixel_counts


# 使用示例 -------------------------------------------------
if __name__ == "__main__":
    data_list = ['boat3', 'boat7', 'airplane2', 'airplane13', 'airplane15', 'hsi_video']
    data_dir = './data/'

    # 加载高光谱数据
    selected_index = 3
    image_file = data_dir + data_list[selected_index] + '.npy'
    dummy_data = np.load(image_file)

    # 生成高光谱视频
    visualize_hyperspectral_video(
        data=dummy_data,
        output_path='./video/' + data_list[selected_index] + '.mp4',
        bands=[5, 5, 5],
        fps=15
    )

    # 加载并处理真值数据（假设真值文件名为*_gt.npy）
    if data_list[selected_index] == 'hsi_video':
        gt_file = data_dir + 'ground_truth' + '.npy'
    else:
        gt_file = data_dir + data_list[selected_index] + '_gt.npy'

    ground_truth = np.load(gt_file)

    # 生成真值视频并统计像素
    visualize_ground_truth(dataname=data_list[selected_index],
        ground_truth=ground_truth,
        output_path='./video/' + data_list[selected_index] + '_gt.mp4',
        fps=15
    )