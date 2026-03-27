import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb
import torch
import cv2
# 假设光流数据已加载到flow_data，形状为(499, 2, 216, 409)
# 此处替换为实际数据加载代码，例如：
flow_data = np.load('flow_maps.npy')

# 选择帧索引（例如第0帧）
frame_idx = 0
flow_frame = flow_data[frame_idx]  # 形状变为(2, 216, 409)

from torchvision.utils import flow_to_image


def flow_to_image_torch(flow):
    flow = torch.from_numpy(flow)
    flow_im = flow_to_image(flow)
    # img = np.transpose(flow_im.numpy(), [1, 2, 0])
    # print(img.shape)
    return flow_im

plt.figure(figsize=(10, 6))
a = flow_to_image_torch(flow_frame)
img = np.transpose(a.numpy(), [1, 2, 0])
plt.imshow(img)
plt.show()

# def draw_flow(im, flow, step=40, norm=1):
#     # 在间隔分开的像素采样点处绘制光流
#     h, w = im.shape[:2]
#     y, x = np.mgrid[step / 2:h:step, step / 2:w:step].reshape(2, -1).astype(int)
#     if norm:
#         fx, fy = flow[y, x].T / abs(flow[y, x]).max() * step // 2
#     else:
#         fx, fy = flow[y, x].T  # / flow[y, x].max() * step // 2
#     # 创建线的终点
#     ex = x + fx
#     ey = y + fy
#     lines = np.vstack([x, y, ex, ey]).T.reshape(-1, 2, 2)
#     lines = lines.astype(np.uint32)
#     # 创建图像并绘制
#     vis = im.astype(np.uint8)  # cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
#     for (x1, y1), (x2, y2) in lines:
#         cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
#         cv2.circle(vis, (x1, y1), 2, (0, 0, 255), -1)
#     return vis
#
#
# image_file = './data/' + 'hsi_video' + '.npy'
# dummy_data = np.load(image_file)
# dummy_data = dummy_data[0, 0, :, :]
# flow_im4 = draw_flow(dummy_data, flow_frame, int(216/50))
#
#
#
#
#
#
# # 提取x和y方向的光流分量
# x_flow = flow_frame[0]
# y_flow = flow_frame[1]
#
# # 计算幅值和角度（弧度）
# magnitude = np.sqrt(x_flow**2 + y_flow**2)
# angle = np.arctan2(y_flow, x_flow)  # 范围[-π, π]
#
# # 归一化幅值到[0,1]
# if np.max(magnitude) > 0:
#     magnitude_normalized = magnitude / np.max(magnitude)
# else:
#     magnitude_normalized = magnitude  # 处理全零情况
#
# # 创建HSV图像
# h = (angle + np.pi) / (2 * np.pi)  # 转换到[0,1]
# s = np.ones_like(h)                # 饱和度设为1
# v = magnitude_normalized           # 亮度由幅值决定
# hsv_image = np.stack([h, s, v], axis=-1)  # 形状(216, 409, 3)
#
# # 转换为RGB图像
# rgb_image = hsv_to_rgb(hsv_image)
#
# # 显示结果
# plt.figure(figsize=(10, 6))
# plt.imshow(rgb_image)
# plt.axis('off')
# plt.title(f'光流可视化 - 第{frame_idx}帧')
# plt.show()