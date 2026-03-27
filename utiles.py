import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import seaborn as sns

from sklearn.metrics import roc_auc_score
import torch
import random

def map01(img):
    img_01 = (img - img.min())/(img.max() - img.min())
    return img_01

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class HSIProcessor:
    def __init__(self):
        self.band_stats = None

    def fit(self, data):
        # 计算各波段均值和方差
        self.band_stats = {
            'mean': data.mean(dim=(0, 1, 3, 4)),
            'std': data.std(dim=(0, 1, 3, 4))
        }

    def transform(self, data):
        # 标准化处理
        return (data - self.band_stats['mean']) / (self.band_stats['std'] + 1e-6)


class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.down2 = nn.AvgPool2d(2)
        self.down4 = nn.AvgPool2d(4)
        self.fuse_conv = nn.Conv2d(3, 1, 3, padding=1)

    def forward(self, x):
        x2 = self.down2(x)
        x4 = self.down4(x)
        return self.fuse_conv(
            torch.cat([
                F.interpolate(x4, x.shape[-2:], mode='bilinear'),
                F.interpolate(x2, x.shape[-2:], mode='bilinear'),
                x
            ], dim=1)
        )

import numpy as np
import matplotlib.pyplot as plt

def plot_anomaly(anomaly_map, title="Anomaly Detection Result", cmap="viridis", colorbar=True):
    """
    可视化异常检测结果。

    参数:
    - anomaly_map: numpy数组，异常检测结果（2D数组，每个像素值为异常得分）
    - title: 图像标题（默认为 "Anomaly Detection Result"）
    - cmap: 颜色映射（默认为 "viridis"）
    - colorbar: 是否显示颜色条（默认为 True）
    """
    plt.figure(figsize=(256, 256))  # 设置图像大小
    plt.imshow(anomaly_map, cmap=cmap)  # 显示异常检测结果
    if colorbar:
        plt.colorbar(label="Anomaly Score")  # 添加颜色条
    plt.title(title)  # 设置标题
    plt.axis("off")  # 关闭坐标轴
    plt.show()  # 显示图像

def save_heatmap(data, working_dir, save_width, save_height, dpi=300, file_name='heatmap.png'):
    save_path = os.path.join(working_dir, file_name + "_heatmap.png")
    plt.figure(figsize=(save_width, save_height))
    sns.heatmap(data, cmap="jet", cbar=False) #热图
    # plt.show()
    plt.axis('off')
    plt.savefig(save_path, dpi=dpi, bbox_inches = 'tight')
    plt.close()


def get_auc(HSI_old, HSI_new, gt):
    n_row, n_col, n_band = HSI_old.shape
    n_pixels = n_row * n_col

    img_olds = np.reshape(HSI_old, (n_pixels, n_band), order='F')
    img_news = np.reshape(HSI_new, (n_pixels, n_band), order='F')
    sub_img = img_olds - img_news

    detectmap = np.linalg.norm(sub_img, ord=2, axis=1, keepdims=True) ** 2
    detectmap = detectmap / n_band

    detectmap1 = np.linalg.norm(img_olds, ord=2, axis=1, keepdims=True) ** 2
    detectmap1 = detectmap1 / n_band

    detectmap2 = np.linalg.norm(img_news, ord=2, axis=1, keepdims=True) ** 2
    detectmap2 = detectmap2 / n_band

    # nomalization
    detectmap = map01(detectmap)
    detectmap1 = map01(detectmap1)
    detectmap2 = map01(detectmap2)
    # get auc
    label = np.reshape(gt, (n_pixels, 1), order='F')

    auc = roc_auc_score(label, detectmap)

    detectmap = np.reshape(detectmap, (n_row, n_col), order='F')
    detectmap1 = np.reshape(detectmap1, (n_row, n_col), order='F')
    detectmap2 = np.reshape(detectmap2, (n_row, n_col), order='F')
    return auc, detectmap, detectmap1, detectmap2


def TensorToHSI(img):

    HSI = img.cpu().data.numpy()
    return HSI

class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether or not to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):

    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether or not dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        return layer_output_list, last_state_list

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param