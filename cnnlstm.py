import torch
import torch.nn as nn

class ConvLSTMCell3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size):
        super(ConvLSTMCell3D, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.padding = tuple(k // 2 for k in kernel_size)  # 自动计算 padding

        # 3D卷积层：输入+隐藏状态 -> 4个门
        self.conv = nn.Conv3d(
            in_channels=input_channels + hidden_channels,
            out_channels=4 * hidden_channels,
            kernel_size=kernel_size,
            padding=self.padding
        )

    def forward(self, x, h_prev, c_prev):
        # x: (B, C_in, 1, H, W) - 当前时间步的输入
        # h_prev: (B, C_hidden, 1, H, W) - 上一时间步的隐藏状态
        # c_prev: (B, C_hidden, 1, H, W) - 上一时间步的细胞状态

        combined = torch.cat([x, h_prev], dim=1)  # (B, C_in + C_hidden, 1, H, W)
        combined_conv = self.conv(combined)       # (B, 4*C_hidden, 1, H, W)

        # 分割为输入门、遗忘门、输出门和候选状态
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_channels, dim=1)

        # 计算门控信号
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        # 更新细胞状态和隐藏状态
        c_next = f * c_prev + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTM3D(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers, batch_first=True):
        super(ConvLSTM3D, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first

        # 多层 3D ConvLSTM 单元
        self.cells = nn.ModuleList()
        for i in range(num_layers):
            in_channels = input_channels if i == 0 else hidden_channels
            self.cells.append(ConvLSTMCell3D(in_channels, hidden_channels, kernel_size))

    def forward(self, x):
        # 输入形状处理
        if self.batch_first:
            x = x.permute(0, 2, 1, 3, 4)  # (B, C, T, H, W) -> (B, T, C, H, W)
        else:
            x = x.permute(1, 0, 2, 3, 4)

        B, T, C, H, W = x.size()
        device = x.device

        # 初始化隐藏状态和细胞状态（时间维度为1）
        h = [torch.zeros(B, self.hidden_channels, 1, H, W).to(device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_channels, 1, H, W).to(device) for _ in range(self.num_layers)]

        # 存储每个时间步的输出
        outputs = []

        # 逐时间步处理
        for t in range(T):
            layer_input = x[:, t, :, :, :].unsqueeze(2)  # (B, C, 1, H, W)

            # 逐层处理
            for i in range(self.num_layers):
                h[i], c[i] = self.cells[i](layer_input, h[i], c[i])
                layer_input = h[i]  # 当前层的输出作为下一层的输入

            # 记录最后一层的输出
            outputs.append(h[-1].squeeze(2))  # (B, C_hidden, H, W)

        # 合并时间步输出
        output = torch.stack(outputs, dim=2)  # (B, C_hidden, T, H, W)

        # 恢复原始维度顺序
        if self.batch_first:
            output = output.permute(0, 2, 1, 3, 4)  # (B, T, C_hidden, H, W)
        else:
            output = output.permute(1, 0, 2, 3, 4)

        return output


class ConvLSTM3DNetwork(nn.Module):
    def __init__(self, input_channels, hidden_channels, kernel_size, num_layers):
        super(ConvLSTM3DNetwork, self).__init__()
        self.input_channels = input_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.num_layers = num_layers

        # 3D ConvLSTM 层
        self.conv_lstm = ConvLSTM3D(
            input_channels, hidden_channels, kernel_size, num_layers, batch_first=True
        )

        # 最终 3D 卷积层：调整通道数
        self.final_conv = nn.Conv3d(
            in_channels=hidden_channels,
            out_channels=input_channels,
            kernel_size=1  # 1x1x1卷积，不改变空间维度
        )

    def forward(self, x):
        # 输入形状: (B, C, T, H, W)
        output = self.conv_lstm(x)        # (B, T, C_hidden, H, W)
        output = output.permute(0, 2, 1, 3, 4)  # (B, C_hidden, T, H, W)
        output = self.final_conv(output)  # (B, C, T, H, W)
        return output


# 示例用法
# input_channels = 25  # 光谱通道数
# hidden_channels = 32  # 隐藏层通道数
# kernel_size = (5, 3, 3)  # 3D卷积核 (T, H, W)
# num_layers = 1 # ConvLSTM层数
#
# model = ConvLSTM3DNetwork(input_channels, hidden_channels, kernel_size, num_layers)
#
# # 输入数据
# B, C, T, H, W = 1, 25, 5, 256, 256
# x = torch.randn(B, C, T, H, W)
#
# # 前向传播
# output = model(x)
# print(output.shape)  # 输出形状应为 (B, C, T, H, W)