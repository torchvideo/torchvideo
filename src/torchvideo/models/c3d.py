from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F


class C3D(nn.Module):
    def __init__(
        self,
        class_count: int,
        *,
        in_channels: int = 3,
        input_space: str = "RGB",
        fc6_dim: int = 4096,
        fc7_dim: int = 4096,
        dropout: float = 0,
    ):
        super().__init__()
        self.dropout = dropout
        self.in_channels = in_channels
        self.input_size = (in_channels, 16, 112, 112)
        self.input_order = "CTHW"
        self.input_space = input_space
        self.input_range = (0, 1)

        self.conv1a = self._make_conv(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv1a_bn = self._make_bn(64)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.conv2a = nn.Conv3d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        nn.init.normal_(self.conv2a.weight, mean=0, std=0.01)
        nn.init.zeros_(self.conv2a.bias)
        self.conv2a_bn = self._make_bn(128)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv3a = self._make_conv(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv3a_bn = self._make_bn(256)
        self.conv3b = self._make_conv(
            in_channels=256,
            out_channels=256,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv3b_bn = self._make_bn(256)
        self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv4a = self._make_conv(
            in_channels=256,
            out_channels=512,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv4a_bn = self._make_bn(512)
        self.conv4b = self._make_conv(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv4b_bn = self._make_bn(512)
        self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.conv5a = self._make_conv(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv5a_bn = self._make_bn(512)
        self.conv5b = self._make_conv(
            in_channels=512,
            out_channels=512,
            kernel_size=(3, 3, 3),
            stride=(1, 1, 1),
            padding=(1, 1, 1),
        )
        self.conv5b_bn = self._make_bn(512)
        self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

        self.fc6 = self._make_linear(512 * 3 * 3, fc6_dim)
        self.fc7 = self._make_linear(fc6_dim, fc7_dim)
        self.last_linear = nn.Linear(fc7_dim, class_count)

    def forward(self, x):
        assert x.dim() == 5
        assert x.shape[1] == self.in_channels

        x = F.relu(self.conv1a_bn(self.conv1a(x)))
        x = self.pool1(x)

        x = F.relu(self.conv2a_bn(self.conv2a(x)))
        x = self.pool2(x)

        x = F.relu(self.conv3a_bn(self.conv3a(x)))
        x = F.relu(self.conv3b_bn(self.conv3b(x)))
        x = self.pool3(x)

        x = F.relu(self.conv4a_bn(self.conv4a(x)))
        x = F.relu(self.conv4b_bn(self.conv4b(x)))
        x = self.pool4(x)

        x = F.relu(self.conv5a_bn(self.conv5a(x)))
        x = F.relu(self.conv5b_bn(self.conv5b(x)))
        x = torch.flatten(self.pool5(x), start_dim=1)

        x = F.dropout(F.relu(self.fc6(x)), self.dropout)
        x = F.dropout(F.relu(self.fc7(x)), self.dropout)
        x = self.last_linear(x)

        return x

    def _make_conv(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple[int, int, int],
        stride: Tuple[int, int, int],
        padding: Tuple[int, int, int],
    ) -> nn.Conv3d:
        conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        nn.init.normal_(conv.weight, mean=0, std=0.01)
        nn.init.zeros_(conv.bias)
        return conv

    def _make_bn(self, num_features):
        return nn.BatchNorm3d(num_features, eps=1e-3, momentum=0.1)

    def _make_linear(self, in_features, out_features) -> nn.Linear:
        linear = nn.Linear(in_features, out_features)
        nn.init.zeros_(linear.bias)
        nn.init.normal_(linear.weight, std=0.005)
        return linear
