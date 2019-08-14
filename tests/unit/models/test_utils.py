import torch
from numpy.testing import assert_almost_equal
from torch.nn import functional as F

from torchvideo.models.utils import inflate_param


def make_image_and_boring_video(in_channels, time, height, width, dtype=torch.float32):
    input_image = torch.randn((in_channels, height, width), dtype=dtype)
    # transpose to go from CTHW to TCHW
    input_video = torch.stack([input_image.clone() for _ in range(time)]).transpose(
        0, 1
    )
    return input_image, input_video


def assert_boring_video_fixed_point_is_satisfied(
    kernel_2d, kernel_3d, input_image, input_video
):
    # N x C_out x H x W
    filtered_image = F.conv2d(input_image.unsqueeze(0), kernel_2d)[0].numpy()
    # N x C_out x T x H x W
    filtered_video = F.conv3d(input_video.unsqueeze(0), kernel_3d)[
        0, :, 0, :, :
    ].numpy()
    assert filtered_image.shape == filtered_video.shape
    assert_almost_equal(filtered_image, filtered_video, decimal=4)


class TestKernelInflation:
    def test_inflating_1x1_kernel(self):
        in_channels = 3
        out_channels = 4
        time, height, width = 5, 1, 1

        input_image, input_video = make_image_and_boring_video(
            in_channels, time, height, width
        )
        kernel_2d, kernel_3d = self.construct_kernels(
            out_channels, in_channels, time, height, width
        )

        kernel_3d = inflate_param(kernel_3d, kernel_2d)

        assert_boring_video_fixed_point_is_satisfied(
            kernel_2d, kernel_3d, input_image, input_video
        )

    def test_inflating_3x3_kernel(self):
        in_channels = 4
        out_channels = 5
        time, height, width = 6, 3, 3

        input_image, input_video = make_image_and_boring_video(
            in_channels, time, height, width
        )
        kernel_2d, kernel_3d = self.construct_kernels(
            out_channels, in_channels, time, height, width
        )

        kernel_3d = inflate_param(kernel_3d, kernel_2d)

        assert_boring_video_fixed_point_is_satisfied(
            kernel_2d, kernel_3d, input_image, input_video
        )

    def construct_kernels(self, out_channels, in_channels, time, height, width):
        kernel_2d = torch.randn((out_channels, in_channels, height, width))
        kernel_3d = torch.randn((out_channels, in_channels, time, height, width))
        return kernel_2d, kernel_3d
