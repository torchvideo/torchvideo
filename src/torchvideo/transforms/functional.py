from typing import Sequence

import torch


def normalize(tensor: torch.Tensor, mean: Sequence, std: Sequence):
    """Normalize a tensor video with mean and standard deviation

    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.

    See :class:`~torchvideo.transforms.NormalizeVideo` for more details.

    Args:
        tensor: Tensor video of size (T, C, H, W) to be normalized.
        mean: Sequence of means for each channel
        std: Sequence of standard deviations for each channel.

    Returns:
        Tensor: Normalised Tensor video.

    """
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor
