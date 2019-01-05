from typing import Sequence

import torch


def normalize(tensor: torch.Tensor, mean: Sequence, std: Sequence):
    """Normalize a tensor video with mean and standard deviation

    .. note::
        This transform acts in-place, i.e., it mutates the input tensor.

    See :class:`~torchvideo.transforms.NormalizeVideo` for more details.

    Args:
        tensor: Tensor video of size :math:`(C, T, H, W)` to be normalized.
        mean: Sequence of means for each channel :math:`c`
        std: Sequence of standard deviations for each channel :math:`c`.

    Returns:
        Tensor: Normalised Tensor video.

    """
    for t, m, s in zip(tensor, mean, std):
        t.sub_(m).div_(s)
    return tensor


def time_to_channel(tensor: torch.Tensor):
    """Reshape video tensor of shape :math:`(C, T, H, W)` into
    :math:`(C \times T, H, W)`

    Args:
        tensor: Tensor video of size :math:`(C, T, H, W)`

    Returns:
        Tensor of shape :math:`(T \times C, H, W)`

    """
    tensor_ndim = len(tensor.size())
    if tensor_ndim != 4:
        raise ValueError("Expected 4D tensor but was {}D".format(tensor_ndim))
    h_w_shape = tensor.shape[-2:]
    return tensor.reshape((-1, *h_w_shape))
