from typing import Sequence

import torch


def normalize(
    tensor: torch.Tensor,
    mean: Sequence,
    std: Sequence,
    channel_dim: int = 0,
    inplace: bool = False,
) -> torch.Tensor:
    r"""Channel-wise normalize a tensor video of shape :math:`(C, T, H, W)` with mean
    and standard deviation

    See :class:`~torchvideo.transforms.NormalizeVideo` for more details.

    Args:
        tensor: Tensor video of size :math:`(C, T, H, W)` to be normalized.
        mean: Sequence of means, :math:`M`, for each channel :math:`c`.
        std: Sequence of standard deviations, :math:`\Sigma`, for each channel
            :math:`c`.
        channel_dim: Index of channel dimension. 0 for ``'CTHW'`` tensors and ` for
            ``'TCHW'`` tensors.
        inplace: Whether to normalise the tensor without cloning or not.

    Returns:
        Channel-wise normalised tensor video,
        :math:`t'_c = \frac{t_c - M_c}{\Sigma_c}`

    """
    channel_count = tensor.shape[channel_dim]
    if len(mean) != len(std):
        raise ValueError(
            "Expected mean and std to be of the same length, but were "
            "{} and {} respectively".format(len(mean), len(std))
        )
    if len(mean) != channel_count:
        raise ValueError(
            "Expected mean to be the same length, {}, as the number of channels"
            "{}".format(len(mean), channel_count)
        )
    if not inplace:
        tensor = tensor.clone()

    statistic_shape = [1] * tensor.dim()
    statistic_shape[channel_dim] = -1
    mean_: torch.Tensor = torch.tensor(mean, dtype=torch.float32).view(*statistic_shape)
    std_: torch.Tensor = torch.tensor(std, dtype=torch.float32).view(*statistic_shape)
    tensor.sub_(mean_).div_(std_)  # type: ignore
    return tensor
