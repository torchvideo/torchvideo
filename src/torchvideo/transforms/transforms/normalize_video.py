import numbers
from typing import Union, Sequence

import torch

from .. import functional as VF
from .transform import Transform


class NormalizeVideo(Transform[torch.Tensor, torch.Tensor, None]):
    r"""

    Normalise ``torch.*Tensor`` :math:`t` given mean:
    :math:`M = (\mu_1, \ldots, \mu_n)`
    and std:
    :math:`\Sigma = (\sigma_1, \ldots, \sigma_n)`:
    :math:`t'_c = \frac{t_c - M_c}{\Sigma_c}`

    Args:
        mean: Sequence of means for each channel, or a single mean applying to all
            channels.
        std: Sequence of standard deviations for each channel, or a single standard
            deviation applying to all channels.
        channel_dim: Index of channel dimension. 0 for ``'CTHW'`` tensors and ` for
            ``'TCHW'`` tensors.
        inplace: Whether or not to perform the operation in place without allocating
            a new tensor.
    """

    def __init__(
        self,
        mean: Union[Sequence[numbers.Number], numbers.Number],
        std: Union[Sequence[numbers.Number], numbers.Number],
        channel_dim: int = 0,
        inplace: bool = False,
    ):
        self.mean = mean
        self.std = std
        self.inplace = inplace
        self.channel_dim = channel_dim
        if isinstance(std, numbers.Number) and std == 0:
            raise ValueError("std cannot be 0")
        if isinstance(std, Sequence) and any([s == 0 for s in std]):
            raise ValueError("std {} contained 0 value, cannot be 0".format(std))

    def _gen_params(self, frames: torch.Tensor) -> None:
        return None

    def __repr__(self) -> str:
        return (
            self.__class__.__name__
            + "(mean={mean!r}, std={std!r}, channel_dim={channel_dim!r})".format(
                mean=self.mean, std=self.std, channel_dim=self.channel_dim
            )
        )

    def _transform(self, frames: torch.Tensor, params: None) -> torch.Tensor:
        channel_count = frames.shape[self.channel_dim]
        mean = self._broadcast_to_seq(self.mean, channel_count)
        std = self._broadcast_to_seq(self.std, channel_count)
        return VF.normalize(
            frames, mean, std, inplace=self.inplace, channel_dim=self.channel_dim
        )

    @staticmethod
    def _broadcast_to_seq(
        x: Union[numbers.Number, Sequence], channel_count: int
    ) -> Sequence[numbers.Number]:
        if isinstance(x, numbers.Number):
            return [x] * channel_count
        # else assume already a sequence
        return x
