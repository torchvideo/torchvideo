import torch

from .. import functional as VF
from .transform import Transform


class TimeToChannel(Transform[torch.Tensor, torch.Tensor, None]):
    r"""Combine time dimension into the channel dimension by reshaping video tensor of
    shape :math:`(C, T, H, W)` into :math:`(C \times T, H, W)`
    """

    def _gen_params(self, frames: torch.Tensor) -> None:
        return None

    def _transform(self, frames: torch.Tensor, params: None):
        return VF.time_to_channel(frames)

    def __repr__(self):
        return self.__class__.__name__ + "()"
