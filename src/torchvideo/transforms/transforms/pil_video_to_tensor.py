from typing import Iterator

import torch
from torchvision.transforms import functional as F

from .types import PILVideo
from .transform import Transform


class PILVideoToTensor(Transform[PILVideo, torch.Tensor, None]):
    r"""Convert a list of PIL Images to a tensor :math:`(C, T, H, W)`."""

    def __init__(self, rescale=True):
        """
        Args:
            rescale: Whether or not to rescale video from :math:`[0, 255]` to
                :math:`[0, 1]`. If ``False`` the tensor will be in range
                :math:`[0, 255]`.
        """
        self.rescale = rescale

    def _gen_params(self, frames: PILVideo) -> None:
        return None

    def _transform(self, frames: PILVideo, params: None) -> torch.Tensor:
        # PIL Images are in the format (H, W, C)
        # F.to_tensor converts (H, W, C) to (C, H, W)
        # Since we have a list of these tensors, when we stack them we get shape
        # (T, C, H, W), we want to swap T and C to get (C, T, H, W)
        if isinstance(frames, Iterator):
            frames = list(frames)
        tensor = torch.stack(list(map(F.to_tensor, frames))).transpose(1, 0)
        # torchvision.transforms.functional.to_tensor rescales by default, so if the
        # rescaling is disabled we effectively have to invert the operation.
        if not self.rescale:
            tensor *= 255
        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "()"
