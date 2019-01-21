from typing import Callable

from PIL.Image import Image

from .types import PILVideo, PILVideoI
from .transform import StatelessTransform


class TimeApply(StatelessTransform[PILVideo, PILVideoI]):
    """Apply a PIL Image transform across time.

    See :std:doc:`torchvision/transforms` for suitable *deterministic*
    transforms to use with meta-transform.

    .. warning:: You should only use this with deterministic image transforms. Using a
       transform like :class:`torchvision.transforms.RandomCrop` will randomly crop
       each individual frame at a different location producing a nonsensical video.

    """

    def __init__(self, img_transform: Callable[[Image], Image]) -> None:
        """
        Args:
            img_transform: Image transform operating on a PIL Image.
        """
        self.img_transform = img_transform

    def _transform(self, frames: PILVideo, params: None) -> PILVideoI:
        for frame in frames:
            yield self.img_transform(frame)
