import random
from typing import Iterator

from PIL.Image import Image
from torchvision.transforms import functional as F

from .types import PILVideo
from .transform import Transform


class RandomHorizontalFlipVideo(Transform[PILVideo, Iterator[Image], bool]):
    """Horizontally flip the given video (composed of PIL Images) randomly with a given
    probability :math:`p`.

    Args:
        p (float): probability of the image being flipped.
    """

    def __init__(self, p=0.5):
        self.p = p

    def _gen_params(self, frames: PILVideo) -> bool:
        if random.random() < self.p:
            return True
        else:
            return False

    def _transform(self, frames: PILVideo, params: bool) -> Iterator[Image]:
        flip = params
        for frame in frames:
            if flip:
                yield F.hflip(frame)
            else:
                yield frame

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(p={})".format(self.p)
