from typing import Tuple, Union

import PIL
from PIL.Image import Image
from torchvision.transforms import transforms as tv, functional as F

from .transform import FramesAndParams, Transform
from .types import PILVideo, PILVideoI
from .internal import canonicalize_size, to_iter, peek_iter


class RandomResizedCropVideo(Transform[PILVideo, PILVideoI, Tuple[int, int, int, int]]):
    """Crop the given video (composed of PIL Images) to random size and aspect ratio.

    A crop of random scale (default: :math:`[0.08, 1.0]`) of the original size and a
    random scale (default: :math:`[3/4, 4/3]`) of the original aspect ratio is
    made. This crop is finally resized to given size. This is popularly used to train
    the Inception networks.

    Args:
        size: Desired output size. If size is an int instead of sequence like
            ``(h, w)``, a square image ``(size, size)`` is made.
        scale: range of size of the origin size cropped.
        ratio: range of aspect ratio of the origin aspect ratio cropped.
        interpolation: Default: :py:const:`PIL.Image.BILINEAR` (see
            :py:meth:`PIL.Image.Image.resize` for other options).
    """

    def _gen_params(
        self, frames: PILVideo
    ) -> FramesAndParams[PILVideo, Tuple[int, int, int, int]]:
        frame, frames = peek_iter(to_iter(frames))
        params = tv.RandomResizedCrop.get_params(frame, self.scale, self.ratio)
        return FramesAndParams(frames=frames, params=params)

    def _transform(
        self, frames: PILVideo, params: Tuple[int, int, int, int]
    ) -> PILVideoI:
        i, j, h, w = params
        for frame in frames:
            yield self._transform_frame(frame, i, j, h, w)

    def __init__(
        self,
        size: Union[Tuple[int, int], int],
        scale: Tuple[float, float] = (0.08, 1.0),
        ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
        interpolation=PIL.Image.BILINEAR,
    ):
        self.size = canonicalize_size(size)
        self.interpolation = interpolation
        self.scale = scale
        self.ratio = ratio

    def __repr__(self):
        return self.__class__.__name__ + (
            "(size={size}, interpolation={interpolation}, "
            "scale={scale}, ratio={ratio}"
        ).format(
            size=self.size,
            interpolation=self.interpolation,
            scale=self.scale,
            ratio=self.ratio,
        )

    def _transform_frame(self, frame: Image, i: int, j: int, h: int, w: int) -> Image:
        return F.resized_crop(frame, i, j, h, w, self.size, self.interpolation)
