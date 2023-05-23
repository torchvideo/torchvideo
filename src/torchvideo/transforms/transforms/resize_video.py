from typing import Union, Tuple

import PIL
from PIL import Image
from torchvision.transforms import transforms as tv, functional as F
from torchvision.transforms.functional import InterpolationMode
from .types import PILVideo, PILVideoI
from .transform import StatelessTransform


_pil_interpolation_to_str = {
    InterpolationMode.NEAREST: 'PIL.Image.NEAREST',
    InterpolationMode.BILINEAR: 'PIL.Image.BILINEAR',
    InterpolationMode.BICUBIC: 'PIL.Image.BICUBIC',
    InterpolationMode.LANCZOS: 'PIL.Image.LANCZOS',
    InterpolationMode.HAMMING: 'PIL.Image.HAMMING',
    InterpolationMode.BOX: 'PIL.Image.BOX',
}


class ResizeVideo(StatelessTransform[PILVideo, PILVideoI]):
    """Resize the input video (composed of PIL Images) to the given size.

    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            ``(h, w)``, output size will be matched to this. If size is an ``int``,
            smaller edge of the image will be matched to this number.
            i.e, if ``height > width``, then image will be rescaled to
            ``(size * height / width, size)``.
        interpolation (int, optional): Desired interpolation. Default is
            :py:const:`PIL.Image.BILINEAR` (see :py:meth:`PIL.Image.Image.resize` for
            other options).
    """



    def __init__(
        self, size: Union[Tuple[int, int], int], interpolation=PIL.Image.BILINEAR
    ):
        self.size = size
        self.interpolation = interpolation

    def __repr__(self):
        interpolate_str = _pil_interpolation_to_str[self.interpolation]
        return self.__class__.__name__ + "(size={0!r}, interpolation={1})".format(
            self.size, interpolate_str
        )

    def _transform(self, frames: PILVideo, params: None) -> PILVideoI:
        for frame in frames:
            yield F.resize(frame, self.size, self.interpolation)