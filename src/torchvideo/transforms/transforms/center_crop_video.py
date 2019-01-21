from typing import Iterator, Union, Tuple

from PIL.Image import Image
from torchvision.transforms import transforms as tv

from .types import PILVideo
from .transform import Transform


class CenterCropVideo(Transform[PILVideo, Iterator[Image], None]):
    """Crops the given video (composed of PIL Images) at the center of the frame.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            ``int`` instead of sequence like ``(h, w)``, a square crop ``(size, size)``
            is made.
    """

    def __init__(self, size: Union[Tuple[int, int], int]):
        super().__init__()
        self._image_transform = tv.CenterCrop(size)

    def _gen_params(self, frames):
        return None

    def __repr__(self) -> str:
        return self.__class__.__name__ + "(size={0})".format(self._image_transform.size)

    def _transform(self, frames: PILVideo, params):
        for frame in frames:
            yield self._image_transform(frame)
