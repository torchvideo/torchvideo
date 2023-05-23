from typing import Iterator, Union, Tuple

from PIL import Image
from torchvision.transforms import transforms as tv

from .types import PILVideo
from .transform import Transform
import numpy as np


class FlipColorChannelsVideo(Transform[PILVideo, Iterator[Image], None]):
    """Flips the colors (rgb->bgr) of given video (composed of PIL Images).

    """

    def __init__(self):
        super().__init__()
        # self._image_transform = tv.CenterCrop(size)

    def _gen_params(self, frames):
        return None

    def _transform(self, frames: PILVideo, params):
        for frame in frames:
            yield Image.fromarray(np.array(frame)[:, :, ::-1])
