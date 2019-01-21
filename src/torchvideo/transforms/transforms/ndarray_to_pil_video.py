import PIL
import numpy as np

from .types import PILVideoI
from .transform import Transform


class NDArrayToPILVideo(Transform[np.ndarray, PILVideoI, None]):
    """Convert :py:class:`numpy.ndarray` of the format :math:`(T, H, W, C)` or :math:`(
    C, T, H, W)` to a PIL video (an iterator of PIL images)
    """

    def __init__(self, format="thwc"):
        """

        Args:
            format: dimensional layout of array, one of ``"thwc"`` or ``"cthw"``
        """
        if format.lower() not in {"thwc", "cthw"}:
            raise ValueError(
                "Invalid format {!r}, only 'thwc' and 'cthw' are "
                "supported".format(format)
            )
        self.format = format

    def _transform(self, frames: np.ndarray, params: None) -> PILVideoI:
        if self.format == "cthw":
            frames = np.moveaxis(frames, 0, -1)
        for frame in frames:
            yield PIL.Image.fromarray(frame)

    def _gen_params(self, frames: np.ndarray) -> None:
        return None

    def __repr__(self):
        return self.__class__.__name__ + "(format={!r})".format(self.format)
