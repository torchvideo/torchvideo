from typing import Tuple, Union, Optional

from PIL.Image import Image
from torchvision.transforms import transforms as tv, functional as F

from .types import PILVideo, PILVideoI
from .internal import canonicalize_size, to_iter, peek_iter
from .transform import Transform


class RandomCropVideo(Transform[PILVideo, PILVideoI, Tuple[int, int, int, int]]):
    """Crop the given Video (composed of PIL Images) at a random location.

    Args:
        size: Desired output size of the crop. If ``size`` is an
            int instead of sequence like ``(h, w)``, a square crop ``(size, size)`` is
            made.
        padding: Optional padding on each border
            of the image. Default is ``None``, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively. If a sequence of length 2 is provided, it is used to
            pad left/right, top/bottom borders, respectively.
        pad_if_needed: Whether to pad the image if smaller than the
            desired size to avoid raising an exception.
        fill: Pixel fill value for constant fill. If a tuple of
            length 3, it is used to fill R, G, B channels respectively.
            This value is only used when the ``padding_mode`` is ``'constant'``.
        padding_mode: Type of padding. Should be one of: ``'constant'``, ``'edge'``,
            ``'reflect'`` or ``'symmetric'``.

             - ``'constant'``: pads with a constant value, this value is specified with
               fill.
             - ``'edge'``: pads with the last value on the edge of the image.
             - ``'reflect'``: pads with reflection of image (without repeating the last
               value on the edge) padding ``[1, 2, 3, 4]`` with 2 elements on both sides
               in reflect mode will result in ``[3, 2, 1, 2, 3, 4, 3, 2]``.
             - ``'symmetric'``: pads with reflection of image (repeating the last value
               on the edge) padding ``[1, 2, 3, 4]`` with 2 elements on both sides in
               symmetric mode will result in ``[2, 1, 1, 2, 3, 4, 4, 3]``.

    """

    def __init__(
        self,
        size: Union[Tuple[int, int], int],
        padding: Optional[Union[Tuple[int, int, int, int], Tuple[int, int]]] = None,
        pad_if_needed: bool = False,
        fill: int = 0,
        padding_mode: str = "constant",
    ):
        super().__init__()
        self.size = canonicalize_size(size)
        self.padding = padding
        self.pad_if_needed = pad_if_needed
        self.fill = fill
        self.padding_mode = padding_mode

    def _gen_params(self, frames: PILVideo) -> Tuple[int, int, int, int]:
        frames = to_iter(frames)
        first_frame, frames = peek_iter(frames)
        first_frame = self._maybe_pad(first_frame)
        params = tv.RandomCrop.get_params(first_frame, self.size)
        return params

    def _transform(
        self, frames: PILVideo, params: Tuple[int, int, int, int]
    ) -> PILVideoI:
        for frame in frames:
            yield F.crop(self._maybe_pad(frame), *params)

    def _maybe_pad(self, frame: Image):
        if self.padding is not None:
            frame = F.pad(frame, self.padding, self.fill, self.padding_mode)
        # pad the width if needed
        frame_width = frame.size[0]
        desired_width = self.size[1]
        if self.pad_if_needed and frame_width < desired_width:
            horizontal_padding = desired_width - frame_width
            frame = F.pad(frame, (horizontal_padding, 0), self.fill, self.padding_mode)
        # pad the height if needed
        frame_height = frame.size[1]
        desired_height = self.size[0]
        if self.pad_if_needed and frame_height < desired_height:
            vertical_padding = desired_height - frame_height
            frame = F.pad(frame, (0, vertical_padding), self.fill, self.padding_mode)
        return frame

    def __repr__(self) -> str:
        return (
            self.__class__.__name__ + "(size={size!r}, padding={padding!r}, "
            "pad_if_needed={pad_if_needed!r}, "
            "fill={fill!r}, padding_mode={padding_mode!r})".format(
                size=tuple(self.size),
                padding=self.padding,
                pad_if_needed=self.pad_if_needed,
                fill=self.fill,
                padding_mode=self.padding_mode,
            )
        )
