import itertools
import random
from typing import Tuple, Iterator, Sequence, List

import PIL
from PIL.Image import Image
from torchvision.transforms import functional as F

from .types import PILVideo, PILVideoI, ImageShape, Point
from .transform import Transform, FramesAndParams
from .internal import canonicalize_size


class MultiScaleCropVideo(Transform[PILVideo, PILVideoI, Tuple[ImageShape, Point]]):
    r"""Random crop the input video (composed of PIL Images) at one of the given
    scales or from a set of fixed crops, then resize to specified size.



    Args:
        size (sequence or int): Desired output size. If size is an
            int instead of sequence like ``(h, w)``, a square image ``(size, size)`` is
            made.
        scales (sequence): A sequence of floats between in the range :math:`[0, 1]`
            indicating the scale of the crop to be made.
        max_distortion (int): Integer between 0--``len(scales)`` that controls
            aspect-ratio distortion. This parameters decides which scales will be
            combined together when creating crop boxes. A max distortion of ``0``
            means that the crop width/height have to be from the same scale, whereas a
            distortion of 1 means that the crop width/height can be from 1 scale
            before or ahead in the ``scales`` sequence thereby stretching or squishing
            the frame.
        fixed_crops (bool): Whether to use upper right, upper left, lower right,
            lower left and center crop positions as the list of candidate crop positions
            instead of those generated from ``scales`` and ``max_distortion``.
        more_fixed_crops (bool): Whether to add center left, center right, upper center,
            lower center, upper quarter left, upper quarter right, lower quarter left,
            lower quarter right crop positions to the list of candidate crop
            positions that are randomly selected. ``fixed_crops`` must be enabled to use
            this setting.
    """

    def _gen_params(
        self, frames: PILVideo
    ) -> FramesAndParams[PILVideo, Tuple[ImageShape, Point]]:
        if isinstance(frames, list):
            frame = frames[0]
        else:
            assert isinstance(frames, Iterator)
            frame = next(frames)
            frames = itertools.chain([frame], frames)

        crop_shape, offset = self.get_params(
            frame,
            self.size,
            self.scales,
            max_distortion=self.max_distortion,
            fixed_crops=self.fixed_crops,
            more_fixed_crops=self.more_fixed_crops,
        )
        return FramesAndParams(frames, (crop_shape, offset))

    def _transform(
        self, frames: PILVideo, params: Tuple[ImageShape, Point]
    ) -> PILVideoI:
        crop_shape, offset = params
        for frame in frames:
            yield F.resized_crop(
                frame,
                offset.y,
                offset.x,
                crop_shape.height,
                crop_shape.width,
                size=self.size,
                interpolation=self.interpolation,
            )
        pass

    def __init__(
        self,
        size,
        scales: Sequence[float] = (1, 0.875, 0.75, 0.66),
        max_distortion: int = 1,
        fixed_crops: bool = True,
        more_fixed_crops: bool = True,
    ):
        self.size = canonicalize_size(size)
        self.scales = scales
        self.max_distortion = max_distortion
        self.fixed_crops = fixed_crops
        self.more_fixed_crops = more_fixed_crops
        if self.more_fixed_crops and not self.fixed_crops:
            raise ValueError("fixed_crops must be True if using more_fixed_crops.")
        self.interpolation = PIL.Image.BILINEAR

    def __repr__(self):
        return (
            self.__class__.__name__
            + "(size={size}, scales={scales}, max_distortion={max_distortion}, "
            "fixed_crops={fixed_crops}, more_fixed_crops={more_fixed_crops})".format(
                size=self.size,
                scales=self.scales,
                max_distortion=self.max_distortion,
                fixed_crops=self.fixed_crops,
                more_fixed_crops=self.more_fixed_crops,
            )
        )

    @classmethod
    def get_params(
        cls,
        frame: Image,
        output_shape: Tuple[int, int],
        scales: Sequence[float],
        max_distortion: int = 0,
        fixed_crops: bool = False,
        more_fixed_crops: bool = False,
    ) -> Tuple[ImageShape, Point]:
        input_width, input_height = frame.size
        input_shape = ImageShape(input_height, input_width)
        output_shape = ImageShape(*output_shape)

        shortest_side_length = min(input_shape)
        crop_sizes = [int(shortest_side_length * scale) for scale in scales]
        crop_shape = cls._sample_crop_shape(crop_sizes, max_distortion, output_shape)
        if not fixed_crops:
            offset = cls._sample_random_offset(input_shape, crop_shape)
        else:
            offset = cls._sample_fixed_offset(
                input_shape, crop_shape, more_fixed_crops=more_fixed_crops
            )

        return crop_shape, offset

    @classmethod
    def _sample_crop_shape(
        cls, crop_sizes: List[int], max_distortion: int, output_shape: ImageShape
    ) -> ImageShape:
        output_height, output_width = output_shape
        candidate_crop_heights = [
            output_height if abs(crop_size - output_height) < 3 else crop_size
            for crop_size in crop_sizes
        ]
        candidate_crop_widths = [
            output_width if abs(crop_size - output_width) < 3 else crop_size
            for crop_size in crop_sizes
        ]
        crop_shapes = []  # elements of the form: (crop_height, crop_width)
        for i, crop_height in enumerate(candidate_crop_heights):
            for j, crop_width in enumerate(candidate_crop_widths):
                if abs(i - j) <= max_distortion:
                    crop_shapes.append(ImageShape(crop_height, crop_width))
        return random.choice(crop_shapes)

    @staticmethod
    def _sample_random_offset(input_shape, crop_shape) -> Point:
        horizontal_offset = random.randint(0, input_shape.width - crop_shape.width)
        vertical_offset = random.randint(0, input_shape.height - crop_shape.height)
        return Point(x=horizontal_offset, y=vertical_offset)

    @classmethod
    def _sample_fixed_offset(
        cls, input_shape: ImageShape, crop_shape: ImageShape, more_fixed_crops=False
    ) -> Point:
        offsets: List[Point] = cls._fixed_crop_offsets(
            input_shape, crop_shape, more_fixed_crops=more_fixed_crops
        )
        return random.choice(offsets)

    @staticmethod
    def _fixed_crop_offsets(
        image_shape: ImageShape, crop_shape: ImageShape, more_fixed_crops=False
    ) -> List[Point]:
        horizontal_step = (image_shape.width - crop_shape.width) // 4
        vertical_step = (image_shape.height - crop_shape.height) // 4

        # Elements of the form (v_offset, h_offset)
        offsets = [
            Point(x=0, y=0),  # upper left
            Point(x=0, y=4 * vertical_step),  # lower left
            Point(x=4 * horizontal_step, y=0),  # upper right
            Point(x=4 * horizontal_step, y=4 * vertical_step),  # lower right
            Point(x=2 * horizontal_step, y=2 * vertical_step),  # center
        ]
        if more_fixed_crops:
            offsets += [
                Point(x=0, y=2 * vertical_step),  # center left
                Point(x=4 * horizontal_step, y=2 * vertical_step),  # center right
                Point(x=2 * horizontal_step, y=4 * vertical_step),  # lower center
                Point(x=2 * horizontal_step, y=0 * vertical_step),  # upper center
                Point(x=1 * horizontal_step, y=1 * vertical_step),  # upper left quarter
                Point(
                    x=3 * horizontal_step, y=1 * vertical_step
                ),  # upper right quarter
                Point(x=1 * horizontal_step, y=3 * vertical_step),  # lower left quarter
                Point(
                    x=3 * horizontal_step, y=3 * vertical_step
                ),  # lower right quarter
            ]

        return offsets
