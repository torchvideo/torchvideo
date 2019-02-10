import itertools
import random
from typing import Tuple, Iterator, Sequence, List

import PIL
from PIL.Image import Image
from torchvision.transforms import functional as F

from .types import PILVideo, PILVideoI, ImageShape
from .transform import Transform, FramesAndParams
from .internal import canonicalize_size


class MultiScaleCropVideo(Transform[PILVideo, PILVideoI, Tuple[int, int, int, int]]):
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
    ) -> FramesAndParams[PILVideo, Tuple[int, int, int, int]]:
        if isinstance(frames, list):
            frame = frames[0]
        else:
            assert isinstance(frames, Iterator)
            frame = next(frames)
            frames = itertools.chain([frame], frames)

        h, w, i, j = self.get_params(
            frame,
            self.size,
            self.scales,
            max_distortion=self.max_distortion,
            fixed_crops=self.fixed_crops,
            more_fixed_crops=self.more_fixed_crops,
        )
        return FramesAndParams(frames, (h, w, i, j))

    def _transform(
        self, frames: PILVideo, params: Tuple[int, int, int, int]
    ) -> PILVideoI:
        h, w, i, j = params
        for frame in frames:
            yield F.resized_crop(
                frame, i, j, h, w, self.size, interpolation=self.interpolation
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
        output_size: Tuple[int, int],
        scales: Sequence[float],
        max_distortion: int = 0,
        fixed_crops: bool = False,
        more_fixed_crops: bool = False,
    ) -> Tuple[int, int, int, int]:
        output_size = ImageShape(*output_size)
        input_width, input_height = frame.size
        input_shape = ImageShape(input_height, input_width)

        shortest_side_length = min(input_shape)
        crop_sizes = [int(shortest_side_length * scale) for scale in scales]
        crop_shape = cls._sample_crop_shape(crop_sizes, max_distortion, output_size)
        if not fixed_crops:
            offset = cls._sample_random_offset(input_shape, crop_shape)
        else:
            offset = cls._sample_fixed_offset(
                input_shape, crop_shape, more_fixed_crops=more_fixed_crops
            )

        crop_height, crop_width = crop_shape
        h_offset, w_offset = offset
        return crop_height, crop_width, h_offset, w_offset

    @classmethod
    def _sample_crop_shape(cls, crop_sizes, max_distortion, output_shape):
        output_height, output_width = output_shape
        candidate_crop_heights = [
            output_height if abs(crop_size - output_height) < 3 else crop_size
            for crop_size in crop_sizes
        ]
        candidate_crop_widths = [
            output_width if abs(crop_size - output_width) < 3 else crop_size
            for crop_size in crop_sizes
        ]
        crop_shapes = []  # elements of the form: (crop_width, crop_shape)
        for i, crop_height in enumerate(candidate_crop_heights):
            for j, crop_width in enumerate(candidate_crop_widths):
                if abs(i - j) <= max_distortion:
                    crop_shapes.append((crop_height, crop_width))
        return random.choice(crop_shapes)

    @staticmethod
    def _sample_random_offset(input_shape, crop_shape) -> Tuple[int, int]:
        input_height, input_width = input_shape
        crop_height, crop_width = crop_shape
        w_offset = random.randint(0, input_width - crop_width)
        h_offset = random.randint(0, input_height - crop_height)
        return w_offset, h_offset

    @classmethod
    def _sample_fixed_offset(
        cls, input_shape: ImageShape, crop_shape: ImageShape, more_fixed_crops=False
    ) -> Tuple[int, int]:
        offsets = cls._fixed_crop_offsets(
            input_shape, crop_shape, more_fixed_crops=more_fixed_crops
        )
        return random.choice(offsets)

    @staticmethod
    def _fixed_crop_offsets(
        image_shape: ImageShape, crop_shape: ImageShape, more_fixed_crops=False
    ) -> List[Tuple[int, int]]:
        image_h, image_w = image_shape
        crop_h, crop_w = crop_shape
        horizontal_step = (image_w - crop_w) // 4
        vertical_step = (image_h - crop_h) // 4

        # Elements of the form (v_offset, h_offset)
        offsets = [
            (0, 0),  # upper left
            (0, 4 * horizontal_step),  # upper right
            (4 * vertical_step, 0),  # lower left
            (4 * vertical_step, 4 * horizontal_step),  # lower right
            (2 * vertical_step, 2 * horizontal_step),  # center
        ]
        if more_fixed_crops:
            offsets += [
                (2 * vertical_step, 0),  # center left
                (2 * vertical_step, 4 * horizontal_step),  # center right
                (4 * vertical_step, 2 * horizontal_step),  # lower center
                (0 * vertical_step, 2 * horizontal_step),  # upper center
                (1 * vertical_step, 1 * horizontal_step),  # upper left quarter
                (1 * vertical_step, 3 * horizontal_step),  # upper right quarter
                (3 * vertical_step, 1 * horizontal_step),  # lower left quarter
                (3 * vertical_step, 3 * horizontal_step),  # lower right quarter
            ]

        return offsets
