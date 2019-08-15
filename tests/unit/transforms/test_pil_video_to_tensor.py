from itertools import permutations

import numpy as np
import pytest
from PIL import Image
from hypothesis import given

from torchvideo.transforms import PILVideoToTensor
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestPILVideoToTensor:
    def test_repr(self):
        assert (
            repr(PILVideoToTensor())
            == "PILVideoToTensor(rescale=True, ordering='CTHW')"
        )

    @given(pil_video())
    def test_transform(self, video):
        transform = PILVideoToTensor()
        tensor = transform(video)
        width, height = video[0].size
        n_channels = 3 if video[0].mode == "RGB" else 1
        assert tensor.size(0) == n_channels
        assert tensor.size(1) == len(video)
        assert tensor.size(2) == height
        assert tensor.size(3) == width

    def test_rescales_between_0_and_1(self):
        transform = PILVideoToTensor()
        frame_arr = 255 * np.ones(shape=(10, 20, 3), dtype=np.uint8)
        frame_arr[0:5, 0:10, :] = 0
        video = [Image.fromarray(frame_arr)]
        tensor = transform(video)

        assert tensor.min().item() == 0
        assert tensor.max().item() == 1

    def test_disabled_rescale(self):
        transform = PILVideoToTensor(rescale=False)
        frame_arr = 255 * np.ones(shape=(10, 20, 3), dtype=np.uint8)
        frame_arr[0:5, 0:10, :] = 0
        video = [Image.fromarray(frame_arr)]
        tensor = transform(video)

        assert tensor.min().item() == 0
        assert tensor.max().item() == 255

    def test_raises_exception_if_ordering_isnt_tchw_or_cthw(self):
        invalid_orderings = [
            "".join(order)
            for order in permutations(list("TCHW"))
            if "".join(order) not in ["TCHW", "CTHW"]
        ]

        for invalid_ordering in invalid_orderings:
            with pytest.raises(ValueError):
                PILVideoToTensor(ordering=invalid_ordering)

    def test_mapping_to_tchw_ordering(self):
        transform = PILVideoToTensor(ordering="TCHW", rescale=False)
        frames = [
            Image.fromarray(frame)
            for frame in np.random.randint(
                low=0, high=255, size=(5, 4, 4, 3), dtype=np.uint8
            )
        ]
        transformed_frames = transform(frames)
        for frame_index, frame in enumerate(frames):
            assert (
                frame.getpixel((0, 0))
                == transformed_frames[frame_index, :, 0, 0].numpy()
            ).all()

    def test_mapping_to_cthw_ordering(self):
        transform = PILVideoToTensor(ordering="CTHW", rescale=False)
        frames = [
            Image.fromarray(frame)
            for frame in np.random.randint(
                low=0, high=255, size=(5, 4, 4, 3), dtype=np.uint8
            )
        ]
        transformed_frames = transform(frames)
        for frame_index, frame in enumerate(frames):
            assert (
                frame.getpixel((0, 0))
                == transformed_frames[:, frame_index, 0, 0].numpy()
            ).all()

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=1, min_height=1).example()
        transform = PILVideoToTensor()

        assert_preserves_label(transform, video)
