import itertools

import numpy as np
import pytest
import torch
from PIL import Image

from torchvideo.tools import _to_list_of_np_frames


class TestToListOfNpFrames:
    def test_from_ndarray_cthw(self):
        # CTHW
        duration, width, height, channels = 5, 7, 4, 3
        frames = np.random.randn(channels, duration, height, width) * 255

        frames_list = _to_list_of_np_frames(frames, ndarray_format="cthw")

        assert len(frames_list) == duration
        first_frame = frames_list[0]
        expected_shape = (height, width, channels)
        assert first_frame.shape == expected_shape
        expected_frame = np.moveaxis(frames[:, 0, :, :], 0, -1)
        np.testing.assert_array_equal(first_frame, expected_frame)

    def test_from_ndarray_thwc(self):
        duration, width, height, channels = 5, 7, 4, 3
        frames = np.random.randn(duration, height, width, channels) * 255

        frames_list = _to_list_of_np_frames(frames, ndarray_format="thwc")

        assert len(frames_list) == duration
        first_frame = frames_list[0]
        assert first_frame.shape == (height, width, channels)
        np.testing.assert_array_equal(first_frame, frames[0])

    def test_from_tensor_with_range_0_1(self):
        # CTHW
        duration, width, height, channels = 5, 7, 4, 3
        frames = torch.randn(channels, duration, height, width)
        frames[frames > 0.5] = 1
        frames[frames <= 0.5] = 0

        frames_list = _to_list_of_np_frames(frames)

        assert len(frames_list) == duration
        first_frame = frames_list[0]
        assert first_frame.shape == (height, width, channels)
        assert first_frame.max() == 255
        assert first_frame.min() == 0
        expected_first_frame_chw = frames.numpy()[:, 0, :, :].squeeze()
        expected_first_frame = (
            np.moveaxis(expected_first_frame_chw, 0, -1) * 255
        ).astype(np.uint8)
        np.testing.assert_array_equal(first_frame, expected_first_frame)

    def test_from_list_of_pil_images(self):
        duration, width, height, channels = 5, 7, 4, 3
        frames = [
            Image.fromarray(
                np.random.randint(
                    0, 255, size=(height, width, channels), dtype=np.uint8
                )
            )
            for _ in range(duration)
        ]

        frames_list = _to_list_of_np_frames(frames)

        assert len(frames_list) == duration
        np.testing.assert_array_equal(frames_list[0], np.array(frames[0]))

    def test_raises_error_on_ndarray_formats_other_than_cthw_or_thwc(self):
        frames = np.random.randn(3, 3, 3, 3)
        for format in itertools.permutations("thwc"):
            format = "".join(format)
            if format in {"thwc", "cthw"}:
                continue
            with pytest.raises(ValueError):
                _to_list_of_np_frames(frames, ndarray_format=format)

    def test_raises_error_on_unknown_format(self):
        with pytest.raises(TypeError):
            _to_list_of_np_frames(["a"])
