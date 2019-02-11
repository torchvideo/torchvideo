import io

import numpy as np
from unittest.mock import Mock
import lintel
import pytest

from torchvideo.internal.readers import lintel_loader


@pytest.fixture()
def loadvid_frame_nums_mock(monkeypatch):
    def side_effect(binary_data, frame_nums, *args, **kwargs):
        width, height = (5, 5)
        flat_shape = width * height * len(frame_nums) * 3
        frames_data = np.random.randint(0, 254, flat_shape, dtype=np.uint8)
        return frames_data.tobytes(), width, height

    mock_loadvid = Mock(side_effect=side_effect)
    with monkeypatch.context() as ctx:
        ctx.setattr(lintel, "loadvid_frame_nums", mock_loadvid)
        yield mock_loadvid


class TestLintelReaderUnit:
    def test_loading_sequential_contiguous_frames(self, loadvid_frame_nums_mock):
        f = io.BytesIO(b"")
        frame_nums = [0, 1, 2, 3]
        frames = list(lintel_loader(f, frame_nums))

        self.assert_loadvid_correctly_called(
            loadvid_frame_nums_mock, f, frame_nums, frames, frame_nums
        )

    def test_loading_duplicate_frames(self, loadvid_frame_nums_mock):
        f = io.BytesIO(b"")

        frame_nums = [0, 0, 0, 1, 2, 3]
        frames = list(lintel_loader(f, frame_nums))

        self.assert_loadvid_correctly_called(
            loadvid_frame_nums_mock, f, frame_nums, frames, [0, 1, 2, 3]
        )

    def test_loading_unorderd_frames(self, loadvid_frame_nums_mock):
        f = io.BytesIO(b"")

        frame_nums = [3, 1]
        frames = list(lintel_loader(f, frame_nums))

        self.assert_loadvid_correctly_called(
            loadvid_frame_nums_mock, f, frame_nums, frames, [1, 3]
        )

    def assert_loadvid_correctly_called(
        self, loadvid_frame_nums_mock, f, frame_nums, frames, expected_load_idx
    ):
        assert len(frames) == len(frame_nums)
        assert loadvid_frame_nums_mock.call_count == 1

        args, kwargs = loadvid_frame_nums_mock.call_args
        assert args[0] == f.read()
        np.testing.assert_array_equal(np.array(expected_load_idx), kwargs["frame_nums"])
        assert kwargs["should_seek"]
