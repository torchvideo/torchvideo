import itertools

import numpy as np
import pytest
from hypothesis import given

from torchvideo.transforms import Compose
from torchvideo.transforms import NDArrayToPILVideo
from torchvideo.transforms import CollectFrames
from ..strategies import video_shape
from .assertions import assert_preserves_label


class TestNDArrayToPILVideo:
    def test_repr(self):
        assert repr(NDArrayToPILVideo()) == "NDArrayToPILVideo(format='thwc')"

    @given(video_shape())
    def test_converts_thwc_to_PIL_video(self, shape):
        t, h, w = shape
        video = self.make_uint8_ndarray((t, h, w, 3))
        transform = Compose([NDArrayToPILVideo(), CollectFrames()])

        pil_video = transform(video)

        assert len(pil_video) == t
        assert pil_video[0].size[0] == w
        assert pil_video[0].size[1] == h
        assert all([f.mode == "RGB" for f in pil_video])

    @given(video_shape())
    def test_converts_cthw_to_PIL_video(self, shape):
        t, h, w = shape
        video = self.make_uint8_ndarray((3, t, h, w))
        transform = Compose([NDArrayToPILVideo(format="cthw"), CollectFrames()])

        pil_video = transform(video)

        assert len(pil_video) == t
        assert pil_video[0].size[0] == w
        assert pil_video[0].size[1] == h
        assert all([f.mode == "RGB" for f in pil_video])

    def test_only_thwc_and_cthw_are_valid_formats(self):
        invalid_formats = [
            "".join(f)
            for f in itertools.permutations("thwc")
            if "".join(f) not in {"thwc", "cthw"}
        ]
        for invalid_format in invalid_formats:
            with pytest.raises(
                ValueError, match="Invalid format '{}'".format(invalid_format)
            ):
                NDArrayToPILVideo(format=invalid_format)

    def test_propagates_label_unchanged(self):
        video = self.make_uint8_ndarray((3, 1, 2, 2))
        transform = NDArrayToPILVideo(format="cthw")

        assert_preserves_label(transform, video)

    @staticmethod
    def make_uint8_ndarray(shape):
        return np.random.randint(0, 255, size=shape, dtype=np.uint8)
