from random import randint

from hypothesis import given

from torchvideo.transforms import CenterCropVideo
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestCenterCropVideo:
    def test_repr(self):
        assert repr(CenterCropVideo((10, 20))) == "CenterCropVideo(size=(10, 20))"

    @given(pil_video(min_width=2, min_height=2))
    def test_crop(self, video):
        width, height = video[0].size
        crop_size = (randint(1, width), randint(1, height))
        transform = CenterCropVideo(crop_size)
        for frame in transform(video):
            assert (frame.size[1], frame.size[0]) == crop_size

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=2, min_height=2).example()
        transform = CenterCropVideo(1)

        assert_preserves_label(transform, video)
