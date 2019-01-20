import numpy as np
from hypothesis import given

from torchvideo.transforms import RandomHorizontalFlipVideo
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestRandomHorizontalFlipVideo:
    def test_repr(self):
        assert (
            repr(RandomHorizontalFlipVideo(p=0.5)) == "RandomHorizontalFlipVideo(p=0.5)"
        )

    @given(pil_video())
    def test_always_flip(self, video):
        transform = RandomHorizontalFlipVideo(p=1)
        transformed_video = list(transform(transform(video)))
        assert len(transformed_video) == len(video)
        for frame, transformed_frame in zip(video, transformed_video):
            assert frame.size == transformed_frame.size
            assert frame.mode == transformed_frame.mode
            assert np.all(np.array(frame) == np.array(transformed_frame))

    @given(pil_video())
    def test_never_flip(self, video):
        transform = RandomHorizontalFlipVideo(p=0)
        transformed_video = list(transform(video))
        assert len(transformed_video) == len(video)
        for frame, transformed_frame in zip(video, transformed_video):
            assert frame.size == transformed_frame.size
            assert frame.mode == transformed_frame.mode
            assert np.all(np.array(frame) == np.array(transformed_frame))

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=2, min_height=2).example()
        transform = RandomHorizontalFlipVideo(0.5)

        assert_preserves_label(transform, video)
