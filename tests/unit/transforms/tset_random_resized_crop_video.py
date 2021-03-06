from hypothesis import given, strategies as st

from torchvideo.transforms import RandomResizedCropVideo
from ..strategies import pil_video
from . import pil_interpolation_settings
from .assertions import assert_preserves_label


class TestRandomResizedCropVideo:
    @given(pil_video(), st.sampled_from(pil_interpolation_settings), st.data())
    def test_resulting_video_are_specified_size(self, video, interpolation, data):
        width, height = video[0].size
        expected_width = data.draw(st.integers(min_value=1, max_value=width))
        expected_height = data.draw(st.integers(min_value=1, max_value=height))
        transform = RandomResizedCropVideo(
            (expected_height, expected_width), interpolation=interpolation
        )

        transformed_video = list(transform(video))

        assert len(transformed_video) == len(video)
        for frame in transformed_video:
            assert (expected_width, expected_height) == frame.size

    def test_propagates_label_unchanged(self):
        frames = pil_video().example()

        transform = RandomResizedCropVideo((1, 1))

        assert_preserves_label(transform, frames)
