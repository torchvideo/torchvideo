from random import randint

from hypothesis import given, strategies as st

from torchvideo.transforms import RandomCropVideo
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestRandomCropVideo:
    def test_repr(self):
        expected_repr = (
            "RandomCropVideo(size=(100, 120), padding=None, "
            "pad_if_needed=False, fill=0, padding_mode='constant')"
        )
        assert repr(RandomCropVideo((100, 120))) == expected_repr

    @given(video=pil_video(min_width=2, min_height=2), pad_if_needed=st.booleans())
    def test_crop_yields_image_of_specified_size(self, video, pad_if_needed):
        width, height = video[0].size
        if pad_if_needed:
            crop_size = (randint(1, height * 2), randint(1, width * 2))
        else:
            crop_size = (randint(1, height), randint(1, width))
        transform = RandomCropVideo(crop_size, pad_if_needed=pad_if_needed)
        transformed_video = transform(video)
        for frame in transformed_video:
            assert (frame.size[1], frame.size[0]) == crop_size

    @given(video=pil_video(min_width=2, min_height=2), fill=st.integers(0, 255))
    def test_crop_with_user_provided_padding(self, video, fill):
        width, height = video[0].size
        crop_size = (randint(1, height), randint(1, width))
        padding = tuple([randint(1, 4)] * 4)
        transform = RandomCropVideo(crop_size, padding=padding, fill=fill)
        transformed_video = transform(video)
        for frame in transformed_video:
            assert (frame.size[1], frame.size[0]) == crop_size

    @given(
        video=pil_video(min_width=2, min_height=2),
        padding_mode=st.sampled_from(["constant", "edge", "reflect", "symmetric"]),
    )
    def test_crop_with_different_padding_modes(self, video, padding_mode):
        width, height = video[0].size
        crop_size = (randint(1, height), randint(1, width))
        padding = tuple([randint(1, 4)] * 4)
        transform = RandomCropVideo(crop_size, padding_mode=padding_mode)
        transformed_video = transform(video)
        for frame in transformed_video:
            assert (frame.size[1], frame.size[0]) == crop_size

    def test_propagates_label_unchanges(self):
        video = pil_video(min_width=2, min_height=2).example()
        transform = RandomCropVideo(1)
        assert_preserves_label(transform, video)
