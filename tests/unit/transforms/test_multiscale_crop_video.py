import numpy as np
from hypothesis import given, strategies as st

from torchvideo.transforms import ImageShape
from torchvideo.transforms import NDArrayToPILVideo
from torchvideo.transforms import MultiScaleCropVideo
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestMultiScaleCropVideo:
    @given(st.data())
    def test_transform_always_yields_crops_of_the_correct_size(self, data):
        crop_height = data.draw(st.integers(1, 10))
        crop_width = data.draw(st.integers(1, 10))
        duration = data.draw(st.integers(1, 10))
        fixed_crops = data.draw(st.booleans())
        if fixed_crops:
            more_fixed_crops = data.draw(st.booleans())
        else:
            more_fixed_crops = False
        height = data.draw(st.integers(crop_height, crop_height * 100))
        width = data.draw(st.integers(crop_width, crop_width * 100))

        video_shape = (duration, height, width, 3)

        scale_strategy = st.floats(
            min_value=min(crop_width, crop_height) / min(height, width), max_value=1
        )
        scales = data.draw(st.lists(scale_strategy, min_size=1, max_size=5))
        max_distortion = data.draw(st.integers(0, len(scales)))
        video = NDArrayToPILVideo()(np.ones(video_shape, dtype=np.uint8))
        transform = MultiScaleCropVideo(
            size=ImageShape(height=crop_height, width=crop_width),
            scales=scales,
            max_distortion=max_distortion,
            fixed_crops=fixed_crops,
            more_fixed_crops=more_fixed_crops,
        )
        transformed_video = list(transform(video))
        print("video_shape", video_shape)
        print("scales", scales)
        print("max_distortion", max_distortion)
        print("fixed_crops", fixed_crops)
        print("more_fixed_crops", more_fixed_crops)

        assert len(transformed_video) == duration
        for frame in transformed_video:
            print("crop_size", np.array(frame).shape)
            np.testing.assert_allclose(np.array(frame), np.ones_like(frame))
            assert frame.height == crop_height
            assert frame.width == crop_width

    def test_repr(self):
        transform = MultiScaleCropVideo(
            size=10,
            scales=(1, 0.875),
            max_distortion=1,
            fixed_crops=False,
            more_fixed_crops=False,
        )

        assert (
            repr(transform) == "MultiScaleCropVideo("
            "size=ImageShape(height=10, width=10), "
            "scales=(1, 0.875), "
            "max_distortion=1, "
            "fixed_crops=False, "
            "more_fixed_crops=False)"
        )

    def test_propagates_label_unchanged(self):
        video = pil_video(min_height=2, min_width=2).example()
        transform = MultiScaleCropVideo((1, 1), scales=(1,))

        assert_preserves_label(transform, video)
