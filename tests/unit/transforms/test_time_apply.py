from unittest.mock import Mock, call

from hypothesis import given

from torchvideo.transforms import TimeApply
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestTimeApply:
    @given(pil_video(min_length=1, max_length=5))
    def test_applies_given_transform_for_each_frame(self, frames):
        mock_transform = Mock(side_effect=lambda frames: frames)
        transform = TimeApply(mock_transform)
        expected_calls = [call(frame) for frame in frames]

        transformed_frames = list(transform(frames))

        assert len(transformed_frames) == len(frames)
        assert transformed_frames == frames
        assert mock_transform.mock_calls == expected_calls

    def test_propagates_label_unchanged(self):
        stub_transform = lambda frames: frames
        frames = pil_video().example()

        transform = TimeApply(stub_transform)

        assert_preserves_label(transform, frames)
