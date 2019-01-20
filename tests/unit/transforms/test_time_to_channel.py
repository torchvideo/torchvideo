import pytest
import torch
from hypothesis import given, note

from torchvideo.transforms import TimeToChannel
from ..strategies import tensor_video
from .assertions import assert_preserves_label


def prod(seq):
    if len(seq) == 0:
        raise ValueError("Expected sequence to have at least 1 element")
    product = seq[0]
    for el in seq[1:]:
        product *= el
    return product


class TestTimeToChannel:
    transform = TimeToChannel()

    def test_repr(self):
        assert repr(TimeToChannel()) == "TimeToChannel()"

    def test_reshaping(self):
        frames = torch.zeros((10, 5, 36, 24))

        transformed_frames_shape = self.transform(frames).size()

        assert (50, 36, 24) == transformed_frames_shape

    @pytest.mark.parametrize("ndim", [1, 2, 3, 5])
    def test_raises_value_error_if_tensor_is_not_4d(self, ndim):
        with pytest.raises(ValueError):
            TimeToChannel()(torch.randn(*list(range(1, ndim + 1))))

    @given(tensor_video())
    def test_element_count_is_preserved(self, frames):
        transformed_frames_size = self.transform(frames).size()

        frames_size = frames.size()
        note(frames_size)
        note(transformed_frames_size)
        assert prod(transformed_frames_size) == prod(frames_size)

    @given(tensor_video())
    def test_first_dim_is_always_larger(self, frames):
        transformed_frames_size = self.transform(frames)

        assert frames.size(0) <= transformed_frames_size.size(0)

    def test_propagates_label_unchanged(self):
        video = tensor_video().example()
        transform = TimeToChannel()

        assert_preserves_label(transform, video)
