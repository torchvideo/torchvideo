import pytest

from torchvideo.internal.utils import compute_sample_length


sample_length_test_data = [
    (1, 1, 1),
    (2, 1, 2),
    (3, 1, 3),
    (4, 1, 4),
    (1, 2, 1),
    (2, 2, 3),
    (3, 2, 5),
    (4, 2, 7),
    (1, 3, 1),
    (2, 3, 4),
    (3, 3, 7),
    (1, 4, 1),
    (2, 4, 5),
    (3, 4, 9),
]


@pytest.mark.parametrize(
    "clip_length,step_size,expected_sample_size", sample_length_test_data
)
def test_compute_sample_length(clip_length, step_size, expected_sample_size):
    sample_size = compute_sample_length(clip_length, step_size)
    assert expected_sample_size == sample_size
