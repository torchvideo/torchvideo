# PyTest has weird syntax for parameterizing fixtures:
# https://docs.pytest.org/en/latest/fixture.html#parametrizing-fixtures
import numpy as np

import pytest
from hypothesis import given
import hypothesis.strategies as st

from assertions.seq import assert_ordered
from torchvideo.internal.utils import frame_idx_to_list
from torchvideo.samplers import FullVideoSampler, TemporalSegmentSampler


def full_video_sampler():
    return FullVideoSampler()


def temporal_segment_sampler():
    segment_count = st.integers(1, 100).example()
    segment_length = st.integers(1, 1000).example()
    return TemporalSegmentSampler(segment_count, segment_length)


@pytest.fixture(params=[full_video_sampler, temporal_segment_sampler])
def frame_sampler(request):
    return request.param()


class TestFrameSampler:
    @given(st.integers(max_value=0))
    def test_frame_sampler_raises_error_0_or_negative_frame_count(
        self, frame_sampler, frame_count
    ):
        with pytest.raises(ValueError, match=r"{} frames".format(frame_count)):
            frame_sampler.sample(frame_count)

    @given(st.integers(min_value=1, max_value=1000))
    def test_frame_sampler_generates_sequential_idx(self, frame_sampler, frame_count):
        frames_idx = frame_sampler.sample(frame_count)
        frames_idx = frame_idx_to_list(frames_idx)

        assert_ordered(frames_idx)
        assert np.all(np.array(frames_idx) >= 0)
        assert np.all(np.array(frames_idx) < frame_count)
