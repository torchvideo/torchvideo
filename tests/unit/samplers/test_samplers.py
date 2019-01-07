# PyTest has weird syntax for parameterizing fixtures:
# https://docs.pytest.org/en/latest/fixture.html#parametrizing-fixtures

import numpy as np

import pytest
from hypothesis import given, assume
import hypothesis.strategies as st

from assertions.seq import assert_ordered
from torchvideo.internal.utils import frame_idx_to_list
from torchvideo.samplers import FullVideoSampler, TemporalSegmentSampler, ClipSampler


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


class TestClipSampler:
    @given(st.integers(1, 100), st.integers(1, 100), st.integers(1, 5))
    def test_produces_frame_idx_of_given_length(
        self, video_length, clip_length, step_size
    ):
        assume(video_length >= clip_length * step_size)
        sampler = ClipSampler(clip_length, step_size)

        frame_idx = sampler.sample(video_length)

        frame_idx = frame_idx_to_list(frame_idx)
        assert len(frame_idx) == clip_length
        assert all([index < video_length for index in frame_idx])
        assert all([index >= 0 for index in frame_idx])

    def test_repr(self):
        assert repr(ClipSampler(10)) == "ClipSampler(clip_length=10, frame_step=1)"
