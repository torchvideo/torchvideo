import numpy as np
import pytest
from hypothesis import given
import hypothesis.strategies as st

from assertions.seq import assert_ordered
from torchvideo.samplers import TemporalSegmentSampler


class TestTemporalSegmentSampler:
    def test_fewer_frames_than_segments(self):
        sampler = TemporalSegmentSampler(3, 1)
        frame_count = 2

        frame_idx = sampler.sample(frame_count)

        assert frame_idx == [0, 0, 0]

    def test_average_segment_frame_count_less_than_1(self):
        segment_count = 3
        sampler = TemporalSegmentSampler(segment_count, 1)
        frame_count = 4

        frame_idx = sampler.sample(frame_count)

        assert len(frame_idx) == segment_count
        assert_ordered(frame_idx)
        assert np.all(np.array(frame_idx) < frame_count)

    def test_average_segment_frame_count_greater_than_1(self):
        segment_count = 3
        sampler = TemporalSegmentSampler(segment_count, 1)
        frame_count = 6

        frame_idx = sampler.sample(frame_count)

        assert len(frame_idx) == segment_count
        assert (
            len(set(frame_idx)) == segment_count
        )  # segment indices should be unique if possible
        assert_ordered(frame_idx)
        assert np.all(np.array(frame_idx) < frame_count)

    @given(st.integers(1, 100), st.integers(1, 1000), st.integers(1, 10000))
    def test_index_properties(self, segment_count, segment_length, video_length):
        sampler = TemporalSegmentSampler(segment_count, segment_length)

        frame_idx = sampler.sample(video_length)

        assert len(frame_idx) == segment_count
        assert_ordered(frame_idx)
        assert np.all(np.array(frame_idx) < video_length)

    def test_repr(self):
        assert (
            repr(TemporalSegmentSampler(1, 5))
            == "TemporalSegmentSampler(segment_count=1, segment_length=5)"
        )

    def test_str(self):
        assert (
            str(TemporalSegmentSampler(1, 5))
            == "TemporalSegmentSampler(segment_count=1, segment_length=5)"
        )

    def test_segment_length_should_be_greater_than_0(self):
        with pytest.raises(ValueError):
            TemporalSegmentSampler(1, 0)

    def test_segment_count_should_be_greater_than_0(self):
        with pytest.raises(ValueError):
            TemporalSegmentSampler(0, 1)
