# PyTest has weird syntax for parameterizing fixtures:
# https://docs.pytest.org/en/latest/fixture.html#parametrizing-fixtures

import pytest
from hypothesis import given
import hypothesis.strategies as st

from assertions.seq import assert_ordered, assert_elems_gte, assert_elems_lt
from torchvideo.samplers import (
    FullVideoSampler,
    TemporalSegmentSampler,
    ClipSampler,
    LambdaSampler,
    frame_idx_to_list,
)


def full_video_sampler():
    return FullVideoSampler()


def temporal_segment_sampler():
    segment_count = st.integers(1, 100).example()
    snippet_length = st.integers(1, 1000).example()
    return TemporalSegmentSampler(segment_count, snippet_length)


def clip_sampler():
    clip_length = st.integers(1, 1000).example()
    return ClipSampler(clip_length)


@pytest.fixture(params=[clip_sampler, full_video_sampler, temporal_segment_sampler])
def frame_sampler(request):
    return request.param()


class TestFrameSampler:
    """
    These tests test general properties that all samplers should abide by.
    """

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
        assert_elems_lt(frames_idx, frame_count)
        assert_elems_gte(frames_idx, 0)


class TestFullVideoSampler:
    @given(st.integers(min_value=1))
    def test_full_video_sampler(self, length):
        sampler = FullVideoSampler()

        idx_slice = sampler.sample(length)

        assert idx_slice.stop == length
        assert idx_slice.start == 0
        assert idx_slice.step == 1

    def test_full_video_sampler_repr(self):
        assert repr(FullVideoSampler()) == "FullVideoSampler()"

    def test_full_video_sampler_str(self):
        assert str(FullVideoSampler()) == "FullVideoSampler()"


class TestClipSampler:
    @given(st.data())
    def test_clip_is_subsampled_from_video_when_video_is_longer_than_clip(self, data):
        clip_length = data.draw(st.integers(1, 1000))
        video_length = data.draw(st.integers(clip_length, 1000))
        sampler = ClipSampler(clip_length=clip_length)

        frame_idx = sampler.sample(video_length)

        assert frame_idx.step == 1
        assert (frame_idx.stop - frame_idx.start) == clip_length

    @given(st.data())
    def test_clip_is_oversampled_when_video_is_shorter_than_clip_length(self, data):
        clip_length = data.draw(st.integers(2, 1000))
        video_length = data.draw(st.integers(1, clip_length - 1))
        sampler = ClipSampler(clip_length=clip_length)

        frame_idx = sampler.sample(video_length)
        frame_idx = frame_idx_to_list(frame_idx)

        assert_elems_lt(frame_idx, clip_length - 1)
        assert_elems_gte(frame_idx, 0)

    def test_repr(self):
        assert repr(ClipSampler(10)) == "ClipSampler(clip_length=10, frame_step=1)"


class TestTemporalSegmentSampler:
    def test_raises_value_error_when_sampling_from_a_video_of_0_frames(self):
        sampler = TemporalSegmentSampler(1, 1)
        with pytest.raises(ValueError):
            sampler.sample(0)

    @pytest.mark.parametrize(
        "video_length,segment_count,snippet_length," "expected_idx",
        [(1, 2, 1, [0, 0]), (2, 2, 2, [0, 0, 0, 0])],
    )
    def test_sampling_when_snippets_are_longer_than_segments(
        self, video_length, segment_count, snippet_length, expected_idx
    ):
        frame_idx = self.sample(video_length, segment_count, snippet_length)

        assert frame_idx == expected_idx

    @pytest.mark.parametrize(
        "video_length,segment_count,snippet_length,expected_idx",
        [(1, 1, 1, [0]), (2, 2, 1, [0, 1]), (9, 3, 3, [0, 1, 2, 3, 4, 5, 6, 7, 8])],
    )
    def test_sampling_snippets_same_length_as_segments(
        self, video_length, segment_count, snippet_length, expected_idx
    ):
        frame_idx = self.sample(video_length, segment_count, snippet_length)

        assert frame_idx == expected_idx

    @pytest.mark.parametrize(
        "video_length,segment_count,snippet_length,expected_idx",
        [
            (9, 3, 1, [1, 4, 7]),
            (10, 2, 1, [2, 7]),
            (5, 1, 3, [1, 2, 3]),
            (5, 1, 4, [0, 1, 2, 3]),
        ],
    )
    def test_sampling_in_test_mode_centres_snippets_in_segments(
        self, video_length, segment_count, snippet_length, expected_idx
    ):

        frame_idx = self.sample(video_length, segment_count, snippet_length, test=True)

        assert frame_idx == expected_idx

    def test_sampling_is_random(self):
        seen_idx = set()
        for i in range(1000):
            frame_idx = self.sample(100, 2, 1)
            seen_idx.add(tuple(frame_idx))

        # A bit handwavy, but basically we want to ensure have a reasonable number of
        # segment positions. We generate 1000 frame_idx, for 2 snippets of length 1.
        assert len(seen_idx) > 10
        distances_between_frames = [idx[1] - idx[0] for idx in seen_idx]
        # We want to ensure offsets are sampled randomly for each segment, and not just
        # once and used for every segment.
        assert len(set(distances_between_frames)) > 1

    @given(st.data())
    def test_fuzz_sampler_training(self, data):
        segment_count, snippet_length, video_length = self.draw_sampler_parameters(data)

        frame_idx = self.sample(video_length, segment_count, snippet_length)

        assert_valid_frame_index(
            frame_idx, segment_count * snippet_length, video_length
        )

    @given(st.data())
    def test_fuzz_sampler_test(self, data):
        segment_count, snippet_length, video_length = self.draw_sampler_parameters(data)

        frame_idx = self.sample(video_length, segment_count, snippet_length, test=True)

        assert_valid_frame_index(
            frame_idx, segment_count * snippet_length, video_length
        )

    def test_repr(self):
        assert (
            repr(TemporalSegmentSampler(1, 5))
            == "TemporalSegmentSampler(segment_count=1, snippet_length=5)"
        )

    def test_str(self):
        assert (
            str(TemporalSegmentSampler(1, 5))
            == "TemporalSegmentSampler(segment_count=1, snippet_length=5)"
        )

    def test_segment_length_should_be_greater_than_0(self):
        with pytest.raises(ValueError):
            TemporalSegmentSampler(1, 0)

    def test_segment_count_should_be_greater_than_0(self):
        with pytest.raises(ValueError):
            TemporalSegmentSampler(0, 1)

    def sample(self, video_length, segment_count, snippet_length, test=False):
        sampler = TemporalSegmentSampler(segment_count, snippet_length, test=test)
        frame_idx = frame_idx_to_list(sampler.sample(video_length))
        return frame_idx

    @staticmethod
    def draw_sampler_parameters(data):
        segment_count = data.draw(st.integers(1, 100))
        snippet_length = data.draw(st.integers(1, 100))
        video_length = data.draw(st.integers(1, 10000))
        return segment_count, snippet_length, video_length


class TestLambdaSampler:
    def test_throws_error_if_user_provided_sampling_fn_returns_invalid_idx(self):
        sampler = LambdaSampler(lambda frames: [frames + 1])

        with pytest.raises(ValueError):
            sampler.sample(10)

    def test_repr(self):
        class MySampler:
            def __call__(self, video_length):
                return slice(0, video_length, 1)

            def __repr__(self):
                return self.__class__.__name__ + "()"

        sampler = LambdaSampler(MySampler())

        assert repr(sampler) == "LambdaSampler(sampler=MySampler())"


def assert_valid_frame_index(frame_idx, expected_frame_count, video_length):
    assert len(frame_idx) == expected_frame_count
    assert_ordered(frame_idx)
    assert_elems_lt(frame_idx, video_length)
    assert_elems_gte(frame_idx, 0)
