import pytest
from hypothesis import given, strategies as st

from torchvideo.samplers import TemporalSegmentSampler, frame_idx_to_list
from unit.samplers.assertions import assert_valid_frame_index


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
            repr(TemporalSegmentSampler(1, 5, test=False))
            == "TemporalSegmentSampler(segment_count=1, snippet_length=5, test=False)"
        )

    def test_str(self):
        assert (
            str(TemporalSegmentSampler(1, 5, test=True))
            == "TemporalSegmentSampler(segment_count=1, snippet_length=5, test=True)"
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
