from hypothesis import given, strategies as st

from assertions.seq import assert_elems_lt, assert_elems_gte
from torchvideo.samplers import ClipSampler, frame_idx_to_list


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

    def test_clip_sampler_samples_central_clip_in_test_mode(self):
        sampler = ClipSampler(clip_length=3, test=True)
        frame_idx = sampler.sample(5)
        assert frame_idx == slice(1, 4, 1)

    @given(st.data())
    def test_clip_sampler_is_deterministic_in_test_mode(self, data):
        clip_length = data.draw(st.integers(1, 1000))
        video_length = data.draw(st.integers(1, 1000))
        sampler = ClipSampler(clip_length=clip_length, test=True)

        sample_count = 10
        frame_idx = [sampler.sample(video_length) for _ in range(sample_count)]

        for i in range(1, sample_count):
            assert frame_idx[i - 1] == frame_idx[i]

    def test_repr(self):
        assert repr(ClipSampler(10)) == "ClipSampler(clip_length=10, frame_step=1)"
