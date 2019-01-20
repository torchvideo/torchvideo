import pytest

from torchvideo.samplers import LambdaSampler


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
