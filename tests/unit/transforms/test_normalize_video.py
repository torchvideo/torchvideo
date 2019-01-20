import numpy as np
import pytest
import torch
from hypothesis import given, strategies as st, note

from torchvideo.transforms import NormalizeVideo
from ..strategies import tensor_video
from ..transforms import stats
from .assertions import assert_preserves_label


class TestNormalizeVideo:
    def test_repr(self):
        assert repr(NormalizeVideo(128, 15)) == "NormalizeVideo(mean=128, std=15)"

    @given(tensor_video())
    def test_scalar_statistics_smoke(self, video):
        NormalizeVideo(128, 1)(video)

    @given(tensor_video())
    def test_vector_statistics_smoke(self, video):
        mean = [128] * video.shape[0]
        std = [1] * video.shape[0]
        NormalizeVideo(mean, std)(video)

    def test_raises_value_error_on_0_std(self):
        with pytest.raises(ValueError):
            NormalizeVideo(10, 0)

    def test_raises_value_error_on_0_element_in_std_vector(self):
        with pytest.raises(ValueError):
            NormalizeVideo([10, 10], [5, 0])

    def test_raises_value_error_when_length_of_std_and_mean_dont_match(self):
        with pytest.raises(ValueError):
            NormalizeVideo([10], [5, 0])

    def test_raises_value_error_when_length_of_mean_is_not_equal_to_channel_count(self):
        transform = NormalizeVideo([10, 10], [5, 5])

        with pytest.raises(ValueError):
            transform(torch.randn(3, 1, 1, 1))

    def test_transform_inplace(self):
        transform = NormalizeVideo([10], [5], inplace=True)
        pre_transform_tensor = torch.randn(1, 2, 3, 4)
        post_transform_tensor = transform(pre_transform_tensor)

        assert torch.equal(pre_transform_tensor, post_transform_tensor)

    def test_transform_not_inplace(self):
        transform = NormalizeVideo([10], [5], inplace=False)
        pre_transform_tensor = torch.randn(1, 2, 3, 4)
        post_transform_tensor = transform(pre_transform_tensor)

        assert not torch.equal(pre_transform_tensor, post_transform_tensor)

    @pytest.mark.skipif(stats is None, reason="scipy.stats is not available")
    @given(st.integers(2, 4))
    def test_distribution_is_normal_after_transform(self, ndim):
        """Basically a direct copy of
        https://github.com/pytorch/vision/blob/master/test/test_transforms.py#L753"""

        def kstest(tensor):
            p_value = stats.kstest(list(tensor.view(-1)), "norm", args=(0, 1)).pvalue
            return p_value

        p_value = 0.0001
        for channel_count in [1, 3]:
            # video is normally distributed ~ N(5, 10)
            if ndim == 2:
                shape = [channel_count, 500]
            elif ndim == 3:
                shape = [channel_count, 10, 50]
            else:
                shape = [channel_count, 5, 10, 10]
            video = torch.randn(*shape) * 10 + 5
            # We want the video not to be sampled from N(0, 1)
            # i.e. we want to reject the null hypothesis that video is from this
            # distribution
            assert kstest(video) <= p_value

            mean = [video[c].mean() for c in range(channel_count)]
            std = [video[c].std() for c in range(channel_count)]
            normalized = NormalizeVideo(mean, std)(video)

            # Check the video *is* sampled from N(0, 1)
            # i.e. we want to maintain the null hypothesis that the normalised video is
            # from this distribution
            assert kstest(normalized) >= 0.0001

    @given(st.data())
    def test_preserves_channel_count(self, data):
        video = data.draw(tensor_video())
        input_channel_count = video.size(0)
        mean = np.random.randn(input_channel_count)
        std = np.random.randn(input_channel_count)
        note(mean)
        note(std)
        transform = NormalizeVideo(mean, std)

        transformed_video = transform(video)

        output_channel_count = transformed_video.size(0)
        assert input_channel_count == output_channel_count

    def test_propagates_label_unchanged(self):
        video = tensor_video(min_width=1, min_height=1).example()
        channel_count = video.shape[0]
        transform = NormalizeVideo(torch.ones(channel_count), torch.ones(channel_count))

        assert_preserves_label(transform, video)
