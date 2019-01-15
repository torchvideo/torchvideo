import itertools
from unittest.mock import Mock, call

import PIL.Image
import numpy as np
from random import randint

import pytest
import torch
from hypothesis import given, note
import hypothesis.strategies as st
from torchvision.transforms import Compose

try:
    from scipy import stats
except ImportError:
    stats = None

from torchvideo.transforms import (
    RandomCropVideo,
    CollectFrames,
    PILVideoToTensor,
    CenterCropVideo,
    RandomHorizontalFlipVideo,
    NormalizeVideo,
    NDArrayToPILVideo,
    TimeToChannel,
    MultiScaleCropVideo,
    ImageShape,
    TimeApply,
    ResizeVideo,
    RandomResizedCropVideo,
)
from unit.strategies import pil_video, tensor_video, video_shape

pil_interpolation_settings = [
    PIL.Image.NEAREST,
    PIL.Image.BOX,
    PIL.Image.BILINEAR,
    PIL.Image.HAMMING,
    PIL.Image.BICUBIC,
    PIL.Image.LANCZOS,
]


def prod(seq):
    if len(seq) == 0:
        raise ValueError("Expected sequence to have at least 1 element")
    product = seq[0]
    for el in seq[1:]:
        product *= el
    return product


def assert_preserves_label(transform, video):
    class my_label:
        pass

    frames, transformed_label = transform(video, my_label)
    assert transformed_label == my_label


class TestRandomCropVideo:
    def test_repr(self):
        expected_repr = (
            "RandomCropVideo(size=(100, 120), padding=None, "
            "pad_if_needed=False, fill=0, padding_mode='constant')"
        )
        assert repr(RandomCropVideo((100, 120))) == expected_repr

    @given(video=pil_video(min_width=2, min_height=2), pad_if_needed=st.booleans())
    def test_crop_yields_image_of_specified_size(self, video, pad_if_needed):
        width, height = video[0].size
        if pad_if_needed:
            crop_size = (randint(1, height * 2), randint(1, width * 2))
        else:
            crop_size = (randint(1, height), randint(1, width))
        transform = RandomCropVideo(crop_size, pad_if_needed=pad_if_needed)
        transformed_video = transform(video)
        for frame in transformed_video:
            assert (frame.size[1], frame.size[0]) == crop_size

    @given(video=pil_video(min_width=2, min_height=2), fill=st.integers(0, 255))
    def test_crop_with_user_provided_padding(self, video, fill):
        width, height = video[0].size
        crop_size = (randint(1, height), randint(1, width))
        padding = tuple([randint(1, 4)] * 4)
        transform = RandomCropVideo(crop_size, padding=padding, fill=fill)
        transformed_video = transform(video)
        for frame in transformed_video:
            assert (frame.size[1], frame.size[0]) == crop_size

    @given(
        video=pil_video(min_width=2, min_height=2),
        padding_mode=st.sampled_from(["constant", "edge", "reflect", "symmetric"]),
    )
    def test_crop_with_different_padding_modes(self, video, padding_mode):
        width, height = video[0].size
        crop_size = (randint(1, height), randint(1, width))
        padding = tuple([randint(1, 4)] * 4)
        transform = RandomCropVideo(crop_size, padding_mode=padding_mode)
        transformed_video = transform(video)
        for frame in transformed_video:
            assert (frame.size[1], frame.size[0]) == crop_size

    def test_propagates_label_unchanges(self):
        video = pil_video(min_width=2, min_height=2).example()
        transform = RandomCropVideo(1)
        assert_preserves_label(transform, video)


class TestCenterCropVideo:
    def test_repr(self):
        assert repr(CenterCropVideo((10, 20))) == "CenterCropVideo(size=(10, 20))"

    @given(pil_video(min_width=2, min_height=2))
    def test_crop(self, video):
        width, height = video[0].size
        crop_size = (randint(1, width), randint(1, height))
        transform = CenterCropVideo(crop_size)
        for frame in transform(video):
            assert (frame.size[1], frame.size[0]) == crop_size

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=2, min_height=2).example()
        transform = CenterCropVideo(1)

        assert_preserves_label(transform, video)


class TestRandomHorizontalFlipVideo:
    def test_repr(self):
        assert (
            repr(RandomHorizontalFlipVideo(p=0.5)) == "RandomHorizontalFlipVideo(p=0.5)"
        )

    @given(pil_video())
    def test_always_flip(self, video):
        transform = RandomHorizontalFlipVideo(p=1)
        transformed_video = list(transform(transform(video)))
        assert len(transformed_video) == len(video)
        for frame, transformed_frame in zip(video, transformed_video):
            assert frame.size == transformed_frame.size
            assert frame.mode == transformed_frame.mode
            assert np.all(np.array(frame) == np.array(transformed_frame))

    @given(pil_video())
    def test_never_flip(self, video):
        transform = RandomHorizontalFlipVideo(p=0)
        transformed_video = list(transform(video))
        assert len(transformed_video) == len(video)
        for frame, transformed_frame in zip(video, transformed_video):
            assert frame.size == transformed_frame.size
            assert frame.mode == transformed_frame.mode
            assert np.all(np.array(frame) == np.array(transformed_frame))

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=2, min_height=2).example()
        transform = RandomHorizontalFlipVideo(0.5)

        assert_preserves_label(transform, video)


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


class TestCollectFrames:
    def test_repr(self):
        assert repr(CollectFrames()) == "CollectFrames()"

    @given(pil_video(max_length=10, max_height=1, max_width=1))
    def test_collect_frames_make_list_from_iterator(self, video):
        transform = CollectFrames()
        assert transform(iter(video)) == video

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=1, min_height=1).example()
        transform = CollectFrames()

        assert_preserves_label(transform, iter(video))


class TestPILVideoToTensor:
    def test_repr(self):
        assert repr(PILVideoToTensor()) == "PILVideoToTensor()"

    @given(pil_video())
    def test_transform(self, video):
        transform = PILVideoToTensor()
        tensor = transform(video)
        width, height = video[0].size
        n_channels = 3 if video[0].mode == "RGB" else 1
        assert tensor.size(0) == n_channels
        assert tensor.size(1) == len(video)
        assert tensor.size(2) == height
        assert tensor.size(3) == width

    def test_rescales_between_0_and_1(self):
        transform = PILVideoToTensor()
        frame_arr = 255 * np.ones(shape=(10, 20, 3), dtype=np.uint8)
        frame_arr[0:5, 0:10, :] = 0
        video = [PIL.Image.fromarray(frame_arr)]
        tensor = transform(video)

        assert tensor.min().item() == 0
        assert tensor.max().item() == 1

    def test_disabled_rescale(self):
        transform = PILVideoToTensor(rescale=False)
        frame_arr = 255 * np.ones(shape=(10, 20, 3), dtype=np.uint8)
        frame_arr[0:5, 0:10, :] = 0
        video = [PIL.Image.fromarray(frame_arr)]
        tensor = transform(video)

        assert tensor.min().item() == 0
        assert tensor.max().item() == 255

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=1, min_height=1).example()
        transform = PILVideoToTensor()

        assert_preserves_label(transform, video)


class TestNDArrayToPILVideo:
    def test_repr(self):
        assert repr(NDArrayToPILVideo()) == "NDArrayToPILVideo(format='thwc')"

    @given(video_shape())
    def test_converts_thwc_to_PIL_video(self, shape):
        t, h, w = shape
        video = self.make_uint8_ndarray((t, h, w, 3))
        transform = Compose([NDArrayToPILVideo(), CollectFrames()])

        pil_video = transform(video)

        assert len(pil_video) == t
        assert pil_video[0].size[0] == w
        assert pil_video[0].size[1] == h
        assert all([f.mode == "RGB" for f in pil_video])

    @given(video_shape())
    def test_converts_cthw_to_PIL_video(self, shape):
        t, h, w = shape
        video = self.make_uint8_ndarray((3, t, h, w))
        transform = Compose([NDArrayToPILVideo(format="cthw"), CollectFrames()])

        pil_video = transform(video)

        assert len(pil_video) == t
        assert pil_video[0].size[0] == w
        assert pil_video[0].size[1] == h
        assert all([f.mode == "RGB" for f in pil_video])

    def test_only_thwc_and_cthw_are_valid_formats(self):
        invalid_formats = [
            "".join(f)
            for f in itertools.permutations("thwc")
            if "".join(f) not in {"thwc", "cthw"}
        ]
        for invalid_format in invalid_formats:
            with pytest.raises(
                ValueError, match="Invalid format '{}'".format(invalid_format)
            ):
                NDArrayToPILVideo(format=invalid_format)

    def test_propagates_label_unchanged(self):
        video = self.make_uint8_ndarray((3, 1, 2, 2))
        transform = NDArrayToPILVideo(format="cthw")

        assert_preserves_label(transform, video)

    @staticmethod
    def make_uint8_ndarray(shape):
        return np.random.randint(0, 255, size=shape, dtype=np.uint8)


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


class TestMultiScaleCropVideo:
    @given(st.data())
    def test_transform_always_yields_crops_of_the_correct_size(self, data):
        crop_height = data.draw(st.integers(1, 10))
        crop_width = data.draw(st.integers(1, 10))
        duration = data.draw(st.integers(1, 10))
        scale_strategy = st.floats(min_value=0.2, max_value=1)
        scales = data.draw(st.lists(scale_strategy, min_size=1, max_size=5))
        max_distortion = data.draw(st.integers(0, len(scales)))
        fixed_crops = data.draw(st.booleans())
        if fixed_crops:
            more_fixed_crops = data.draw(st.booleans())
        else:
            more_fixed_crops = False
        height = data.draw(st.integers(crop_height, crop_height * 100))
        width = data.draw(st.integers(crop_width, crop_width * 100))

        video_shape = (duration, height, width, 3)
        video = NDArrayToPILVideo()(np.zeros(video_shape, dtype=np.uint8))
        transform = MultiScaleCropVideo(
            size=ImageShape(height=crop_height, width=crop_width),
            scales=scales,
            max_distortion=max_distortion,
            fixed_crops=fixed_crops,
            more_fixed_crops=more_fixed_crops,
        )
        transformed_video = list(transform(video))

        assert len(transformed_video) == duration
        assert all([frame.height == crop_height for frame in transformed_video])
        assert all([frame.width == crop_width for frame in transformed_video])

    def test_repr(self):
        transform = MultiScaleCropVideo(
            size=10,
            scales=(1, 0.875),
            max_distortion=1,
            fixed_crops=False,
            more_fixed_crops=False,
        )

        assert (
            repr(transform) == "MultiScaleCropVideo("
            "size=ImageSize(height=10, width=10), "
            "scales=(1, 0.875), "
            "max_distortion=1, "
            "fixed_crops=False, "
            "more_fixed_crops=False)"
        )

    def test_propagates_label_unchanged(self):
        video = pil_video(min_height=2, min_width=2).example()
        transform = MultiScaleCropVideo((1, 1), scales=(1,))

        assert_preserves_label(transform, video)


class TestTimeApply:
    @given(pil_video(min_length=1, max_length=5))
    def test_applies_given_transform_for_each_frame(self, frames):
        mock_transform = Mock(side_effect=lambda frames: frames)
        transform = TimeApply(mock_transform)
        expected_calls = [call(frame) for frame in frames]

        transformed_frames = list(transform(frames))

        assert len(transformed_frames) == len(frames)
        assert transformed_frames == frames
        assert mock_transform.mock_calls == expected_calls

    def test_propagates_label_unchanged(self):
        stub_transform = lambda frames: frames
        frames = pil_video().example()

        transform = TimeApply(stub_transform)

        assert_preserves_label(transform, frames)


class TestResizeVideo:
    @given(pil_video(), st.sampled_from(pil_interpolation_settings), st.data())
    def test_resizes_to_given_size(self, video, interpolation, data):
        width, height = video[0].size
        expected_width = data.draw(st.integers(min_value=1, max_value=width))
        expected_height = data.draw(st.integers(min_value=1, max_value=height))
        transform = ResizeVideo(
            size=(expected_height, expected_width), interpolation=interpolation
        )

        transformed_video = list(transform(video))
        assert len(transformed_video) == len(video)
        for frame in transformed_video:
            assert (expected_width, expected_height) == frame.size

    def test_propagates_label_unchanged(self):
        frames = pil_video().example()

        transform = ResizeVideo((1, 1))

        assert_preserves_label(transform, frames)


class TestRandomResizedCropVideo:
    @given(pil_video(), st.sampled_from(pil_interpolation_settings), st.data())
    def test_resulting_video_are_specified_size(self, video, interpolation, data):
        width, height = video[0].size
        expected_width = data.draw(st.integers(min_value=1, max_value=width))
        expected_height = data.draw(st.integers(min_value=1, max_value=height))
        transform = RandomResizedCropVideo(
            (expected_height, expected_width), interpolation=interpolation
        )

        transformed_video = list(transform(video))

        assert len(transformed_video) == len(video)
        for frame in transformed_video:
            assert (expected_width, expected_height) == frame.size

    def test_propagates_label_unchanged(self):
        frames = pil_video().example()

        transform = RandomResizedCropVideo((1, 1))

        assert_preserves_label(transform, frames)
