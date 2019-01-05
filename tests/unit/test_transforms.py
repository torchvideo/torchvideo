import itertools

import PIL.Image
import numpy as np
from random import randint

import pytest
import torch
from hypothesis import given, assume, note
import hypothesis.strategies as st
from torchvision.transforms import Compose

from torchvideo.transforms import (
    RandomCropVideo,
    CollectFrames,
    PILVideoToTensor,
    CenterCropVideo,
    RandomHorizontalFlipVideo,
    NormalizeVideo,
    NDArrayToPILVideo,
    TimeToChannel,
)
from unit.strategies import pil_video, tensor_video, video_shape


def prod(seq):
    if len(seq) == 0:
        raise ValueError("Expected sequence to have at least 1 element")
    product = seq[0]
    for el in seq[1:]:
        product *= el
    return product


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

    @given(
        data=st.data(),
        video=pil_video(min_width=2, min_height=2),
        fill=st.integers(0, 255),
    )
    def test_crop_with_user_provided_padding(self, data, video, fill):
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


class TestNormalizeVideo:
    def test_repr(self):
        assert repr(NormalizeVideo(128, 15)) == "NormalizeVideo(mean=128, std=15)"

    @given(tensor_video())
    def test_mean_centers_tensor(self, video):
        transform = NormalizeVideo(-0.5, 1)
        transformed_video = transform(video)
        # video.mean() - transformed_video.mean() == 0.5
        # TODO: Finish test


class TestCollectFrames:
    def test_repr(self):
        assert repr(CollectFrames()) == "CollectFrames()"

    @given(pil_video(max_length=10, max_height=1, max_width=1))
    def test_collect_frames_make_list_from_iterator(self, video):
        transform = CollectFrames()
        assert transform(iter(video)) == video


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


class TestNDArrayToPILVideo:
    def test_repr(self):
        assert repr(NDArrayToPILVideo()) == "NDArrayToPILVideo()"

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
