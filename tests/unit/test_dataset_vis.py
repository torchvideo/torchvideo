import torch
from unittest.mock import Mock

import itertools
import numpy as np
import moviepy

import moviepy.editor
import moviepy.video.io.html_tools
import sys

from PIL import Image

import torchvideo
from torchvideo.datasets.vis import show_video, _to_list_of_np_frames
import importlib

import pytest

original_moviepy_editor = moviepy.editor


class TestShowVideo:
    frames = np.random.randn(10, 32, 32, 3)

    def test_raises_error_if_moviepy_not_available(self):
        # OK, so this is a nasty nasty test with import fuckery going on.
        try:
            # First we make moviepy.editor unimportable
            sys.modules["moviepy.editor"] = None
            # but we have already imported torchvideo.datasets.vis so that code is
            # cached with a reference to the *original* moviepy, so we need to reload
            # it to invoke the failure path of the import code
            importlib.reload(torchvideo.datasets.vis)
            from torchvideo.datasets.vis import show_video

            # Now we've imported torchvideo.datasets.vis where moviepy.editor was
            # *not* available.
            with pytest.raises(ImportError, match="please install moviepy"):
                show_video(self.frames)
        finally:
            # We've borked up the system modules and we have to put them right otherwise
            # any tests that depend on torchvideo.datasets.vis or moviepy.editor are
            # going to fail.
            # We first restore the original moviepy editor object back
            sys.modules["moviepy.editor"] = original_moviepy_editor
            # I don't think we need to reload moviepy (other tests pass without this
            # line), but for good measure let's reload it just in case it does
            # something useful.
            importlib.reload(moviepy)
            # We also want to reload torchvideo.datasets.vis so moviepy *is* available
            importlib.reload(torchvideo.datasets.vis)

    def test_uses_ipython_display_if_ipython_is_available(self, monkeypatch):
        display_mock = Mock()
        with monkeypatch.context() as ctx:
            ctx.setattr(moviepy.video.io.html_tools, "ipython_available", True)
            ctx.setattr(
                moviepy.editor.ImageSequenceClip, "ipython_display", display_mock
            )

            importlib.reload(torchvideo.datasets.vis)
            show_video(self.frames)

            display_mock.assert_called_once_with()

    def test_fallsback_on_pygame_display(self, monkeypatch):
        show_mock = Mock()
        with monkeypatch.context() as ctx:
            ctx.setattr(moviepy.video.io.html_tools, "ipython_available", False)
            ctx.setattr(moviepy.editor.ImageSequenceClip, "show", show_mock)

            importlib.reload(torchvideo.datasets.vis)
            show_video(self.frames)

            show_mock.assert_called_once_with()


class TestToListOfNpFrames:
    def test_from_ndarray_cthw(self):
        # CTHW
        duration, width, height, channels = 5, 7, 4, 3
        frames = np.random.randn(channels, duration, height, width) * 255

        frames_list = _to_list_of_np_frames(frames, ndarray_format="cthw")

        assert len(frames_list) == duration
        first_frame = frames_list[0]
        expected_shape = (height, width, channels)
        assert first_frame.shape == expected_shape
        expected_frame = np.moveaxis(frames[:, 0, :, :], 0, -1)
        np.testing.assert_array_equal(first_frame, expected_frame)

    def test_from_ndarray_thwc(self):
        duration, width, height, channels = 5, 7, 4, 3
        frames = np.random.randn(duration, height, width, channels) * 255

        frames_list = _to_list_of_np_frames(frames, ndarray_format="thwc")

        assert len(frames_list) == duration
        first_frame = frames_list[0]
        assert first_frame.shape == (height, width, channels)
        np.testing.assert_array_equal(first_frame, frames[0])

    def test_from_tensor_with_range_0_1(self):
        # CTHW
        duration, width, height, channels = 5, 7, 4, 3
        frames = torch.randn(channels, duration, height, width)
        frames[frames > 0.5] = 1
        frames[frames <= 0.5] = 0

        frames_list = _to_list_of_np_frames(frames)

        assert len(frames_list) == duration
        first_frame = frames_list[0]
        assert first_frame.shape == (height, width, channels)
        assert first_frame.max() == 255
        assert first_frame.min() == 0
        expected_first_frame_chw = frames.numpy()[:, 0, :, :].squeeze()
        expected_first_frame = (
            np.moveaxis(expected_first_frame_chw, 0, -1) * 255
        ).astype(np.uint8)
        np.testing.assert_array_equal(first_frame, expected_first_frame)

    def test_from_list_of_pil_images(self):
        duration, width, height, channels = 5, 7, 4, 3
        frames = [
            Image.fromarray(
                np.random.randint(
                    0, 255, size=(height, width, channels), dtype=np.uint8
                )
            )
            for _ in range(duration)
        ]

        frames_list = _to_list_of_np_frames(frames)

        assert len(frames_list) == duration
        np.testing.assert_array_equal(frames_list[0], np.array(frames[0]))

    def test_raises_error_on_ndarray_formats_other_than_cthw_or_thwc(self):
        frames = np.random.randn(3, 3, 3, 3)
        for format in itertools.permutations("thwc"):
            format = "".join(format)
            if format in {"thwc", "cthw"}:
                continue
            with pytest.raises(ValueError):
                _to_list_of_np_frames(frames, ndarray_format=format)

    def test_raises_error_on_unknown_format(self):
        with pytest.raises(TypeError):
            _to_list_of_np_frames(["a"])
