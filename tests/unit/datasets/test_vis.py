from unittest.mock import patch

import numpy as np

import moviepy
import sys

import pytest

original_moviepy = moviepy

frames = np.random.randn(10, 32, 32, 3)


def test_vis_raises_error_if_moviepy_not_available():
    try:
        sys.modules["moviepy"] = None
        from torchvideo.datasets.vis import show_video
    finally:
        sys.modules["moviepy"] = original_moviepy
    with pytest.raises(ModuleNotFoundError, match="please install moviepy"):
        show_video(frames)
