from PIL.Image import Image
import numpy as np

import torch
from typing import Union

try:
    from moviepy.editor import ImageSequenceClip
    from moviepy.video.io.html_tools import ipython_available

    moviepy_available = True
except ImportError:
    moviepy_available = False


def show_video(frames: Union[torch.Tensor, np.ndarray], fps=30):
    """

    Args:
        frames: Either a :class:`torch.Tensor` or :class:`numpy.ndarray` of shape
            :math:`T \times C \times H \times W` or a list of :class:`PIL.Image.Image`s
        fps (optional): Frame rate of video

    Returns:

    """
    if not moviepy_available:
        raise ModuleNotFoundError("moviepy not found, please install moviepy")
    # Input format: (C, T, H, W)
    # Desired shape: (T, H, W, C)
    if isinstance(frames, torch.Tensor):
        frames = torch.clamp((frames * 255), 0, 255).to(torch.uint8)
        frames_list = list(frames.permute(1, 2, 3, 0).cpu().numpy())
    elif isinstance(frames, np.ndarray):
        frames_list = list(np.roll(frames, -1))
    elif isinstance(frames, list):
        if not isinstance(frames[0], Image):
            raise ValueError("Expected a list of PIL Images when passed a sequence")
        frames_list = list(map(np.array, frames))
    else:
        raise TypeError(
            "Unknown type: {}, expected np.ndarray, torch.Tensor, "
            "or sequence of PIL.Image.Image".format(type(frames).__name__)
        )

    video = ImageSequenceClip(frames_list, fps=fps)
    if ipython_available:
        return video.ipython_display()
    else:
        return video.show()
