from PIL.Image import Image
import numpy as np

import torch
from typing import Union, List

try:
    from moviepy.editor import ImageSequenceClip
    from moviepy.video.io.html_tools import ipython_available

    moviepy_available = True
except ImportError:
    moviepy_available = False


def show_video(
    frames: Union[torch.Tensor, np.ndarray, List[Image]], fps=30, ndarray_format="THWC"
):
    """Show ``frames`` as a video in Jupyter, or in a PyGame window using ``moviepy``.

    Args:
        frames: One of:

            - :class:`torch.Tensor` with layout ``CTHW``.
            - :class:`numpy.ndarray` of layout ``THWC`` or ``CTHW``, if the latter,
              then set ``ndarray_format`` to ``CTHW``. The array should have a
              ``np.uint8`` dtype and range ``[0, 255]``.
            - a list of :class:`PIL.Image.Image`.

        fps (optional): Frame rate of video
        ndarray_format: 'CTHW' or 'THWC' depending on layout of ndarray.

    Returns:
        ImageSequenceClip displayed.

    """
    clip = convert_to_clip(frames, fps=fps, ndarray_format=ndarray_format)
    if ipython_available:
        return clip.ipython_display()
    else:
        return clip.show()


def convert_to_clip(frames, fps=30, ndarray_format="THWC"):
    """Convert ``frames`` to a ``moviepy`` ``ImageSequenceClip``.

    Args:
        frames: One of:

            - :class:`torch.Tensor` with layout ``CTHW``.
            - :class:`numpy.ndarray` of layout ``THWC`` or ``CTHW``, if the latter,
              then set ``ndarray_format`` to ``CTHW``. The array should have a
              ``np.uint8`` dtype and range ``[0, 255]``.
            - a list of :class:`PIL.Image.Image`.

        fps (optional): Frame rate of video
        ndarray_format: 'CTHW' or 'THWC' depending on layout of ndarray.

    Returns:
        ImageSequenceClip
    """

    if not moviepy_available:
        raise ModuleNotFoundError("moviepy not found, please install moviepy")
    frames_list = _to_list_of_np_frames(frames, ndarray_format=ndarray_format)
    clip = ImageSequenceClip(frames_list, fps=fps)
    return clip


def _to_list_of_np_frames(
    frames: Union[torch.Tensor, np.ndarray, List[Image]], ndarray_format="THWC"
) -> List[np.ndarray]:
    """

    Args:
        frames: A tensor with range ``[0, 1]``, a numpy array with CTHW or THWC
            format with range ``[0, 255]``, or a list of PIL Images.
        ndarray_format: 'CTHW' or 'THWC' depending on layout of ndarray.
    """
    if isinstance(frames, torch.Tensor):
        # Input format: (C, T, H, W), Input range: 0--1 (float)
        # Desired shape: (T, H, W, C), Output range: 0-255 (uint8)
        frames = torch.clamp((frames * 255), 0, 255).to(torch.uint8)
        thwc = frames.numpy()
        return list(np.moveaxis(thwc, 0, -1))
    elif isinstance(frames, np.ndarray):
        # Input format: (C, T, H, W), Input range: 0--255 (uint8)
        # Desired shape: (T, H, W, C), Output range: 0-255 (uint8)
        if ndarray_format.lower() == "cthw":
            # Input format: (C, T, H, W)
            # Desired shape: (T, H, W, C)
            thwc = np.moveaxis(frames, 0, -1)
            return list(thwc)
        elif ndarray_format.lower() == "thwc":
            return list(frames)
        else:
            raise ValueError(
                "Unknown ndarray format {!r}, expected on of 'CTHW' or "
                "'THWC'".format(ndarray_format)
            )
    elif isinstance(frames, list):
        if not isinstance(frames[0], Image):
            raise TypeError("Expected a list of PIL Images when passed a sequence")
        return list(map(np.array, frames))
    else:
        raise TypeError(
            "Unknown type: {}, expected np.ndarray, torch.Tensor, "
            "or sequence of PIL.Image.Image".format(type(frames).__name__)
        )
