import numpy as np

import torch
from typing import Union

from moviepy.editor import ImageSequenceClip
from moviepy.video.io.html_tools import ipython_available


def show_video(frames: Union[torch.Tensor, np.ndarray]):
    if isinstance(frames, torch.Tensor):
        frames = frames.cpu().numpy()
    video = ImageSequenceClip(list(frames))
    if ipython_available:
        video.ipython_display()
    else:
        video.show()
