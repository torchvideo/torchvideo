import numpy as np

import torch
from pathlib import Path
from typing import Union, List

from torchvideo.internal.utils import frame_idx_to_list


def lintel_loader(path: Path, frames_idx: Union[slice, List[slice], List[int]]) -> torch.Tensor:
    import lintel
    with path.open('rb') as f:
        video = f.read()

    # TODO: Read these from container metadata
    width = 10
    height = 10

    frames_idx = frame_idx_to_list(frames_idx)
    frames_data = lintel.loadvid_frame_nums(video,
                                            frame_nums=frames_idx,
                                            width=width,
                                            height=height,
                                            should_seek=False)
    frames = np.frombuffer(frames_data, dtype=np.uint8)
    # TODO: Support 1 channel grayscale video
    frames = np.reshape(frames, newshape=(len(frames_idx), height, width, 3))
    return torch.Tensor(frames)


def default_loader(path: Path, frames_idx: Union[slice, List[slice], List[int]]) -> torch.Tensor:
    from torchvideo import get_video_backend
    backend = get_video_backend()
    if backend == 'lintel':
        loader = lintel_loader
    else:
        raise ValueError("Unknown backend '{}'".format(backend))
    return loader(path, frames_idx)


