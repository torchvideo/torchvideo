import subprocess
from collections import namedtuple

import numpy as np

from pathlib import Path
from typing import Union, List, Iterator

from PIL import Image

from torchvideo.internal.utils import frame_idx_to_list

VideoInfo = namedtuple("VideoInfo", ("height", "width", "n_frames"))
_VIDEO_FILE_EXTENSIONS = {
    "mp4",
    "webm",
    "avi",
    "3gp",
    "wmv",
    "mpg",
    "mpeg",
    "mov",
    "mkv",
}


def lintel_loader(
    path: Path, frames_idx: Union[slice, List[slice], List[int]]
) -> Iterator[Image.Image]:
    import lintel

    with path.open("rb") as f:
        video = f.read()

    frames_idx = frame_idx_to_list(frames_idx)
    frames_data, width, height = lintel.loadvid_frame_nums(
        video, frame_nums=frames_idx, should_seek=False
    )
    frames = np.frombuffer(frames_data, dtype=np.uint8)
    # TODO: Support 1 channel grayscale video
    frames = np.reshape(frames, newshape=(len(frames_idx), height, width, 3))
    return (Image.fromarray(frame) for frame in frames)


def default_loader(
    path: Path, frames_idx: Union[slice, List[slice], List[int]]
) -> Iterator[Image.Image]:
    from torchvideo import get_video_backend

    backend = get_video_backend()
    if backend == "lintel":
        loader = lintel_loader
    else:
        raise ValueError("Unknown backend '{}'".format(backend))
    return loader(path, frames_idx)


def _get_videofile_frame_count(video_file_path: Path) -> int:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames",
        "-of",
        "default=nokey=1:noprint_wrappers=1",
        str(video_file_path),
    ]
    result = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
    )
    # Final character of output is a newline so we drop it
    n_frames = int(result.stdout.decode("utf-8").split("\n")[0])
    return n_frames


def _is_video_file(path: Path) -> bool:
    extension = path.name.lower().split(".")[-1]
    return extension in _VIDEO_FILE_EXTENSIONS
