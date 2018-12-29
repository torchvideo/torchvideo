import torch

import numpy as np

from PIL import Image
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


@st.composite
def numpy_video(
    draw,
    min_length=1,
    max_length=3,
    min_width=1,
    max_width=10,
    min_height=1,
    max_height=10,
    mode=None,
):
    height, length, width = draw(
        video_shape(
            min_length, max_length, min_height, max_height, min_width, max_width
        )
    )
    if mode is None:
        mode = draw(st.sampled_from(["RGB", "L"]))
    if mode == "RGB":
        array_st = arrays(dtype=np.uint8, shape=(length, width, height, 3))
    else:
        array_st = arrays(dtype=np.uint8, shape=(length, width, height))
    return draw(array_st)


@st.composite
def pil_video(
    draw,
    min_length=1,
    max_length=3,
    min_width=1,
    max_width=10,
    min_height=1,
    max_height=10,
    mode=None,
):
    video_arr = draw(
        numpy_video(
            min_length,
            max_length,
            min_width,
            max_width,
            min_height,
            max_height,
            mode=mode,
        )
    )
    mode = "RGB" if video_arr.ndim == 4 else "L"

    return [Image.fromarray(img, mode=mode) for img in video_arr]


@st.composite
def tensor_video(
    draw,
    min_length=1,
    max_length=10,
    min_width=1,
    max_width=10,
    min_height=1,
    max_height=10,
    mode=None,
):
    arr = draw(
        numpy_video(
            min_length,
            max_length,
            min_width,
            max_width,
            min_height,
            max_height,
            mode=mode,
        )
    )
    if arr.ndim == 3:
        arr = arr[..., np.newaxis]
    # arr has shape: (T, H, W, C)
    # desired shape: (C, T, H, W)
    return (
        torch.from_numpy(arr)
        .clamp(0, 255)
        .to(torch.float32)
        .div_(255)
        .permute(3, 0, 1, 2)
    )


@st.composite
def video_shape(
    draw, min_length, max_length, min_height, max_height, min_width, max_width
):
    length = draw(st.integers(min_length, max_length))
    width = draw(st.integers(min_width, max_width))
    height = draw(st.integers(min_height, max_height))
    return height, length, width
