import collections
import itertools
import numpy as np
from typing import List, Union


def slice_to_list(slice_: slice) -> List[int]:
    return list(range(slice_.start, slice_.stop, slice_.step))


def _is_int(maybe_int):
    try:
        return int(maybe_int) == maybe_int
    except TypeError:
        pass
    return False


def frame_idx_to_list(frames_idx: Union[slice, List[slice], List[int]]) -> List[int]:
    if isinstance(frames_idx, list):
        if isinstance(frames_idx[0], slice):
            return list(
                itertools.chain.from_iterable([slice_to_list(s) for s in frames_idx])
            )
        if _is_int(frames_idx[0]):
            return frames_idx
    if isinstance(frames_idx, slice):
        return slice_to_list(frames_idx)
    raise ValueError(
        "Can't handle {} objects, must be slice, List[slice], or List[int]".format(
            type(frames_idx)
        )
    )
