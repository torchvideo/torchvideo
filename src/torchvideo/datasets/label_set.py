"""Label sets are an abstraction that encapsulate how videos are labelled.

This allows for the video data loading to be decoupled from reading labels associated with
those videos.
"""
from abc import ABC
from typing import Any, TypeVar

Label = TypeVar("Label")


class LabelSet(ABC):  # pragma: no cover
    def __getitem__(self, video_name) -> Label:
        raise NotImplementedError()


# TODO:
#   - CSV label set
#   - Filelist label set
#   - Gulp label set
#   - Dummy label set (always return the same label)
