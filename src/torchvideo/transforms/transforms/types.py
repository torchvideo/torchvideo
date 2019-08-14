from typing import Union, Iterable, Iterator, Tuple, TypeVar, NamedTuple

from PIL.Image import Image

#: An iterator of :class:`Image`
PILVideoI = Iterator[Image]
#: Either an iterable or iterator of :class:`Image`
PILVideo = Union[Iterable[Image], PILVideoI]
#: Either an int describing both with and weight, or a tuple ``(height, width)``
ImageSizeParam = Union[int, Tuple[int, int]]

#: Type variable that takes on a type of input video
InputFramesType = TypeVar("InputFramesType")
#: Type variable that takes on a type of output video
OutputFramesType = TypeVar("OutputFramesType")
#: Type variable that takes on a data structure type describing the complete
#  state of a transform
ParamsType = TypeVar("ParamsType")


class ImageShape(NamedTuple):
    """A data class representing the dimensions of an image."""

    height: int
    width: int


class Point(NamedTuple):
    """A data class representing a point in 2D space"""

    x: int
    y: int
