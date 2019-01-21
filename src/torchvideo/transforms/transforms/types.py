from collections import namedtuple
from typing import Union, Iterable, Iterator, Tuple, TypeVar

from PIL.Image import Image

PILVideo = Union[Iterable[Image], Iterator[Image]]
PILVideoI = Iterator[Image]
ImageSizeParam = Union[int, Tuple[int, int]]
ImageShape = namedtuple("ImageSize", ["height", "width"])
InputFramesType = TypeVar("InputFramesType")
OutputFramesType = TypeVar("OutputFramesType")
ParamsType = TypeVar("ParamsType")
