import numbers
from typing import cast, Tuple, TypeVar, Union, Iterator, Iterable

from .types import ImageSizeParam, ImageShape


def _canonicalize_size(size: ImageSizeParam) -> ImageShape:
    """

    Args:
        size:

    Returns:

    """
    if isinstance(size, numbers.Number):
        return ImageShape(int(size), int(size))
    else:
        size = cast(Tuple[int, int], size)
        return ImageShape(size[0], size[1])


T = TypeVar("T")


def _to_iter(seq: Union[Iterator[T], Iterable[T]]) -> Iterator[T]:
    try:
        return seq.__iter__()
    except AttributeError:
        pass
    return cast(Iterator, seq)


def _peek_iter(iterator: Iterator[T]) -> Tuple[T, Iterator[T]]:
    from itertools import chain

    first_elem = next(iterator)
    return first_elem, chain([first_elem], iterator)
