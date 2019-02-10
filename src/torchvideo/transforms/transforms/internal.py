import numbers
from typing import cast, Tuple, TypeVar, Union, Iterator, Iterable

from .types import ImageSizeParam, ImageShape


def canonicalize_size(size: ImageSizeParam) -> ImageShape:
    """Canonicalize a user provided size parameter into a named tuple with width and
    height properties.

    Args:
        size: An int for a square image, or a ``(height, width)`` tuple.

    Returns:
        A ``(height, width)`` named tuple.

    """
    if isinstance(size, numbers.Number):
        return ImageShape(int(size), int(size))
    else:
        size = cast(Tuple[int, int], size)
        return ImageShape(size[0], size[1])


T = TypeVar("T")


def to_iter(seq: Union[Iterator[T], Iterable[T]]) -> Iterator[T]:
    """Convert an iterable or iterator to an iterator"""
    try:
        return seq.__iter__()
    except AttributeError:
        pass
    return cast(Iterator, seq)


def peek_iter(iterator: Iterator[T]) -> Tuple[T, Iterator[T]]:
    """Pop out the first element of an iterator, then construct a new iterator with
    the first element consed onto the remaining iterator"""
    from itertools import chain

    first_elem = next(iterator)
    return first_elem, chain([first_elem], iterator)
