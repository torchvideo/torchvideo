from typing import List

from PIL.Image import Image

from .types import PILVideo
from .transform import Transform


class CollectFrames(Transform[PILVideo, List[Image], None]):
    """Collect frames from iterator into list.

    Used at the end of a sequence of PIL video transformations.
    """

    def _gen_params(self, frames: PILVideo) -> None:
        return None

    def _transform(self, frames: PILVideo, params: None) -> List[Image]:
        return list(frames)

    def __repr__(self):
        return self.__class__.__name__ + "()"
