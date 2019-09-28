from .transform import StatelessTransform
from .types import InputFramesType


class IdentityTransform(StatelessTransform[InputFramesType, InputFramesType]):
    """Identity transformation that returns frames (and labels) unchanged. This is
    primarily of use when conditionally adding in transforms and you want to default
    to a transform that doesn't do anything. Whilst you could just use an identity
    lambda this transform has a nicer repr that shows that no transform is taking place.
    """

    def _transform(self, frames: InputFramesType, params: None) -> InputFramesType:
        return frames

    def __repr__(self):
        return self.__class__.__name__ + "()"
