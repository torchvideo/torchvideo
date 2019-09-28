from inspect import signature, Parameter
from typing import List

from .transform import Transform, empty_target


class Compose:
    """Similar to :py:class:`torchvision.transforms.transforms.Compose` except
    supporting transforms that take either a mandatory or optional target parameter
    in __call__. This facilitates chaining a mix of transforms: those that don't support
    target parameters, those that do, and those that require them.
    """

    def __init__(self, transforms: List[Transform]):
        self.transforms = transforms
        self._requires_target = [_requires_target(t) for t in self.transforms]
        self._supports_target = [_supports_target(t) for t in self.transforms]
        self._first_target_requiring_transform = None
        for requires_target, t in zip(self._requires_target, self.transforms):
            if requires_target:
                self._first_target_requiring_transform = t
                break

    def __call__(self, frames, target=empty_target):
        if target == empty_target:
            self._check_transforms_dont_require_target()
            for t in self.transforms:
                frames = t(frames)
            return frames
        else:
            for t in self.transforms:
                if _supports_target(t):
                    frames, target = t(frames, target)
                else:
                    frames = t(frames)
            return frames, target

    def _check_transforms_dont_require_target(self):
        if self._first_target_requiring_transform is not None:
            raise TypeError(
                "{!r} requires a target to be passed. But not "
                "target was passed in the composed "
                "transform".format(self._first_target_requiring_transform)
            )

    def __repr__(self):
        return "{cls_name}(transforms={transform_reprs})".format(
            cls_name=self.__class__.__name__, transform_reprs=repr(self.transforms)
        )


def _supports_target(transform):
    sig = signature(transform)
    parameters = sig.parameters
    return len(parameters) >= 2


def _requires_target(transform):
    sig = signature(transform)
    parameters = sig.parameters
    if len(parameters) < 2:
        return False

    parameter_names = list(parameters)
    if "target" in parameter_names:
        target_param = parameters.get("target")
    else:
        target_param = parameters.get(parameter_names[1])
    return target_param.default == Parameter.empty
