import itertools
from abc import ABC, abstractmethod
from typing import Generic, Iterator, Union

from .types import InputFramesType, OutputFramesType, ParamsType


class empty_target:
    pass


class FramesAndParams(Generic[InputFramesType, ParamsType]):
    def __init__(self, frames: InputFramesType, params: ParamsType):
        self.frames = frames
        self.params = params


class Transform(Generic[InputFramesType, OutputFramesType, ParamsType], ABC):
    def __call__(self, frames, target=empty_target):
        if isinstance(frames, Iterator):
            frames, frames_copy = itertools.tee(frames)
        else:
            frames_copy = frames

        maybe_params = self._gen_params(frames_copy)
        if isinstance(maybe_params, FramesAndParams):
            params = maybe_params.params
            frames = maybe_params.frames
        else:
            params = maybe_params

        transformed_frames = self._transform(frames, params)

        if target is empty_target:
            return transformed_frames

        return transformed_frames, target

    @abstractmethod
    def _gen_params(
        self, frames: InputFramesType
    ) -> Union[ParamsType, FramesAndParams[InputFramesType, ParamsType]]:
        pass

    @abstractmethod
    def _transform(
        self, frames: InputFramesType, params: ParamsType
    ) -> OutputFramesType:
        pass


class StatelessTransform(Transform[InputFramesType, OutputFramesType, None], ABC):
    def _gen_params(self, frames: InputFramesType) -> None:
        return None
