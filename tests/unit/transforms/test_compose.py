from typing import Tuple, List, Any
from unittest.mock import Mock

import pytest

from torchvideo.transforms import Compose, CenterCropVideo
from ..mock_transforms import (
    MockFramesOnlyTransform,
    MockFramesAndOptionalTargetTransform,
    MockFramesAndRequiredTargetTransform,
)
from ..strategies import pil_video


class TestCompose:
    def test_calls_frames_only_transforms_sequentially(self):
        frames = pil_video().example()
        transforms, results = self.gen_transforms(5)
        composed_transform = Compose(transforms)

        transformed_frames = composed_transform(frames)

        transforms[0].assert_called_once_with(frames)
        transforms[1].assert_called_once_with(results[0])
        transforms[2].assert_called_once_with(results[1])
        transforms[3].assert_called_once_with(results[2])
        transforms[4].assert_called_once_with(results[3])
        assert transformed_frames == results[-1]

    def test_passes_target_to_supporting_transforms(self):
        results = ["transform_result_0", "transform_result_1", "transform_result_2"]
        transforms = [
            MockFramesOnlyTransform(results[0]),
            MockFramesAndOptionalTargetTransform(results[1], target_return_value=-2),
            MockFramesAndRequiredTargetTransform(results[2], target_return_value=-3),
        ]
        composed_transform = Compose(transforms)
        frames = pil_video().example()

        target = -1
        transformed_frames, transformed_target = composed_transform(frames, target)

        transforms[0].assert_called_once_with(frames)
        transforms[1].assert_called_once_with(results[0], target=target)
        transforms[2].assert_called_once_with(results[1], -2)
        assert results[-1] == transformed_frames
        assert transformed_target == -3

    def test_raises_error_if_target_is_not_passed_when_a_transform_requires_target(
        self
    ):
        transforms = [
            MockFramesAndRequiredTargetTransform(None, None, name="MyTransform")
        ]
        composed_transform = Compose(transforms)
        frames = pil_video().example()

        with pytest.raises(TypeError, match="MyTransform"):
            composed_transform(frames)

    def test_single_level_repr(self):
        t = CenterCropVideo(224)
        assert repr(Compose([t])) == f"Compose(transforms=[{t!r}])"

    def test_nested_repr(self):
        t1 = CenterCropVideo(224)
        t2 = CenterCropVideo(16)
        assert (
            repr(Compose([t1, Compose([t2])]))
            == f"Compose(transforms=[{t1!r}, Compose(transforms=[{t2!r}])])"
        )

    def gen_transforms(self, count: int) -> Tuple[List[Mock], List[Any]]:
        transforms = []
        results = []
        for i in range(count):
            result = "transform_result_{}".format(i)
            transform = MockFramesOnlyTransform(return_value=result)
            transforms.append(transform)
            results.append(result)
        return transforms, results

    def make_result_class(self, result_class_name):
        return type(result_class_name, (), {})
