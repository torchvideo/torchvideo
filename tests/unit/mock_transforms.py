import inspect
from abc import ABC
from unittest.mock import call


class _empty_target:
    pass


def _has_at_least_one_param(fn):
    sig = inspect.signature(fn)
    return len(list(sig.parameters)) > 0


class MockTransform(ABC):
    def __init__(self, return_value, name=None):
        self.calls = []
        self.return_value = return_value
        self.name = name

    def assert_called_once_with(self, *args, **kwargs):
        assert len(self.calls) == 1, "Expected one call, but has {}: {}".format(
            len(self.calls), self.calls
        )
        expected_call = call(*args, **kwargs)
        assert expected_call in self.calls, "{} not found in {}".format(
            expected_call, self.calls
        )

    def __repr__(self):
        if self.name is not None:
            return self.name + "()"
        else:
            return super().__repr__()

    def _get_frames_return_value(self, frames):
        if callable(self.return_value):
            if _has_at_least_one_param(self.return_value):
                return self.return_value(frames)
            else:
                return self.return_value()
        return self.return_value


class MockTransformWithTarget(MockTransform, ABC):
    def __init__(
        self, frames_return_value, target_return_value=_empty_target, **kwargs
    ):
        super().__init__(frames_return_value, **kwargs)
        self.target_return_value = target_return_value

    def _get_target_return_value(self, target):
        if callable(self.target_return_value):
            if _has_at_least_one_param(self.target_return_value):
                return self.target_return_value(target)
            return self.target_return_value()
        return self.target_return_value


class MockFramesOnlyTransform(MockTransform):
    def __call__(self, frames):
        self.calls.append(call(frames))
        return self._get_frames_return_value(frames)


class MockFramesAndOptionalTargetTransform(MockTransformWithTarget):
    def __call__(self, frames, target=_empty_target):
        self.calls.append(call(frames, target=target))
        if target is _empty_target:
            return self._get_frames_return_value(frames)
        else:
            return (
                self._get_frames_return_value(frames),
                self._get_target_return_value(target),
            )


class MockFramesAndRequiredTargetTransform(MockTransformWithTarget):
    def __init__(
        self, frames_return_value, target_return_value=_empty_target, name=None
    ):
        super().__init__(frames_return_value, name=name)
        self.target_return_value = target_return_value

    def __call__(self, frames, target):
        self.calls.append(call(frames, target))
        return (
            self._get_frames_return_value(frames),
            self._get_target_return_value(target),
        )
