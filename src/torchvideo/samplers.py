import itertools
from abc import ABC
from typing import Union, List, Callable, Tuple, cast
import numpy as np
from numpy.random import randint

from torchvideo.internal.utils import _is_int


class FrameSampler(ABC):  # pragma: no cover
    """Abstract base class that all frame samplers implement.

    If you are creating your own sampler, you should inherit from this base class."""

    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        """Generate frame indices to sample from a video of ``video_length`` frames.

        Args:
            video_length: The duration in frames of the video to be sampled from

        Returns:
            Frame indices
        """
        raise NotImplementedError()


class FullVideoSampler(FrameSampler):
    """Sample all frames in a video.

    Args:
        frame_step: The step size between frames, this controls FPS reduction, a step
            size of 2 will halve FPS, step size of 3 will reduce FPS to 1/3.
    """

    def __init__(self, frame_step=1):
        self.frame_step = frame_step

    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        """

        Args:
            video_length: The duration in frames of the video to be sampled from.

        Returns:

a
            ``slice`` from ``0`` to ``video_length`` with step size ``frame_step``
        """
        if video_length <= 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".format(
                    video_length
                )
            )
        return slice(0, video_length, self.frame_step)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "{cls_name}()".format(cls_name=self.__class__.__name__)


class ClipSampler(FrameSampler):
    """Sample clips of a fixed duration uniformly randomly from a video."""

    def __init__(self, clip_length: int, frame_step: int = 1, test: bool = False):
        """
        Args:
            clip_length: Duration of clip in frames
            frame_step: The step size between frames, this controls FPS reduction, a
                step size of 2 will halve FPS, step size of 3 will reduce FPS to 1/3.
            test: Whether or not to sample in test mode (in test mode the central
                clip is sampled from the video)
        """
        self.clip_length = clip_length
        self.frame_step = frame_step
        self.test_mode = test

    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        if video_length <= 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".format(
                    video_length
                )
            )
        sample_length = compute_sample_length(self.clip_length, self.frame_step)
        if video_length < sample_length:
            return _oversample(video_length, sample_length)

        max_offset = video_length - sample_length

        if self.test_mode:
            start_index = int(np.floor(max_offset / 2))
        else:
            start_index = 0 if max_offset == 0 else randint(0, max_offset)
        return slice(start_index, start_index + sample_length, self.frame_step)

    def __repr__(self):
        return self.__class__.__name__ + "(clip_length={!r}, frame_step={!r})".format(
            self.clip_length, self.frame_step
        )


class TemporalSegmentSampler(FrameSampler):
    """[TSN]_ style sampling.

    The video is equally divided into a number of segments, ``segment_count`` and from
    within each segment a snippet, a contiguous sequence of frames,
    ``snippet_length`` fr+ames long is sampled.

    There are two variants of sampling. One for training and one for testing. During
    training the snippet location within the segment is uniformly randomly sampled.
    During testing snippets are sampled centrally within their segment (i.e.
    deterministically).

    [TSN]_ Uses the following configurations:

    +---------+------------+-------------------+--------------------+
    | Network | Train/Test | ``segment_count`` | ``snippet_length`` |
    +=========+============+===================+====================+
    | RGB     | Train      | 3                 | 1                  |
    +         +------------+-------------------+--------------------+
    |         | Test       | 25                | 1                  |
    +---------+------------+-------------------+--------------------+
    | Flow    | Train      | 3                 | 5                  |
    +         +------------+-------------------+--------------------+
    |         | Test       | 25                | 5                  |
    +---------+------------+-------------------+--------------------+


    """

    def __init__(self, segment_count: int, snippet_length: int, test: bool = False):
        """
        Args:
            segment_count: Number of segments to split the video into, from which a
                snippet is sampled.
            snippet_length: The number of frames in each snippet
            test: Whether to sample in test mode or not (see class docstring for
                training/testing differences)
        """

        if segment_count < 1:
            raise ValueError("segment_count must be greater than 0")
        if snippet_length < 1:
            raise ValueError("snippet_length must be greater than 0")

        self.test_mode = test
        self.segment_count = segment_count
        self.snippet_length = snippet_length

    def sample(self, video_length: int) -> Union[List[slice], List[int]]:
        """

        Args:
            video_length: The duration in frames of the video to be sampled from

        Returns:
            Frame indices as list of slices
        """
        if video_length <= 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".format(
                    video_length
                )
            )
        # This is quite a fiddly and important bit of code so it's well commented.
        # We first get the indices of the starts of each segment, and the length of
        # the segment (which is the same for all segments)
        # segment_start_idx: shape=(self.segment_count,), range=(0, video_length - 1)
        segment_start_idx, segment_length = self.segment_video(video_length)
        # If we don't have enough frames in each segment to sample a snippet we have to
        # fallback to some basic strategy that yields the same number of frames as we'd
        # get if we *did* have large enough segments, that way the user can always
        # reshape the video in to segments of equal length.
        if segment_length < self.snippet_length:
            # Not a particularly clever way of dealing with short snippets. A better
            # way would be to see how many potential points we have
            return [0] * self.segment_count * self.snippet_length
        # We now sample the position of the snippet within each segment by sampling
        # an offset and adding it to the starting position of the segment.
        # snippet_start_idx: shape=(self.segment_count,),
        #                    range=(0, segment_length - self.snippet_length)
        segment_offsets = self._get_segment_offsets(segment_length)
        # snippet_start_idx: shape=(self.segment_count,), range=(0, video_length - 1)
        snippet_start_idx = segment_start_idx + segment_offsets
        return [self._make_snippet_slice(start) for start in snippet_start_idx]

    def _get_segment_offsets(self, segment_length: int) -> Union[int, np.ndarray]:
        max_offset = segment_length - self.snippet_length
        if self.test_mode:
            return max_offset // 2
        if max_offset == 0:
            return max_offset
        return np.random.randint(0, max_offset, dtype=np.uint, size=self.segment_count)

    def segment_video(self, video_length: int) -> Tuple[np.ndarray, int]:
        """Segment a video of ``video_length`` frames into ``self.segment_count``
        segments.

        Args:
            video_length: num

        Returns:
            ``(segment_start_idx, segment_length)``. The ``segment_start_idx`` contains
            the indices of the beginning of each segment in the video.
            ``segment_length`` is the length for all segments.
        """
        segment_length = video_length // self.segment_count
        segment_start_idx = (
            np.arange(self.segment_count, dtype=np.uint) * segment_length
        )
        return segment_start_idx, segment_length

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            "{cls_name}("
            "segment_count={segment_count}, "
            "snippet_length={snippet_length}, "
            "test={test}"
            ")"
        ).format(
            cls_name=self.__class__.__name__,
            segment_count=self.segment_count,
            snippet_length=self.snippet_length,
            test=self.test_mode,
        )

    def _make_snippet_slice(self, start: int) -> slice:
        # int casts are because we pass in np.intX numbers which throw errors down the
        # line
        return slice(int(start), int(start + self.snippet_length), 1)


class LambdaSampler(FrameSampler):
    """Custom sampler constructed from a user provided function."""

    def __init__(self, sampler: Callable[[int], Union[slice, List[slice], List[int]]]):
        """

        Args:
            sampler: Function that takes an ``int``, the video length in frames and
                returns a slice, list of ints, or list of slices representing indices
                to sample from the video. All the indices should be less than the
                video length - 1.
        """
        self._fn = sampler

    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        frame_idx = self._fn(video_length)
        if not all([i < (video_length - 1) for i in frame_idx_to_list(frame_idx)]):
            raise ValueError(
                "Invalid frame_idx {} from user provided sampler for video of "
                "length {}".format(frame_idx, video_length)
            )
        return frame_idx

    def __repr__(self):
        return self.__class__.__name__ + "(sampler={!r})".format(self._fn)


def frame_idx_to_list(frames_idx: Union[slice, List[slice], List[int]]) -> List[int]:
    """
    Converts a frame_idx object to a list of indices. Useful for testing.

    Args:
        frames_idx: Frame indices represented as a slice, list of slices, or list of
            ints.

    Returns:
        frame idx as a list of ints.

    """
    # mypy needs type assertions within these conditional blocks to get the correct
    # types
    if isinstance(frames_idx, list):
        if len(frames_idx) == 0:
            return cast(List[int], frames_idx)
        if isinstance(frames_idx[0], slice):
            frames_idx = cast(List[slice], frames_idx)
            return list(
                itertools.chain.from_iterable([_slice_to_list(s) for s in frames_idx])
            )
        if _is_int(frames_idx[0]):
            return cast(List[int], frames_idx)
    if isinstance(frames_idx, slice):
        return _slice_to_list(frames_idx)
    raise ValueError(
        "Can't handle {} objects, must be slice, List[slice], or List[int]".format(
            type(frames_idx)
        )
    )


def compute_sample_length(clip_length, step_size):
    """Computes the number of frames to be sampled for a clip of length
    ``clip_length`` with frame step size of ``step_size`` to be generated.

    Args:
        clip_length: Number of frames to sample
        step_size: Number of frames to skip in between adjacent frames in the output

    Returns:
        Number of frames to sample to read a clip of length ``clip_length`` while
        skipping ``step_size - 1`` frames.

    """
    return 1 + step_size * (clip_length - 1)


def _slice_to_list(slice_: slice) -> List[int]:
    step = 1 if slice_.step is None else slice_.step
    start = 0 if slice_.start is None else slice_.start
    stop = slice_.stop
    if stop is None:
        raise ValueError("Cannot convert slice with no stop attribute to a list")
    return list(range(start, stop, step))


def _oversample(video_length: int, sample_length: int) -> List[int]:
    assert (
        sample_length > video_length
    ), "No point oversampling a video that has more frames than the sample length"

    missing_frames_count = sample_length - video_length
    return ([0] * missing_frames_count) + list(range(0, video_length))


_default_sampler = FullVideoSampler
