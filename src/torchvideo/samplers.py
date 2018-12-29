from abc import ABC
from typing import Union, List
from numpy.random import randint, np


class FrameSampler(ABC):  # pragma: no cover
    """Abstract base class that all frame samplers implement.

    If you are creating your own sampler, you should inherit from this base class."""

    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        """Generate frame indices to sample from a video.

        Args:
            video_length: The duration in frames of the video to be sampled from

        Returns:
            One of:
              - A slice representing a contiguous chunk of frames
              - A list of frame indices
              - A list of slices representing a set of contiguous chunks of frames
        """
        raise NotImplementedError()


class FullVideoSampler(FrameSampler):
    """FrameSampler that samples all frames in a video

    Args:
        frame_step: The step size between frames, this controls FPS reduction,
        a step size of 2 will halve FPS, step size of 3 will reduce FPS to 1/3.
    """

    def __init__(self, frame_step=1):
        self.frame_step = frame_step

    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        """

        Args:
            video_length: The duration in frames of the video to be sampled from

        Returns:
            slice from ``0`` to ``video_length`` with step size ``frame_step``
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


class TemporalSegmentSampler(FrameSampler):
    """FrameSampler that implements the sampling style originated in [TSN]_

    The video is equally divided into a number of segments, and from within each segment
    a contiguous sequence of frames ``segment_length`` long is sampled.

    [TSN]_ Uses the following configurations:

    * Training
        * RGB network: ``segment_count``: 3, ``segment_length``: 1
        * Flow network: ``segment_count``: 3, ``segment_length``: 5
    * Testing
        * RGB network: ``segment_count``: 25, ``segment_length``: 1
        * Flow network: ``segment_count``: 25, ``segment_length``: 1
    """

    def __init__(self, segment_count, segment_length):
        if segment_count < 1:
            raise ValueError("Segment count must be greater than 0")
        if segment_length < 1:
            raise ValueError("Segment length must be greater than 0")

        self.segment_count = segment_count
        self.segment_length = segment_length

    def sample(self, video_length: int):
        """

        Args:
            video_length: The duration in frames of the video to be sampled from

        Returns:
            Frame indices as list of ints

        """
        average_segment_duration = (
            video_length - self.segment_length + 1
        ) // self.segment_count  # type: int
        if video_length <= 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".format(
                    video_length
                )
            )

        if average_segment_duration >= 1:
            return self._sample_non_overlapping_idx(average_segment_duration)
        elif video_length >= self.segment_length + self.segment_count:
            return self._sample_overlapping_idx(video_length)

        else:
            return [0] * self.segment_count

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return (
            "{cls_name}(segment_count={segment_count}, "
            "segment_length={segment_length})"
        ).format(
            cls_name=self.__class__.__name__,
            segment_count=self.segment_count,
            segment_length=self.segment_length,
        )

    def _sample_overlapping_idx(self, video_length: int) -> List[int]:
        highest_segment_start_index = video_length - self.segment_length - 1
        assert highest_segment_start_index >= 1
        return list(
            randint(low=0, high=highest_segment_start_index, size=self.segment_count)
        )

    def _sample_non_overlapping_idx(self, average_segment_duration: int) -> List[int]:
        segment_start_idx = (
            np.array(list(range(self.segment_count))) * average_segment_duration
        )
        segment_start_offsets = randint(
            low=0, high=average_segment_duration, size=self.segment_count
        )
        return list(segment_start_idx + segment_start_offsets)


# TODO:
# - Random subsequence sampler
# - TSN sampler
