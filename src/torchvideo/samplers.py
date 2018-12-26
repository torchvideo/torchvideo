from abc import ABC
from typing import Union, List
from numpy.random import randint, np


class FrameSampler(ABC):  # pragma: no cover
    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        raise NotImplementedError()


class FullVideoSampler(FrameSampler):
    def sample(self, video_length: int) -> Union[slice, List[int], List[slice]]:
        if video_length <= 0:
            raise ValueError(
                "Video must be at least 1 frame long but was {} frames long".format(
                    video_length
                )
            )
        return slice(0, video_length, 1)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return "{cls_name}()".format(cls_name=self.__class__.__name__)


class TemporalSegmentSampler(FrameSampler):
    def __init__(self, segment_count, segment_length):
        if segment_count < 1:
            raise ValueError("Segment count must be greater than 0")
        if segment_length < 1:
            raise ValueError("Segment length must be greater than 0")

        self.segment_count = segment_count
        self.segment_length = segment_length

    def sample(self, video_length: int):
        average_segment_duration = (
            video_length - self.segment_length + 1
        ) // self.segment_count
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
        return "{cls_name}(segment_count={segment_count}, segment_length={segment_length})".format(
            cls_name=self.__class__.__name__,
            segment_count=self.segment_count,
            segment_length=self.segment_length,
        )

    def _sample_overlapping_idx(self, video_length):
        highest_segment_start_index = video_length - self.segment_length - 1
        assert highest_segment_start_index >= 1
        return list(
            randint(low=0, high=highest_segment_start_index, size=self.segment_count)
        )

    def _sample_non_overlapping_idx(self, average_segment_duration):
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
# - Reduced FPS sampler
