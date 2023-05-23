import numbers
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, cast, List
import torch

import numpy as np
from gulpio2 import GulpDirectory

from .label_sets import LabelSet, GulpLabelSet
from .video_dataset import VideoDataset
from .types import NDArrayVideoTransform, empty_label, Label
from .helpers import invoke_transform
from ..samplers import FrameSampler, _default_sampler


class GulpVideoDataset(VideoDataset):
    """GulpIO Video dataset.

    The folder hierarchy should look something like this: ::

        root/data_0.gulp
        root/data_1.gulp
        ...

        root/meta_0.gulp
        root/meta_1.gulp
        ...
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        *,
        gulp_directory: Optional[GulpDirectory] = None,
        filter: Optional[Callable[[str], bool]] = None,
        label_field: Optional[str] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[NDArrayVideoTransform] = None,
    ):
        """
        Args:
            root_path: Path to GulpIO dataset folder on disk. The ``.gulp`` and
                ``.gmeta`` files are direct children of this directory.
            filter: Filter function that determines whether a video is included into
                the dataset. The filter is called on each video id, and should return
                ``True`` to include the video, and ``False`` to exclude it.
            label_field: Meta data field name that stores the label of an example,
                this is used to construct a :class:`GulpLabelSet` that performs the
                example labelling. Defaults to ``'label'``.
            label_set: Optional label set for labelling examples. This is mutually
                exclusive with ``label_field``.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform over the :class:`ndarray` with layout
                ``THWC``. Note you'll probably want to remap the channels to ``CTHW`` at
                the end of this transform.
            gulp_directory: Optional gulp directory residing at root_path. Useful if
                you wish to create a custom label_set using the gulp_directory,
                which you can then pass in with the gulp_directory itself to avoid
                reading the gulp metadata twice.
        """

        if transform is None:

            def transform(frames):
                return torch.Tensor(np.rollaxis(frames, -1, 0)).div_(255)

        if gulp_directory is not None:
            if Path(gulp_directory.output_dir) != Path(root_path):
                raise ValueError(
                    "Expected gulp_dir.output ({}) to be the same as "
                    "root_path ({})".format(gulp_directory.output_dir, root_path)
                )
            self.gulp_dir = gulp_directory
        else:
            self.gulp_dir = GulpDirectory(str(root_path))

        label_set = self._get_label_set(self.gulp_dir, label_field, label_set)
        super().__init__(
            root_path, label_set=label_set, sampler=sampler, transform=transform
        )
        self._video_ids = self._get_video_ids(self.gulp_dir, filter)
        self.labels = self._label_examples(self._video_ids, self.label_set)

    @property
    def video_ids(self):
        return self._video_ids

    def __len__(self):
        return len(self._video_ids)

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        id_ = self._video_ids[index]
        frame_count = self._get_frame_count(id_)
        frame_idx = self.sampler.sample(frame_count)
        if isinstance(frame_idx, slice):
            frames = self._load_frames(id_, frame_idx)
        elif isinstance(frame_idx, list):
            if isinstance(frame_idx[0], slice):
                frame_idx = cast(List[slice], frame_idx)
                frames = np.concatenate(
                    [self._load_frames(id_, slice_) for slice_ in frame_idx]
                )
            elif isinstance(frame_idx[0], numbers.Number):
                frames = np.concatenate(
                    [
                        self._load_frames(id_, slice(index, index + 1))
                        for index in frame_idx
                    ]
                )
            else:
                raise TypeError(
                    "frame_idx was a list of {} but we only support "
                    "int and slice elements".format(type(frame_idx[0]).__name__)
                )
        else:
            raise TypeError(
                "frame_idx was of type {} but we only support slice, "
                "List[slice], List[int]".format(type(frame_idx).__name__)
            )

        if self.labels is not None:
            label = self.labels[index]
        else:
            label = empty_label

        frames, label = invoke_transform(self.transform, frames, label)

        if label is not empty_label:
            return frames, label
        return frames

    @staticmethod
    def _label_examples(video_ids: List[str], label_set: Optional[LabelSet]):
        if label_set is None:
            return None
        else:
            return [label_set[video_id] for video_id in video_ids]

    @staticmethod
    def _get_video_ids(
        gulp_dir, filter_fn: Optional[Callable[[str], bool]]
    ) -> List[str]:
        return sorted(
            [
                id_
                for id_ in gulp_dir.merged_meta_dict.keys()
                if filter_fn is None or filter_fn(id_)
            ]
        )

    @staticmethod
    def _get_label_set(
        gulp_dir, label_field: Optional[str], label_set: Optional[LabelSet]
    ):
        if label_field is None:
            label_field = "label"
        if label_set is None:
            label_set = GulpLabelSet(gulp_dir.merged_meta_dict, label_field=label_field)
        return label_set

    def _load_frames(self, id_: str, frame_idx: slice) -> np.ndarray:
        frames, _ = self.gulp_dir[id_, frame_idx]
        return np.array(frames, dtype=np.uint8)

    def _get_frame_count(self, id_: str):
        info = self.gulp_dir.merged_meta_dict[id_]
        return len(info["frame_info"])
