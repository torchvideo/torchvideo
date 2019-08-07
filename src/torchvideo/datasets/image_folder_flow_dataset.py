from pathlib import Path
from typing import Union, Optional, Tuple, Iterator, TypeVar

import PIL.Image
import torch

from torchvideo.datasets.helpers import invoke_transform
from .video_dataset import VideoDataset
from .label_sets import LabelSet
from .types import PILVideoTransform, Label, empty_label
from ..samplers import FrameSampler, _default_sampler, frame_idx_to_list


T = TypeVar("T")


def interleave(*iterables: Iterator[T]) -> Iterator[T]:
    while True:
        try:
            values = [next(it) for it in iterables]
        except StopIteration:
            return
        for val in values:
            yield val


class ImageFolderFlowDataset(VideoDataset):
    """Dataset stored as a folder containing folders of examples.
    """

    def __init__(
        self,
        root_path: Union[Path, str],
        filename_template: str,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
    ):
        super().__init__(
            root_path=root_path,
            label_set=label_set,
            sampler=sampler,
            transform=transform,
        )
        self.filename_template = filename_template
        self._video_dirs = sorted([d for d in self.root_path.iterdir()])
        self._video_lengths = list(map(self._measure_video_length, self._video_dirs))
        if label_set is not None:
            self.labels = [label_set[video_dir.name] for video_dir in self._video_dirs]

    @property
    def video_ids(self):
        return self._video_dirs

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_dir = self._video_dirs[index]
        frame_idx = frame_idx_to_list(self.sampler.sample(self._video_lengths[index]))
        frames_iter = self._load_frames(video_dir, frame_idx)
        if self.labels is not None:
            label = self.labels[index]
        else:
            label = empty_label
        frames_tensor, label = invoke_transform(self.transform, frames_iter, label)
        return frames_tensor, label

    def _load_frames(self, video_dir, frame_idx) -> Iterator[PIL.Image.Image]:
        frame_paths = dict()
        for axis in ["u", "v"]:
            frame_paths[axis] = [
                video_dir / self.filename_template.format(axis=axis, index=(index + 1))
                for index in frame_idx
            ]
        frames = dict()
        for axis in ["u", "v"]:
            frames[axis] = (self._load_image(path) for path in frame_paths[axis])
        frames_iter = interleave(frames["u"], frames["v"])
        return frames_iter

    def _measure_video_length(self, video_dir: Path) -> int:
        return len(list(video_dir.iterdir()))

    def _load_image(self, path: Path) -> PIL.Image.Image:
        return PIL.Image.open(str(path))

    def __len__(self) -> int:
        return len(self._video_dirs)
