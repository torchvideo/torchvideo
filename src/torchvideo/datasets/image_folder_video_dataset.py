import torch
from pathlib import Path
from typing import Union, Optional, Callable, Tuple, List, Iterator

import PIL.Image
from PIL.Image import Image

from torchvideo.samplers import FrameSampler, frame_idx_to_list, _default_sampler
from torchvideo.transforms import PILVideoToTensor
from .video_dataset import VideoDataset
from .types import Label, empty_label, PILVideoTransform
from .helpers import invoke_transform
from .label_sets import LabelSet


class ImageFolderVideoDataset(VideoDataset):
    """Dataset stored as a folder containing folders of images, where each folder
    represents a video.

    The folder hierarchy should look something like this: ::

        root/video1/frame_000001.jpg
        root/video1/frame_000002.jpg
        root/video1/frame_000003.jpg
        ...

        root/video2/frame_000001.jpg
        root/video2/frame_000002.jpg
        root/video2/frame_000003.jpg
        root/video2/frame_000004.jpg
        ...

    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filename_template: str,
        filter: Optional[Callable[[Path], bool]] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ):
        """

        Args:
            root_path: Path to dataset on disk. Contents of this folder should be
                example folders, each with frames named according to the
                ``filename_template`` argument.
            filename_template: Python 3 style formatting string describing frame
                filenames: e.g. ``"frame_{:06d}.jpg"`` for the example dataset in the
                class docstring.
            filter: Optional filter callable that decides whether a given example folder
                is to be included in the dataset or not.
            label_set: Optional label set for labelling examples.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform performed over the loaded clip.
            frame_counter: Optional callable used to determine the number of frames
                each video contains. The callable will be passed the path to a video
                folder and should return a positive integer representing the number of
                frames. This tends to be useful if you've precomputed the number of
                frames in a dataset.
        """
        super().__init__(root_path, label_set, sampler=sampler, transform=transform)
        self._video_dirs = sorted(
            [d for d in self.root_path.iterdir() if filter is None or filter(d)]
        )
        self.labels = self._label_examples(self._video_dirs, label_set)
        self.video_lengths = self._measure_video_lengths(
            self._video_dirs, frame_counter
        )
        self.filename_template = filename_template
        if self.transform is None:
            self.transform = PILVideoToTensor()

    @property
    def video_ids(self):
        return self._video_dirs

    def __len__(self) -> int:
        return len(self._video_dirs)

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_folder = self._video_dirs[index]
        video_length = self.video_lengths[index]
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_folder)
        if self.labels is not None:
            label = self.labels[index]
        else:
            label = empty_label

        frames_tensor, label = invoke_transform(self.transform, frames, label)

        if label == empty_label:
            return frames_tensor
        return frames_tensor, label

    @staticmethod
    def _measure_video_lengths(
        video_dirs, frame_counter: Optional[Callable[[Path], int]]
    ):
        if frame_counter is None:
            return [len(list(video_dir.iterdir())) for video_dir in video_dirs]
        else:
            return [frame_counter(video_dir) for video_dir in video_dirs]

    @staticmethod
    def _label_examples(video_dirs, label_set: Optional[LabelSet]):
        if label_set is not None:
            return [label_set[video_dir.name] for video_dir in video_dirs]
        else:
            return None

    def _load_frames(
        self, frames_idx: Union[slice, List[slice], List[int]], video_folder: Path
    ) -> Iterator[Image]:
        frame_numbers = frame_idx_to_list(frames_idx)
        filepaths = [
            video_folder / self.filename_template.format(index + 1)
            for index in frame_numbers
        ]
        frames = (self._load_image(path) for path in filepaths)
        # shape: (n_frames, height, width, channels)
        return frames

    def _load_image(self, path: Path) -> Image:
        if not path.exists():
            raise ValueError("Image path {} does not exist".format(path))
        return PIL.Image.open(str(path))
