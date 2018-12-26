from abc import ABC
from pathlib import Path
from typing import Union, Tuple, List

import PIL.Image
import numpy as np
import torch.utils.data

from torchvideo.internal.utils import frame_idx_to_list
from torchvideo.datasets.label_set import LabelSet, Label
from torchvideo.samplers import FrameSampler, FullVideoSampler

_default_sampler = FullVideoSampler
_VIDEO_FILE_EXTENSIONS = {
    "mp4",
    "webm",
    "avi",
    "3gp",
    "wmv",
    "mpg",
    "mpeg",
    "mov",
    "mkv",
}


def _is_video_file(path: Path) -> bool:
    extension = path.name.lower().split(".")[-1]
    return extension in _VIDEO_FILE_EXTENSIONS


class VideoDataset(torch.utils.data.Dataset, ABC):
    def __init__(
        self,
        root_path: Union[str, Path],
        label_set: LabelSet = None,
        sampler: FrameSampler = _default_sampler(),
    ) -> None:
        self.root_path = Path(root_path)
        self.label_set = label_set
        self.sampler = sampler

    def __getitem__(
        self, index
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:  # pragma: no cover
        raise NotImplementedError()

    def __len__(self):  # pragma: no cover
        raise NotImplementedError()


class ImageFolderVideoDataset(VideoDataset):
    """VideoDataset from a folder containing folders of images, each folder represents
    a video

    The expected folder hierarchy is like the below:

        root/video1/frame_000001.jpg
        root/video1/frame_000002.jpg
        root/video1/frame_000003.jpg

        root/video2/frame_000001.jpg
        root/video2/frame_000002.jpg
        root/video2/frame_000003.jpg
        root/video2/frame_000004.jpg
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filename_template: str,
        label_set: LabelSet = None,
        sampler: FrameSampler = _default_sampler(),
    ):
        super().__init__(root_path, label_set, sampler=sampler)
        self.video_dirs = sorted(list(root_path.iterdir()))
        self.video_lengths = [
            len(list(video_dir.iterdir())) for video_dir in self.video_dirs
        ]
        self.filename_template = filename_template

    def __len__(self):
        return len(self.video_dirs)

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_folder = self.video_dirs[index]
        video_length = self.video_lengths[index]
        video_name = video_folder.name
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_folder)

        frames_tensor = torch.Tensor(frames)
        if self.label_set is not None:
            label = self.label_set[video_name]
            return frames_tensor, label
        else:
            return frames_tensor

    def _load_frames(
        self, frames_idx: Union[slice, List[slice], List[int]], video_folder: Path
    ) -> np.ndarray:
        frame_numbers = frame_idx_to_list(frames_idx)
        filepaths = [
            video_folder / self.filename_template.format(index + 1)
            for index in frame_numbers
        ]
        frames = [self._load_image(path) for path in filepaths]
        # shape: (n_frames, height, width, channels)
        return np.array(frames)

    def _load_image(self, path: Path) -> np.ndarray:
        if not path.exists():
            raise ValueError("Image path {} does not exist".format(path))
        return np.array(PIL.Image.open(str(path)))


class VideoFolderDataset(VideoDataset):
    """VideoDataset built from a folder of videos, each forming a single example in the
    dataset.

    We need to know the duration of the video files, to do this, we expect
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        label_set: LabelSet = None,
        sampler: FrameSampler = _default_sampler(),
    ) -> None:
        super(VideoFolderDataset, self).__init__(
            root_path, label_set=label_set, sampler=sampler
        )
        self.video_paths = sorted(
            [child for child in root_path.iterdir() if _is_video_file(child)]
        )
        self.video_lengths = []

    # TODO: This is very similar to ImageFolderVideoDataset consider merging into
    #  VideoDataset
    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_file = self.video_paths[index]
        video_name = video_file.stem
        video_length = self.video_lengths[index]
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_file)
        if self.label_set is not None:
            label = self.label_set[video_name]
            return frames, label
        else:
            return frames

    def __len__(self):
        return len(self.video_paths)

    def _load_frames(
        self, frame_idx: Union[slice, List[slice], List[int]], video_file: Path
    ) -> torch.Tensor:
        from torchvideo.internal.readers import default_loader

        return default_loader(video_file, frame_idx)


class GulpVideoDataset(VideoDataset):
    def __init__(
        self,
        root_path: Union[str, Path],
        label_set: LabelSet = None,
        sampler: FrameSampler = _default_sampler(),
    ):
        super(GulpVideoDataset, self).__init__(
            root_path, label_set=label_set, sampler=sampler
        )
        from gulpio import GulpDirectory

        self.gulp_dir = GulpDirectory(str(self.root_path))
        self._video_ids = sorted(list(self.gulp_dir.merged_meta_dict.keys()))

    def __len__(self):
        return len(self._video_ids)

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        id_ = self._video_ids[index]
        frame_count = self._get_frame_count(id_)
        frame_idx = self.sampler.sample(frame_count)
        meta = self.gulp_dir.merged_meta_dict[id_]["meta_data"][0]
        if isinstance(frame_idx, slice):
            frames, _ = self.gulp_dir[id_, frame_idx]
            return torch.Tensor(frames), meta["label"]
        elif isinstance(frame_idx, list):
            if isinstance(frame_idx[0], slice):
                return torch.Tensor(
                    np.array([self.gulp_dir[id_, slice_][0] for slice_ in frame_idx])
                )
            elif isinstance(frame_idx[0], int):
                return torch.Tensor(
                    np.array(
                        [
                            self.gulp_dir[id_, slice(index, index + 1)][0]
                            for index in frame_idx
                        ]
                    )
                )

    def _get_frame_count(self, id_):
        info = self.gulp_dir.merged_meta_dict[id_]
        return len(info["frame_info"])
