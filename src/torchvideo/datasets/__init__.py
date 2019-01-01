from abc import ABC
from collections import namedtuple
from pathlib import Path
from typing import Union, Tuple, List, Callable, Any, Iterator, Optional, Dict

import PIL.Image
from PIL.Image import Image
import numpy as np
import torch.utils.data

from torchvideo.internal.readers import _get_videofile_frame_count, _is_video_file
from torchvideo.internal.utils import frame_idx_to_list
from torchvideo.samplers import FrameSampler, FullVideoSampler
from torchvideo.transforms import PILVideoToTensor


Label = Any

Transform = Callable[[Any], torch.Tensor]
PILVideoTransform = Callable[[Iterator[Image]], torch.Tensor]
NDArrayVideoTransform = Callable[[np.ndarray], torch.Tensor]


_default_sampler = FullVideoSampler


class LabelSet(ABC):  # pragma: no cover
    """Abstract base class that all ``LabelSets`` inherit from

    If you are implementing your own ``LabelSet``, you should inherit from this
    class."""

    def __getitem__(self, video_name: str) -> Label:
        """
        Args:
            video_name: The filename or id of the video

        Returns:
            The corresponding label
        """
        raise NotImplementedError()


class VideoDataset(torch.utils.data.Dataset):
    """Abstract base class that all ``VideoDatasets`` inherit from

    If you are implementing your own ``VideoDataset``, you should inherit from this
    class.
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[Transform] = None,
    ) -> None:
        """

        Args:
            root_path: Path to dataset on disk
            label_set: Optional label set for labelling examples
            sampler: Optional sampler for drawing frames from each video
            transform: Optional transform over the list of frames
        """
        self.root_path = Path(root_path)
        self.label_set = label_set
        self.sampler = sampler
        self.transform = transform

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:  # pragma: no cover
        """Load an example by index

        Args:
            index: index of the example within the dataset

        Returns:
            Example transformed by transform if set, otherwise the example is
            converted to a tensor without an transformations applied to it.

            If a label set is present, the method return a tuple:
            ``(video_tensor, label)``
        """
        raise NotImplementedError()

    def __len__(self) -> int:  # pragma: no cover
        """Total number of examples in the dataset"""
        raise NotImplementedError()


class ImageFolderVideoDataset(VideoDataset):
    """VideoDataset from a folder containing folders of images, each folder represents
    a video

    The expected folder hierarchy is like the below: ::

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
    ):
        """

        Args:
            root_path: Path to dataset on disk. Contents of this folder should be
                example folders, each with frames named according to the
                ``filename_template`` argument.
            filename_template: Python 3 style formatting string describing frame
                filenames: e.g. ``"frame_{:05d}.jpg"``
            label_set: Optional label set for labelling examples
            sampler: Optional sampler for drawing frames from each video
            transform: Optional transform over the list of frames
        """
        super().__init__(root_path, label_set, sampler=sampler, transform=transform)
        self.video_dirs = sorted(
            [d for d in self.root_path.iterdir() if filter is None or filter(d)]
        )
        self.video_lengths = [
            len(list(video_dir.iterdir())) for video_dir in self.video_dirs
        ]
        self.filename_template = filename_template

    def __len__(self) -> int:
        return len(self.video_dirs)

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_folder = self.video_dirs[index]
        video_length = self.video_lengths[index]
        video_name = video_folder.name
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_folder)
        if self.transform is None:
            frames_tensor = PILVideoToTensor()(frames)
        else:
            frames_tensor = self.transform(frames)

        if self.label_set is not None:
            label = self.label_set[video_name]
            return frames_tensor, label
        else:
            return frames_tensor

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


class VideoFolderDataset(VideoDataset):
    """VideoDataset built from a folder of videos, each forming a single example in the
    dataset.

    The expected folder hierarchy is like the below: ::

        root/video1.mp4
        root/video2.mp4
        ...


    We need to know the duration of the video files, to do this, we expect
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filter: Optional[Callable[[Path], bool]] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
    ) -> None:
        """
        Args:
            root_path: Path to dataset folder on disk. The contents of this folder
                should be video files.
            label_set: Optional label set for labelling examples
            sampler: Optional sampler for drawing frames from each video
            transform: Optional transform over the list of frames
        """
        super().__init__(
            root_path, label_set=label_set, sampler=sampler, transform=transform
        )
        self.video_paths = sorted(
            [
                child
                for child in self.root_path.iterdir()
                if _is_video_file(child) and (filter is None or filter(child))
            ]
        )
        self.video_lengths = [
            _get_videofile_frame_count(vid_path) for vid_path in self.video_paths
        ]

    # TODO: This is very similar to ImageFolderVideoDataset consider merging into
    #  VideoDataset
    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
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
    ) -> Iterator[Image]:
        from torchvideo.internal.readers import default_loader

        return default_loader(video_file, frame_idx)


class GulpVideoDataset(VideoDataset):
    """GulpIO Video dataset.

    The expected folder hierarchy is like the below: ::

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
        label_field: Optional[str] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[NDArrayVideoTransform] = None,
    ):
        """
        Args:
            root_path: Path to GulpIO dataset folder on disk. The ``.gulp`` and
                ``.gmeta`` files are direct children of this directory.
            label_field: Meta data field name that stores the label of an example,
                this is used to construct a :class:`GulpLabelSet` that performs the
                example labelling. Defaults to ``'label'``.
            label_set: Optional label set for labelling examples. This is mutually
                exclusive with ``label_field``.
            sampler: Optional sampler for drawing frames from each video
            transform: Optional transform over the :class:`ndarray` with layout
                ``THWC``. Note you'll probably want to remap the channels to ``CTHW`` at
                the end of this transform.
        """
        from gulpio import GulpDirectory

        self.gulp_dir = GulpDirectory(str(root_path))
        if label_field is None:
            label_field = "label"
        if label_set is None:
            label_set = GulpLabelSet(
                self.gulp_dir.merged_meta_dict, label_field=label_field
            )
        super().__init__(
            root_path, label_set=label_set, sampler=sampler, transform=transform
        )
        self._video_ids = sorted(list(self.gulp_dir.merged_meta_dict.keys()))

    def __len__(self):
        return len(self._video_ids)

    def __getitem__(self, index) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        id_ = self._video_ids[index]
        frame_count = self._get_frame_count(id_)
        frame_idx = self.sampler.sample(frame_count)
        label = self.label_set[id_]
        if isinstance(frame_idx, slice):
            frames = self._load_frames(id_, frame_idx)
        elif isinstance(frame_idx, list):
            if isinstance(frame_idx[0], slice):
                frames = np.concatenate(
                    [self._load_frames(id_, slice_) for slice_ in frame_idx]
                )
            elif isinstance(frame_idx[0], int):
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

        if self.transform is not None:
            return self.transform(frames), label
        return torch.Tensor(np.rollaxis(frames, -1, 0)), label

    def _load_frames(self, id_: str, frame_idx: slice) -> np.ndarray:
        frames, _ = self.gulp_dir[id_, frame_idx]
        return np.array(frames) / 255

    def _get_frame_count(self, id_):
        info = self.gulp_dir.merged_meta_dict[id_]
        return len(info["frame_info"])


class DummyLabelSet(LabelSet):
    """A dummy label set that returns the same label regardless of video"""

    def __init__(self, label: Label = 0):
        """
        Args:
            label: The label given to any video
        """
        self.label = label

    def __getitem__(self, video_name) -> Label:
        return self.label

    def __repr__(self):
        return self.__class__.__name__ + "(label={!r})".format(self.label)


class GulpLabelSet(LabelSet):
    """LabelSet for GulpIO datasets where the label is contained within the metadata of
    the gulp directory. Assuming you've written the label of each video to a field
    called ``'label'`` in the metadata you can create a LabelSet like:
    ``GulpLabelSet(gulp_dir.merged_meta_dict, label_field='label')``
    """

    def __init__(self, merged_meta_dict: Dict[str, Any], label_field: str = "label"):
        self.merged_meta_dict = merged_meta_dict
        self.label_field = label_field

    def __getitem__(self, video_name: str) -> Label:
        # The merged meta dict has the form: { video_id: { meta_data: [{ meta... }] }}
        video_meta_data = self.merged_meta_dict[video_name]["meta_data"][0]
        return video_meta_data[self.label_field]
