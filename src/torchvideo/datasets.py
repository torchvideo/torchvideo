from abc import ABC
from pathlib import Path
from typing import Union, Tuple, List, Callable, Any, Iterator, Optional, Dict, cast
import numbers

import PIL.Image
from PIL.Image import Image
import numpy as np
import torch.utils.data

from torchvideo.internal.readers import _get_videofile_frame_count, _is_video_file
from torchvideo.samplers import FrameSampler, FullVideoSampler, frame_idx_to_list
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
    """Abstract base class that all ``VideoDatasets`` inherit from. If you are
    implementing your own ``VideoDataset``, you should inherit from this class.
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
            root_path: Path to dataset on disk.
            label_set: Optional label set for labelling examples.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform over the list of frames.
        """
        self.root_path = Path(root_path)
        self.label_set = label_set
        self.sampler = sampler
        self.transform = transform
        self.labels = None  # type: Optional[List[Any]]
        """The labels corresponding to the examples in the dataset. To get the label
        for example at index ``i`` you simple call ``dataset.labels[i]``, although
        this will be returned by ``__getitem__`` if this field is not None."""
        """The unique ID of each video (usually a path is possible)"""

    @property
    def video_ids(self):
        raise NotImplementedError()

    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:  # pragma: no cover
        """Load an example by index

        Args:
            index: index of the example within the dataset.

        Returns:
            Example transformed by ``transform`` if one was passed during
            instantiation, otherwise the example is converted to a tensor without any
            transformations applied to it. Additionally, if a label set is present, the
            method return a tuple: ``(video_tensor, label)``
        """
        raise NotImplementedError()

    def __len__(self) -> int:  # pragma: no cover
        """Total number of examples in the dataset"""
        raise NotImplementedError()


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
        if self.transform is None:
            frames_tensor = PILVideoToTensor()(frames)
        else:
            frames_tensor = self.transform(frames)

        if self.labels is not None:
            label = self.labels[index]
            return frames_tensor, label
        else:
            return frames_tensor

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


class VideoFolderDataset(VideoDataset):
    """Dataset stored as a folder of videos, where each video is a single example
    in the dataset.

    The folder hierarchy should look something like this: ::

        root/video1.mp4
        root/video2.mp4
        ...
    """

    def __init__(
        self,
        root_path: Union[str, Path],
        filter: Optional[Callable[[Path], bool]] = None,
        label_set: Optional[LabelSet] = None,
        sampler: FrameSampler = _default_sampler(),
        transform: Optional[PILVideoTransform] = None,
        frame_counter: Optional[Callable[[Path], int]] = None,
    ) -> None:
        """
        Args:
            root_path: Path to dataset folder on disk. The contents of this folder
                should be video files.
            filter: Optional filter callable that decides whether a given example video
                is to be included in the dataset or not.
            label_set: Optional label set for labelling examples.
            sampler: Optional sampler for drawing frames from each video.
            transform: Optional transform over the list of frames.
            frame_counter: Optional callable used to determine the number of frames
                each video contains. The callable will be passed the path to a video and
                should return a positive integer representing the number of frames.
                This tends to be useful if you've precomputed the number of frames in a
                dataset.
        """
        if transform is None:
            transform = PILVideoToTensor()
        super().__init__(
            root_path, label_set=label_set, sampler=sampler, transform=transform
        )
        self._video_paths = self._get_video_paths(self.root_path, filter)
        self.labels = self._label_examples(self._video_paths, label_set)
        self.video_lengths = self._measure_video_lengths(
            self._video_paths, frame_counter
        )

    @property
    def video_ids(self):
        return self._video_paths

    # TODO: This is very similar to ImageFolderVideoDataset consider merging into
    #  VideoDataset
    def __getitem__(
        self, index: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Label]]:
        video_file = self._video_paths[index]
        video_length = self.video_lengths[index]
        frames_idx = self.sampler.sample(video_length)
        frames = self._load_frames(frames_idx, video_file)
        if self.transform is not None:
            frames = self.transform(frames)

        if self.labels is not None:
            return frames, self.labels[index]
        else:
            return frames

    def __len__(self):
        return len(self._video_paths)

    @staticmethod
    def _measure_video_lengths(video_paths, frame_counter):
        if frame_counter is None:
            frame_counter = _get_videofile_frame_count
        return [frame_counter(vid_path) for vid_path in video_paths]

    @staticmethod
    def _label_examples(video_paths, label_set: Optional[LabelSet]):
        if label_set is None:
            return None
        else:
            return [label_set[video_path.name] for video_path in video_paths]

    @staticmethod
    def _get_video_paths(root_path, filter):
        return sorted(
            [
                child
                for child in root_path.iterdir()
                if _is_video_file(child) and (filter is None or filter(child))
            ]
        )

    @staticmethod
    def _load_frames(
        frame_idx: Union[slice, List[slice], List[int]], video_file: Path
    ) -> Iterator[Image]:
        from torchvideo.internal.readers import default_loader

        return default_loader(video_file, frame_idx)


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
        """
        from gulpio import GulpDirectory

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

        if self.transform is not None:
            frames = self.transform(frames)
        else:
            frames = torch.Tensor(np.rollaxis(frames, -1, 0)).div_(255)

        if self.labels is not None:
            label = self.labels[index]
            return frames, label
        else:
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


class LambdaLabelSet(LabelSet):
    """A label set that wraps a function used to retrieve a label for an example"""

    def __init__(self, labeller_fn: Callable[[str], Label]):
        """
        Args:
            labeller_fn: Function for labelling examples.
        """
        self._labeller_fn = labeller_fn

    def __getitem__(self, video_name: str) -> Label:
        return self._labeller_fn(video_name)


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


class CsvLabelSet(LabelSet):
    """LabelSet for a pandas DataFrame or Series. The index of the DataFrame/Series
    is assumed to be the set of video names and the values in a series the label. For a
    dataframe the ``field`` kwarg specifies which field to use as the label

    Examples:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'video': ['video1', 'video2'],
        ...                    'label': [1, 2]}).set_index('video')
        >>> label_set = CsvLabelSet(df, col='label')
        >>> label_set['video1']
        1

    """

    def __init__(self, df, col: Optional[str] = "label"):
        """

        Args:
            df: pandas DataFrame or Series containing video names/ids and their
                corresponding labels.
            col: The column to read the label from when df is a DataFrame.
        """
        self.df = df
        self._field = col

    def __getitem__(self, video_name: str) -> Label:
        try:
            return self.df[self._field].loc[video_name]
        except KeyError:
            return self.df.loc[video_name]
