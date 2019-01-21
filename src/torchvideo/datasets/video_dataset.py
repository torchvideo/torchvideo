from pathlib import Path
from typing import Union, Optional, Tuple, List, Any  # noqa

import torch.utils.data

from torchvideo.samplers import FrameSampler, _default_sampler
from .label_sets import LabelSet
from .types import Label, Transform


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
