from abc import ABC

from ..types import Label


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
