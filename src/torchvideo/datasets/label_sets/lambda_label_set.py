from typing import Callable

from .label_set import LabelSet, Label


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
