from .label_set import LabelSet, Label


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
