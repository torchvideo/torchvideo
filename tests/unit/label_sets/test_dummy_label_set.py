from hypothesis import given, strategies as st

from torchvideo.datasets import DummyLabelSet


class TestDummyLabelSet:
    @given(st.text())
    def test_return_label_for_any_video_name(self, video_name):
        label = 0
        label_set = DummyLabelSet(label)

        assert label_set[video_name] == label

    def test_repr(self):
        assert repr(DummyLabelSet(1)) == "DummyLabelSet(label=1)"
