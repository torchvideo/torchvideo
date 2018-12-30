from hypothesis import given

from torchvideo.datasets import DummyLabelSet, GulpLabelSet

import hypothesis.strategies as st


class TestDummyLabelSet:
    @given(st.text())
    def test_return_label_for_any_video_name(self, video_name):
        label = 0
        label_set = DummyLabelSet(label)

        assert label_set[video_name] == label

    def test_repr(self):
        assert repr(DummyLabelSet(1)) == "DummyLabelSet(label=1)"


class TestGulpLabelSet:
    meta = {
        "1": {
            "frame_info": [[0, 3, 7260], [7260, 3, 7252]],
            "meta_data": [{"label": "something", "2nd_label": "blah"}],
        }
    }

    def test_defaults_to_label_field(self):
        label_set = GulpLabelSet(self.meta)

        assert label_set["1"] == "something"

    def test_custom_label_field(self):
        label_set = GulpLabelSet(self.meta, label_field="2nd_label")

        assert label_set["1"] == "blah"
