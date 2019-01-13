from hypothesis import given
import pandas as pd

from torchvideo.datasets import DummyLabelSet, GulpLabelSet, CsvLabelSet

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


class TestCsvLabelSet:
    def test_returns_label_field_for_dataframe(self):
        df = pd.DataFrame({"name": ["video1", "video2"], "label": [1, 2]}).set_index(
            "name"
        )

        label_set = CsvLabelSet(df, col="label")

        assert label_set["video1"] == 1

    def test_returns_element_from_series(self):
        series = pd.Series([1, 2], index=["video1", "video2"])

        label_set = CsvLabelSet(series)

        assert label_set["video2"] == 2

        assert label_set["video1"] == 1
