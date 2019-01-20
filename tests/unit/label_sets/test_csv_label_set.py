import pandas as pd

from torchvideo.datasets import CsvLabelSet


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
