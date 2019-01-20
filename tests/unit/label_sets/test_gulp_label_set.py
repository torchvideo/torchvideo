from torchvideo.datasets import GulpLabelSet


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
