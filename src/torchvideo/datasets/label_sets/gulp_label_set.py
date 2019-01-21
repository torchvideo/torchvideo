from typing import Dict, Any

from .label_set import LabelSet, Label


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
