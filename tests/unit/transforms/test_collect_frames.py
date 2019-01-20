from hypothesis import given

from torchvideo.transforms import CollectFrames
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestCollectFrames:
    def test_repr(self):
        assert repr(CollectFrames()) == "CollectFrames()"

    @given(pil_video(max_length=10, max_height=1, max_width=1))
    def test_collect_frames_make_list_from_iterator(self, video):
        transform = CollectFrames()
        assert transform(iter(video)) == video

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=1, min_height=1).example()
        transform = CollectFrames()

        assert_preserves_label(transform, iter(video))
