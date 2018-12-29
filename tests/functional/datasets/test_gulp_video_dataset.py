import pytest

from tests import TEST_DATA_ROOT
from torchvideo.datasets import GulpVideoDataset


@pytest.fixture()
def gulp_dir():
    return TEST_DATA_ROOT / "media" / "gulp_output"


def test_gulp_video_dataset(gulp_dir):
    dataset = GulpVideoDataset(gulp_dir)
    frames, label = dataset[0]
    assert dataset._video_ids[0] == "video0"

    assert int(label) == 0
    assert frames.size() == (3, 50, 368, 640)
