from pathlib import Path

import pytest

from tests import TEST_DATA_ROOT
from torchvideo.datasets import ImageFolderVideoDataset, LabelSet, Label


class NullLabelSet(LabelSet):
    def __getitem__(self, video_name) -> Label:
        return 0


@pytest.fixture
def image_folder() -> Path:
    return TEST_DATA_ROOT / 'media' / 'video_image_folder'


def test_image_folder_video_dataset_loads_all_frames_by_default(image_folder):
    video_count = 11
    video1_frame_count = 50
    video1_size = (368, 640)

    dataset = ImageFolderVideoDataset(image_folder, filename_template='frame_{:05d}.jpg')

    assert len(dataset) == video_count

    frames = dataset[1]
    assert len(frames) == video1_frame_count
    assert frames.shape == (video1_frame_count, *video1_size, 3)
