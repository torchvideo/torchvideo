from pathlib import Path

import pytest

from tests import TEST_DATA_ROOT
from torchvideo.datasets import ImageFolderVideoDataset
from torchvideo.samplers import LambdaSampler


@pytest.fixture(scope="module")
def image_folder() -> Path:
    return TEST_DATA_ROOT / "media" / "video_image_folder"


@pytest.fixture(scope="module")
def image_folder_video_dataset(image_folder):
    return ImageFolderVideoDataset(image_folder, filename_template="frame_{:05d}.jpg")


class TestImageFolderVideoDataset:
    video_count = 11
    video1_frame_count = 50
    video1_size = (368, 640)

    def test_length(self, image_folder_video_dataset):
        dataset = image_folder_video_dataset
        assert len(dataset) == self.video_count

    def test_video_range(self, image_folder_video_dataset):
        frames = image_folder_video_dataset[0]

        assert frames.min() >= 0
        assert frames.max() <= 1

    def test_loads_all_frames_by_default(self, image_folder_video_dataset):
        dataset = image_folder_video_dataset

        frames = dataset[1]
        assert frames.shape == (3, self.video1_frame_count, *self.video1_size)

    def test_loading_by_slice(self, image_folder_video_dataset):
        dataset = image_folder_video_dataset
        length = 4
        dataset.sampler = LambdaSampler(lambda video_length: slice(0, length, 1))

        frames = dataset[1]
        shape = frames.shape
        assert shape == (3, length, *self.video1_size)

    def test_loading_by_list_of_slice(self, image_folder_video_dataset):
        dataset = image_folder_video_dataset
        lengths = 2, 3, 4
        dataset.sampler = LambdaSampler(
            lambda video_length: [slice(0, l, 1) for l in lengths]
        )

        frames = dataset[1]
        shape = frames.shape
        assert shape == (3, sum(lengths), *self.video1_size)

    def test_loading_by_list_of_ints(self, image_folder_video_dataset):
        dataset = image_folder_video_dataset
        frame_idx = [0, 1, 3, 4]
        dataset.sampler = LambdaSampler(lambda video_length: frame_idx)

        frames = dataset[1]
        shape = frames.shape
        assert shape == (3, len(frame_idx), *self.video1_size)
