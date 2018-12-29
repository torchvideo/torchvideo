import pytest

from torchvideo.samplers import LambdaSampler
from ... import TEST_DATA_ROOT
from torchvideo.datasets import VideoFolderDataset


@pytest.fixture(scope="module")
def video_folder_dir():
    return TEST_DATA_ROOT / "media" / "video_folder"


@pytest.fixture(scope="module")
def video_folder_dataset(video_folder_dir):
    return VideoFolderDataset(video_folder_dir)


class TestVideoFolderDataset:
    video_count = 11
    video_length = 50
    size = (368, 640)

    def test_dataset_length(self, video_folder_dataset):
        assert len(video_folder_dataset) == self.video_count

    def test_loads_all_frames_by_default(self, video_folder_dataset):
        dataset = video_folder_dataset

        frames = dataset[1]

        shape = frames.shape
        assert shape == (3, self.video_length, *self.size)

    def test_loading_by_slice(self, video_folder_dataset):
        dataset = video_folder_dataset
        length = 4
        dataset.sampler = LambdaSampler(lambda video_length: slice(0, length, 1))

        frames = dataset[1]

        frames_shape = frames.shape
        assert frames_shape == (3, length, *self.size)

    def test_loading_by_list_of_slices(self, video_folder_dataset):
        dataset = video_folder_dataset
        slices = [(0, 2), (3, 5), (8, 10)]
        lengths = [(s[1] - s[0]) for s in slices]
        video_folder_dataset.sampler = LambdaSampler(
            lambda video_length: [slice(*s, 1) for s in slices]
        )

        frames = dataset[1]

        frames_shape = frames.size()
        assert frames_shape == (3, sum(lengths), *self.size)

    def test_loading_by_list_of_int(self, video_folder_dataset):
        dataset = video_folder_dataset
        frames_idx = [0, 1, 4, 7]

        video_folder_dataset.sampler = LambdaSampler(lambda video_length: frames_idx)

        frames = dataset[1]

        frames_shape = frames.size()
        assert frames_shape == (3, len(frames_idx), *self.size)
