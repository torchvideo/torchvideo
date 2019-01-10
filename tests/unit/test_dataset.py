import numpy
import os
from pathlib import Path
from unittest.mock import Mock

import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

import torchvideo
from torchvideo.datasets import (
    ImageFolderVideoDataset,
    VideoFolderDataset,
    DummyLabelSet,
    LambdaLabelSet,
)
from torchvideo.internal.utils import frame_idx_to_list


@pytest.fixture
def mock_frame_count(monkeypatch):
    def get_videofile_frame_count(path):
        return 10

    monkeypatch.setattr(
        torchvideo.datasets, "_get_videofile_frame_count", get_videofile_frame_count
    )


@pytest.fixture()
def dataset_dir(fs: FakeFilesystem):
    path = "/tmp/dataset"
    fs.create_dir(path)
    return path


class TestImageFolderVideoDatasetUnit:
    def test_all_videos_folders_are_present_in_video_dirs_by_default(self, dataset_dir):
        video_count = 10
        self.make_video_dirs(dataset_dir, video_count)

        dataset = ImageFolderVideoDataset(dataset_dir, "frame_{:05d}.jpg")

        assert len(dataset.video_dirs) == video_count

    def test_filtering_video_folders(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 10)

        def filter(video_path: Path):
            return video_path.name.endswith(("1", "2", "3"))

        dataset = ImageFolderVideoDataset(
            dataset_dir, "frame_{:05d}.jpg", filter=filter
        )

        assert len(dataset.video_dirs) == 3
        assert dataset.video_dirs[0].name == "video1"
        assert dataset.video_dirs[1].name == "video2"
        assert dataset.video_dirs[2].name == "video3"

    def test_labels_are_accessible(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 10)

        dataset = ImageFolderVideoDataset(
            dataset_dir,
            "frame_{:05d}.jpg",
            label_set=LambdaLabelSet(lambda p: int(p[-1])),
        )

        assert 10 == len(dataset.labels)
        assert all([label == i for i, label in enumerate(dataset.labels)])

    @staticmethod
    def make_video_dirs(dataset_dir, video_count):
        for i in range(0, video_count):
            os.makedirs(os.path.join(dataset_dir, "video{}".format(i)))


class TestVideoFolderDatasetUnit:
    def test_all_videos_are_present_in_video_paths_by_default(
        self, dataset_dir, fs, mock_frame_count
    ):
        video_count = 10
        self.make_video_files(dataset_dir, fs, video_count)

        dataset = VideoFolderDataset(dataset_dir)

        assert len(dataset.video_paths) == video_count

    def test_filtering_video_files(self, dataset_dir, fs, mock_frame_count):
        self.make_video_files(dataset_dir, fs, 10)

        def filter(path):
            return path.name.endswith(("1.mp4", "2.mp4", "3.mp4"))

        dataset = VideoFolderDataset(dataset_dir, filter=filter)

        assert len(dataset.video_paths) == 3
        assert dataset.video_paths[0].name == "video1.mp4"
        assert dataset.video_paths[1].name == "video2.mp4"
        assert dataset.video_paths[2].name == "video3.mp4"

    def test_labels_are_accessible(self, dataset_dir, fs, mock_frame_count):
        video_count = 10
        self.make_video_files(dataset_dir, fs, video_count)

        dataset = VideoFolderDataset(
            dataset_dir, label_set=LambdaLabelSet(lambda name: int(name[-len("X.mp4")]))
        )

        assert len(dataset.labels) == video_count
        assert all([label == i for i, label in enumerate(dataset.labels)])

    def test_transform_is_called_if_provided(self, dataset_dir, fs, monkeypatch):
        def _load_mock_frames(self, frames_idx, video_file):
            frames_count = len(frame_idx_to_list(frames_idx))
            return numpy.zeros((frames_count, 10, 20, 3))

        monkeypatch.setattr(
            torchvideo.datasets.VideoFolderDataset, "_load_frames", _load_mock_frames
        )
        video_count = 10
        self.make_video_files(dataset_dir, fs, video_count)
        mock_transform = Mock(side_effect=lambda frames: frames)
        dataset = VideoFolderDataset(
            dataset_dir, transform=mock_transform, frame_counter=lambda p: 20
        )

        frames = dataset[0]

        mock_transform.assert_called_once_with(frames)

    @staticmethod
    def make_video_files(dataset_dir, fs, video_count):
        for i in range(0, video_count):
            path = os.path.join(dataset_dir, "video{}.mp4".format(i))
            fs.create_file(path)
