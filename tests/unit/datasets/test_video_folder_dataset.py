import os

import numpy

import torchvideo
from torchvideo.datasets import LambdaLabelSet
from torchvideo.datasets import DummyLabelSet
from torchvideo.datasets import VideoFolderDataset
from torchvideo.datasets import ImageFolderVideoDataset
from torchvideo.samplers import frame_idx_to_list
from ..mock_transforms import (
    MockFramesOnlyTransform,
    MockFramesAndOptionalTargetTransform,
)


class TestVideoFolderDatasetUnit:
    def test_all_videos_are_present_in_video_paths_by_default(
        self, dataset_dir, fs, mock_frame_count
    ):
        video_count = 10
        self.make_video_files(dataset_dir, fs, video_count)

        dataset = VideoFolderDataset(dataset_dir)

        assert len(dataset._video_paths) == video_count

    def test_filtering_video_files(self, dataset_dir, fs, mock_frame_count):
        self.make_video_files(dataset_dir, fs, 10)

        def filter(path):
            return path.name.endswith(("1.mp4", "2.mp4", "3.mp4"))

        dataset = VideoFolderDataset(dataset_dir, filter=filter)

        assert len(dataset._video_paths) == 3
        assert dataset._video_paths[0].name == "video1.mp4"
        assert dataset._video_paths[1].name == "video2.mp4"
        assert dataset._video_paths[2].name == "video3.mp4"

    def test_labels_are_accessible(self, dataset_dir, fs, mock_frame_count):
        video_count = 10
        self.make_video_files(dataset_dir, fs, video_count)

        dataset = VideoFolderDataset(
            dataset_dir, label_set=LambdaLabelSet(lambda name: int(name[-len("X.mp4")]))
        )

        assert len(dataset.labels) == video_count
        assert all([label == i for i, label in enumerate(dataset.labels)])

    def test_transform_is_applied(self, dataset_dir, fs, monkeypatch):
        def _load_mock_frames(self, frames_idx, video_file):
            frames_count = len(frame_idx_to_list(frames_idx))
            return numpy.zeros((frames_count, 10, 20, 3))

        monkeypatch.setattr(
            torchvideo.datasets.video_folder_dataset.VideoFolderDataset,
            "_load_frames",
            _load_mock_frames,
        )
        video_count = 10
        self.make_video_files(dataset_dir, fs, video_count)
        transform = MockFramesOnlyTransform(lambda frames: frames)
        dataset = VideoFolderDataset(
            dataset_dir, transform=transform, frame_counter=lambda p: 20
        )

        frames = dataset[0]

        transform.assert_called_once_with(frames)

    def test_transform_is_passed_target_if_it_supports_it(
        self, dataset_dir, fs, monkeypatch
    ):
        monkeypatch.setattr(
            torchvideo.internal.readers, "default_loader", lambda file, idx: file
        )
        self.make_video_files(dataset_dir, fs, 1)
        transform = MockFramesAndOptionalTargetTransform(lambda f: f, lambda t: t)
        dataset = VideoFolderDataset(
            dataset_dir,
            transform=transform,
            label_set=DummyLabelSet(1),
            frame_counter=lambda p: 20,
        )

        frames, target = dataset[0]

        assert target == 1
        transform.assert_called_once_with(frames, target=target)

    def test_video_ids(self, dataset_dir, fs):
        video_count = 10
        self.make_video_files(dataset_dir, fs, video_count)

        dataset = ImageFolderVideoDataset(
            dataset_dir, "frame_{:05d}.jpg", frame_counter=(lambda path: 10)
        )

        assert list(map(lambda p: p.name, dataset.video_ids)) == sorted(
            ["video{}.mp4".format(i) for i in range(0, video_count)]
        )

    @staticmethod
    def make_video_files(dataset_dir, fs, video_count):
        for i in range(0, video_count):
            path = os.path.join(dataset_dir, "video{}.mp4".format(i))
            fs.create_file(path)
