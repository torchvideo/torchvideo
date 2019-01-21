import os
from pathlib import Path

from torchvideo.datasets import LambdaLabelSet
from torchvideo.datasets import DummyLabelSet
from torchvideo.datasets import ImageFolderVideoDataset
from ..mock_transforms import (
    MockFramesOnlyTransform,
    MockFramesAndOptionalTargetTransform,
)


class TestImageFolderVideoDatasetUnit:
    def test_all_videos_folders_are_present_in_video_dirs_by_default(self, dataset_dir):
        video_count = 10
        self.make_video_dirs(dataset_dir, video_count)

        dataset = ImageFolderVideoDataset(dataset_dir, "frame_{:05d}.jpg")

        assert len(dataset._video_dirs) == video_count

    def test_filtering_video_folders(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 10)

        def filter(video_path: Path):
            return video_path.name.endswith(("1", "2", "3"))

        dataset = ImageFolderVideoDataset(
            dataset_dir, "frame_{:05d}.jpg", filter=filter
        )

        assert len(dataset._video_dirs) == 3
        assert dataset._video_dirs[0].name == "video1"
        assert dataset._video_dirs[1].name == "video2"
        assert dataset._video_dirs[2].name == "video3"

    def test_labels_are_accessible(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 10)

        dataset = ImageFolderVideoDataset(
            dataset_dir,
            "frame_{:05d}.jpg",
            label_set=LambdaLabelSet(lambda p: int(p[-1])),
        )

        assert 10 == len(dataset.labels)
        assert all([label == i for i, label in enumerate(dataset.labels)])

    def test_transform_is_applied(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 1)
        transform = MockFramesOnlyTransform(lambda frames: frames)

        dataset = ImageFolderVideoDataset(
            dataset_dir, "frame_{:05d}.jpg", transform=transform
        )

        frames = dataset[0]

        transform.assert_called_once_with(frames)

    def test_transform_is_passed_target_if_it_supports_it(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 1)
        transform = MockFramesAndOptionalTargetTransform(lambda f: f, lambda t: t)
        dataset = ImageFolderVideoDataset(
            dataset_dir,
            "frame_{:05d}.jpg",
            transform=transform,
            label_set=DummyLabelSet(1),
        )

        frames, target = dataset[0]

        assert target == 1
        transform.assert_called_once_with(frames, target=target)

    def test_video_ids(self, dataset_dir):
        video_count = 10
        self.make_video_dirs(dataset_dir, video_count)

        dataset = ImageFolderVideoDataset(
            dataset_dir, "frame_{:05d}.jpg", frame_counter=(lambda path: 10)
        )

        assert list(map(lambda p: p.name, dataset.video_ids)) == sorted(
            ["video{}".format(i) for i in range(0, video_count)]
        )

    @staticmethod
    def make_video_dirs(dataset_dir, video_count, frame_count=10):
        for i in range(0, video_count):
            video_dir = os.path.join(dataset_dir, "video{}".format(i))
            os.makedirs(video_dir)
            for i in range(frame_count):
                frame_path = os.path.join(video_dir, "frame_{:05d}.jpg".format(i + 1))
                open(frame_path, "ab").close()
