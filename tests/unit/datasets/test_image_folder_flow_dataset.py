import os

from torchvideo.datasets import ImageFolderFlowDataset, LambdaLabelSet, DummyLabelSet
from unit.mock_transforms import (
    MockFramesOnlyTransform,
    MockFramesAndOptionalTargetTransform,
)


class TestImageFolderFlowDataset:
    def test_empty_directory_has_zero_length(self, dataset_dir):
        dataset = ImageFolderFlowDataset(dataset_dir, "{axis}/frame_{index:05d}.jpg")

        assert len(dataset) == 0

    def test_directory_with_one_video_has_length_one(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 1)

        dataset = ImageFolderFlowDataset(dataset_dir, "{axis}/frame_{index:05d}.jpg")

        assert len(dataset) == 1

    def test_video_ids_are_directory_paths(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 2)

        dataset = ImageFolderFlowDataset(dataset_dir, "{axis}/frame_{index:05d}.jpg")

        assert list(map(lambda p: p.name, dataset.video_ids)) == ["video0", "video1"]

    def test_examples_are_labeled_with_label_set(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 2)

        dataset = ImageFolderFlowDataset(
            dataset_dir,
            "{axis}/frame_{index:05d}.jpg",
            label_set=LambdaLabelSet(lambda video_name: int(video_name[-1])),
        )

        assert dataset.labels == [0, 1]

    def test_transform_is_applied_to_examples(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 2)

        transform = MockFramesOnlyTransform(lambda fs: fs)
        dataset = ImageFolderFlowDataset(
            dataset_dir,
            "{axis}/frame_{index:05d}.jpg",
            label_set=DummyLabelSet(1),
            transform=transform,
        )

        frames, target = dataset[0]

        transform.assert_called_once_with(frames)

    def test_transform_is_passed_target_if_supported(self, dataset_dir):
        self.make_video_dirs(dataset_dir, 2)

        transform = MockFramesAndOptionalTargetTransform(lambda fs: fs, lambda t: t)
        dataset = ImageFolderFlowDataset(
            dataset_dir,
            "{axis}/frame_{index:05d}.jpg",
            label_set=DummyLabelSet(1),
            transform=transform,
        )

        frames, target = dataset[0]

        transform.assert_called_once_with(frames, target=target)

    @staticmethod
    def make_video_dirs(dataset_dir, video_count, frame_count=10):
        for i in range(0, video_count):
            video_dir = os.path.join(dataset_dir, "video{}".format(i))
            for axis in ["u", "v"]:
                for i in range(frame_count):
                    frame_path = os.path.join(
                        video_dir, axis, "frame_{:05d}.jpg".format(i + 1)
                    )
                    os.makedirs(os.path.dirname(frame_path), exist_ok=True)
                    open(frame_path, "ab").close()
