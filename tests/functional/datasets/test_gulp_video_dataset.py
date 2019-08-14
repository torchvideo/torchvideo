from pathlib import Path
from unittest.mock import Mock

import numpy as np

import pytest
from gulpio import GulpDirectory

from tests import TEST_DATA_ROOT
from torchvideo.datasets import DummyLabelSet
from torchvideo.datasets import GulpVideoDataset
from torchvideo.samplers import LambdaSampler
from unit.mock_transforms import (
    MockFramesAndOptionalTargetTransform,
    MockFramesOnlyTransform,
)


@pytest.fixture(scope="module")
def gulp_path():
    return TEST_DATA_ROOT / "media" / "gulp_output"


@pytest.fixture(scope="module")
def gulp_dataset(gulp_path):
    return GulpVideoDataset(gulp_path)


class TestGulpVideoDataset:
    video_count = 11
    video_length = 50
    size = (368, 640)

    def test_dataset_length(self, gulp_dataset):
        assert len(gulp_dataset) == self.video_count

    def test_video_id(self, gulp_dataset):
        assert gulp_dataset._video_ids[0] == "video0"

    def test_video_range(self, gulp_dataset):
        frames = gulp_dataset[0][0]

        assert frames.min() >= 0
        assert frames.max() <= 1

    def test_loads_all_frames_by_default(self, gulp_dataset):
        dataset = gulp_dataset
        frames, label = dataset[0]

        assert int(label) == 0
        frame_shape = frames.size()
        assert frame_shape == (3, self.video_length, *self.size)

    def test_loading_by_slice(self, gulp_dataset):
        dataset = gulp_dataset
        length = 4
        gulp_dataset.sampler = LambdaSampler(lambda video_length: slice(0, length, 1))

        frames, _ = dataset[0]

        frames_shape = frames.size()
        assert frames_shape == (3, length, *self.size)

    def test_loading_by_list_of_slices(self, gulp_dataset):
        dataset = gulp_dataset
        lengths = 2, 3, 4
        gulp_dataset.sampler = LambdaSampler(
            lambda video_length: [slice(0, l, 1) for l in lengths]
        )

        frames, _ = dataset[0]

        frames_shape = frames.size()
        assert frames_shape == (3, sum(lengths), *self.size)

    def test_loading_by_list_of_int(self, gulp_dataset):
        dataset = gulp_dataset
        frames_idx = [0, 1, 4, 7]

        gulp_dataset.sampler = LambdaSampler(lambda video_length: frames_idx)

        frames, _ = dataset[0]

        frames_shape = frames.size()
        assert frames_shape == (3, len(frames_idx), *self.size)

    def test_filtering_videos(self, gulp_path):
        video_ids = {"video1", "video2", "video3"}

        def filter(video_id: str):
            return video_id in video_ids

        dataset = GulpVideoDataset(gulp_path, filter=filter)

        assert len(dataset) == 3
        assert set(dataset._video_ids) == video_ids

    def test_transforms_are_passed_uint8_ndarray_video(self, gulp_path):
        dataset = GulpVideoDataset(gulp_path, transform=lambda f: f)

        vid, _ = dataset[0]

        assert type(vid) == np.ndarray
        assert vid.dtype == np.uint8
        assert vid.ndim == 4

    def test_transform_is_called(self, gulp_path):
        transform = MockFramesOnlyTransform(lambda frames: frames)
        dataset = GulpVideoDataset(gulp_path, transform=transform)

        frames, _ = dataset[0]

        assert frames is not None
        transform.assert_called_once_with(frames)

    def test_transform_is_passed_target_if_it_supports_it(self, gulp_path):
        transform = MockFramesAndOptionalTargetTransform(lambda f: f, lambda t: t)
        dataset = GulpVideoDataset(
            gulp_path, transform=transform, label_set=DummyLabelSet(1)
        )

        frames, target = dataset[0]

        assert target == 1
        transform.assert_called_once_with(frames, target=target)

    def test_labels_are_accessible(self, gulp_dataset):
        assert len(gulp_dataset.labels) == self.video_count
        assert all([label == "0" for label in gulp_dataset.labels])

    def test_video_ids(self, gulp_dataset):

        assert gulp_dataset.video_ids == sorted(
            ["video{}".format(i) for i in range(0, self.video_count)]
        )

    def test_creating_gulp_video_dataset_from_gulp_directory(self, gulp_path):
        gulp_dir = GulpDirectory(gulp_path)

        dataset = GulpVideoDataset(gulp_dir.output_dir, gulp_directory=gulp_dir)

        assert dataset.gulp_dir == gulp_dir
        assert dataset.gulp_dir.output_dir == dataset.root_path
        assert len(gulp_dir.merged_meta_dict) == len(dataset)

    def test_dataset_throws_error_if_root_path_is_different_from_gulp_dir_path(
        self, gulp_path
    ):
        gulp_dir = GulpDirectory(gulp_path)

        with pytest.raises(ValueError):
            GulpVideoDataset(
                Path(gulp_dir.output_dir).with_name("not-a-gulp-dir"),
                gulp_directory=gulp_dir,
            )
