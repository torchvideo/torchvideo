import pytest
from pyfakefs.fake_filesystem import FakeFilesystem

import torchvideo


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
