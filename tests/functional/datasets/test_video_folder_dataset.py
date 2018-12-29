from ... import TEST_DATA_ROOT
from torchvideo.datasets import VideoFolderDataset


def test_video_folder_dataset():
    video_count = 11
    frame_count = 50
    size = (368, 640)

    video_folder = TEST_DATA_ROOT / 'media' / 'video_folder'

    dataset = VideoFolderDataset(video_folder)

    assert len(dataset) == video_count

    frames = dataset[1]
    shape = frames.shape
    assert shape == (3, frame_count, *size)
