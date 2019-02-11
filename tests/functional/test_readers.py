from tests import TEST_DATA_ROOT
from torchvideo.internal.readers import lintel_loader


class TestLintelReader:
    video_path = TEST_DATA_ROOT / "media" / "big_buck_bunny_360p_5mb.mp4"
    frame_count = 759
    width, height = (640, 368)

    def test_reading_sequential_contiguous_frames(self):
        frames = list(lintel_loader(self.video_path, [0, 1, 2, 3]))

        self.check_frames(frames, 4)

    def test_reading_last_frame(self):
        frames = self.load_frames([self.frame_count - 1])

        assert len(frames) == 1

    def test_reading_frames_out_of_order(self):
        frames = self.load_frames([4, 1, 3, 0])

        self.check_frames(frames, 4)

    def test_reading_frames_beyond_length_of_video(self):
        frames = self.load_frames([self.frame_count - 1, self.frame_count])

        self.check_frames(frames, 2)

    def load_frames(self, frame_idx):
        return list(lintel_loader(self.video_path, frame_idx))

    def check_frames(self, frames, expected_frame_count):
        assert len(frames) == expected_frame_count
        for frame in frames:
            assert frame.width == self.width
            assert frame.height == self.height
            assert frame.mode == "RGB"
