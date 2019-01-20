import PIL.Image
import numpy as np
from hypothesis import given

from torchvideo.transforms import PILVideoToTensor
from ..strategies import pil_video
from .assertions import assert_preserves_label


class TestPILVideoToTensor:
    def test_repr(self):
        assert repr(PILVideoToTensor()) == "PILVideoToTensor()"

    @given(pil_video())
    def test_transform(self, video):
        transform = PILVideoToTensor()
        tensor = transform(video)
        width, height = video[0].size
        n_channels = 3 if video[0].mode == "RGB" else 1
        assert tensor.size(0) == n_channels
        assert tensor.size(1) == len(video)
        assert tensor.size(2) == height
        assert tensor.size(3) == width

    def test_rescales_between_0_and_1(self):
        transform = PILVideoToTensor()
        frame_arr = 255 * np.ones(shape=(10, 20, 3), dtype=np.uint8)
        frame_arr[0:5, 0:10, :] = 0
        video = [PIL.Image.fromarray(frame_arr)]
        tensor = transform(video)

        assert tensor.min().item() == 0
        assert tensor.max().item() == 1

    def test_disabled_rescale(self):
        transform = PILVideoToTensor(rescale=False)
        frame_arr = 255 * np.ones(shape=(10, 20, 3), dtype=np.uint8)
        frame_arr[0:5, 0:10, :] = 0
        video = [PIL.Image.fromarray(frame_arr)]
        tensor = transform(video)

        assert tensor.min().item() == 0
        assert tensor.max().item() == 255

    def test_propagates_label_unchanged(self):
        video = pil_video(min_width=1, min_height=1).example()
        transform = PILVideoToTensor()

        assert_preserves_label(transform, video)
