class TestTransformImports:
    def test_importing_center_crop(self):
        from torchvideo.transforms import CenterCropVideo

    def test_importing_collect_frames(self):
        from torchvideo.transforms import CollectFrames

    def test_importing_compose(self):
        from torchvideo.transforms import Compose

    def test_importing_multiscale_crop_video(self):
        from torchvideo.transforms import MultiScaleCropVideo

    def test_importing_ndarray_to_pil_video(self):
        from torchvideo.transforms import NDArrayToPILVideo

    def test_importing_normalize_video(self):
        from torchvideo.transforms import NormalizeVideo

    def test_importing_pil_video_to_tensor(self):
        from torchvideo.transforms import PILVideoToTensor

    def test_importing_random_crop_video(self):
        from torchvideo.transforms import RandomCropVideo

    def test_importing_random_horizontal_flip_video(self):
        from torchvideo.transforms import RandomHorizontalFlipVideo

    def test_importing_resize_video(self):
        from torchvideo.transforms import ResizeVideo

    def test_importing_time_apply(self):
        from torchvideo.transforms import TimeApply

    def test_importing_time_to_channel(self):
        from torchvideo.transforms import TimeToChannel

    def test_importing_random_resized_crop_video(self):
        from torchvideo.transforms import RandomResizedCropVideo
