import numpy as np
import torch
from pretrainedmodels import resnet18
from torch.nn.functional import softmax

from torchvideo.models.resnet3d import resnet3d18
from unit.models.test_utils import make_image_and_boring_video


class TestModelInflation:
    def test_inflated_model_produces_output_closer_to_image_output_than_random_init(
        self
    ):
        model_2d = resnet18(pretrained="imagenet")
        inflated_model_3d = resnet3d18(num_classes=1000, pretrained="imagenet")
        random_init_model_3d = resnet3d18(num_classes=1000, pretrained=None)

        in_channels = 3
        time = 10
        height = 224
        width = 224
        image, boring_video = make_image_and_boring_video(
            in_channels, time, height, width
        )

        with torch.no_grad():
            image_probs = softmax(model_2d(image.unsqueeze(0))).squeeze().numpy()
            inflated_video_probs = (
                softmax(inflated_model_3d(boring_video.unsqueeze(0))).squeeze().numpy()
            )
            random_init_video_probs = (
                softmax(random_init_model_3d(boring_video.unsqueeze(0)))
                .squeeze()
                .numpy()
            )

        inflated_difference = np.linalg.norm(image_probs - inflated_video_probs)
        random_difference = np.linalg.norm(image_probs - random_init_video_probs)
        relative_difference = inflated_difference / random_difference

        # Basically we want a much smaller difference to the image network output using
        # the inflated network than using the randomly initialised network.
        # Ideally the probabilities would be the same, but due to numerical instability
        # issues not even the ordering of the class scores is preserved with the
        # inflated network
        assert relative_difference < 0.1
