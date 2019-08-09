import numpy as np
import torch
from pretrainedmodels import resnet18
from torch.nn.functional import softmax

from torchvideo.models.resnet3d import resnet3d18
from unit.models.test_utils import make_image_and_boring_video


class TestModelInflation:
    def test_randomly_initialized_model_produces_different_output_on_boring_video_to_2d_model(
        self
    ):
        model_2d = resnet18(pretrained="imagenet")
        model_3d = resnet3d18(num_classes=1000, pretrained=None)
        in_channels = 3
        time = 10
        height = 224
        width = 224
        image, boring_video = make_image_and_boring_video(
            in_channels, time, height, width
        )

        with torch.no_grad():
            image_probs = softmax(model_2d(image.unsqueeze(0))).squeeze().numpy()
            video_probs = softmax(model_3d(boring_video.unsqueeze(0))).squeeze().numpy()
        mean_class_prob_difference = np.abs(image_probs - video_probs).mean()
        assert mean_class_prob_difference > 1e-4

    def test_inflated_model_produces_same_output_on_boring_video(self):
        model_2d = resnet18(pretrained="imagenet")
        model_3d = resnet3d18(num_classes=1000, pretrained="imagenet")
        in_channels = 3
        time = 10
        height = 224
        width = 224
        image, boring_video = make_image_and_boring_video(
            in_channels, time, height, width
        )

        with torch.no_grad():
            image_probs = softmax(model_2d(image.unsqueeze(0))).squeeze().numpy()
            video_probs = softmax(model_3d(boring_video.unsqueeze(0))).squeeze().numpy()
        mean_class_prob_difference = np.abs(image_probs - video_probs).mean()
        # Typically the inflation causes the logits to be different, to the extent
        # the the sorted classes indices are different, however, the rough ordering of
        # classes is preserved. The best test I could come up with is checking that
        # average deviance of the probability
        assert mean_class_prob_difference < 1e-4
