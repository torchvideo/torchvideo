import numpy as np

from torchvideo.transforms.transforms.identity_transform import IdentityTransform
from unit.transforms.assertions import assert_preserves_label


class TestIdentityTransform:
    def test_identity_transform_preserves_frames(self):
        transform = IdentityTransform()
        frames = np.ones((5, 4, 4, 3), dtype=np.uint8)
        transformed_frames = transform(frames)

        np.testing.assert_array_equal(transformed_frames, frames)

    def test_propagates_label_unchanged(self):
        frames = np.ones((5, 4, 4, 3), dtype=np.uint8)

        assert_preserves_label(IdentityTransform(), frames)

    def test_repr(self):
        assert repr(IdentityTransform()) == "IdentityTransform()"
