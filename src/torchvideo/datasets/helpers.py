from torchvideo.transforms.transforms.compose import _supports_target


def invoke_transform(transform, frames, label):
    if _supports_target(transform):
        return transform(frames, label)
    return transform(frames), label
