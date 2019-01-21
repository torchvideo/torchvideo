import torch


def time_to_channel(tensor: torch.Tensor) -> torch.Tensor:
    r"""Reshape video tensor of shape :math:`(C, T, H, W)` into
    :math:`(C \times T, H, W)`

    Args:
        tensor: Tensor video of size :math:`(C, T, H, W)`

    Returns:
        Tensor of shape :math:`(C \times T, H, W)`

    """
    tensor_ndim = len(tensor.size())
    if tensor_ndim != 4:
        raise ValueError("Expected 4D tensor but was {}D".format(tensor_ndim))
    h_w_shape = tensor.shape[-2:]
    return tensor.reshape((-1, *h_w_shape))
