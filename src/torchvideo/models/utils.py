from typing import Dict, Any

import torch
from torch.utils import model_zoo

CONV_3D_DIM_TIME = 2


def rename_keys(
    pretrained_state_dict: Dict[str, torch.Tensor], keys_to_rename: Dict[str, str]
) -> Dict[str, torch.Tensor]:
    keys = pretrained_state_dict.keys()
    to_rename = dict()
    for key in keys:
        for old_key, new_key in keys_to_rename.items():
            if key.startswith(old_key):
                to_rename[key] = key.replace(old_key, new_key)
    for old_key, new_key in to_rename.items():
        pretrained_state_dict[new_key] = pretrained_state_dict[old_key]
        del pretrained_state_dict[old_key]
    return pretrained_state_dict


def inflate_pretrained(
    model: torch.nn.Module, settings: Dict[str, Any]
) -> torch.nn.Module:
    """
    Inflate a 3D network from 2D variant's weights. Originally proposed by
    Carreira et al. in
    "Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset"
    https://arxiv.org/abs/1705.07750.

    Args:
        model: 3D network module to inflate
        settings: dictionary containing pretrained model details (url, input_space,
            input_size, input_range, mean, std)

    Returns:
        model loaded with inflated 2D parameters.
    """
    net_3d_state_dict = model.state_dict()
    pretrained_state_dict = model_zoo.load_url(settings["url"])
    rename_keys(pretrained_state_dict, {"fc": "last_linear"})
    inflate_state_dict(net_3d_state_dict, pretrained_state_dict)
    model.load_state_dict(pretrained_state_dict)
    _copy_settings_to_model(model, settings)
    return model


def inflate_state_dict(state_dict_3d, state_dict_2d):
    """
    Inflate a 3D model state dictionary from a trained 2D model's state dictionary.

    .. warning::
        Updates ``state_dict_3d`` in place overwriting 3D parameters that can
        be inflated from 2D counterparts.
    """
    for param_name, net_3d_param in state_dict_3d.items():
        try:
            net_2d_param = state_dict_2d[param_name]
            if net_2d_param.shape != net_3d_param.shape:
                state_dict_2d[param_name] = inflate_param(net_3d_param, net_2d_param)
        except KeyError:
            pass


def inflate_param(net_3d_param, net_2d_param):
    """
    Inflate a 2D network parameter (e.g. from a Conv2D, or BatchNorm2D) to its 3D
    counterpart (e.g. to a Conv3D, or BatchNorm3D). The uninitialised 3D parameter
    must be provided as this provides details on the new dimension.

    Args:
        net_3d_param: Untrained 3D network parameter
        net_2d_param: Trained 2d network parameter

    Returns:
        Inflated 2D network parameter with the same shape as ``net_3d_param``.
    """
    kernel_time_size = net_3d_param.shape[CONV_3D_DIM_TIME]
    # Conv3d kernels have parameters shaped [out, in, time, height, width]
    # Conv2d kernels have parameters shaped [out, in, height, width]
    # We need to introduce the new time dimension to the 2D kernel to
    # inflate it to 3D.
    kernel_shape_3d = (
        net_2d_param.shape[0],
        net_2d_param.shape[1],
        kernel_time_size,
        net_2d_param.shape[2],
        net_2d_param.shape[3],
    )
    # Duplicate the parameter across the time dimension
    # It's important to call .contiguous here otherwise the pointer to the 2D kernel
    # is just duplicated rather than the data itself. We need to expand it out so that
    # when we divide by the kernel_time_size we don't shoot ourselves in the foot and
    # end up dividing the parameters by kernel_time_size^2!
    inflated_pretrained_params = (
        net_2d_param.unsqueeze(CONV_3D_DIM_TIME).expand(kernel_shape_3d)
    ).contiguous()

    # Normalise the inflated kernel so that kernel produces the same
    # response on a boring video (image duplicated over time) as the
    # original 2D kernel does on the image.
    inflated_pretrained_params.div_(kernel_time_size)
    return inflated_pretrained_params


def _copy_settings_to_model(model, settings):
    model.input_space = settings["input_space"]
    model.input_size = settings["input_size"]
    model.input_range = settings["input_range"]
    model.mean = settings["mean"]
    model.std = settings["std"]
