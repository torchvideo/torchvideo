import argparse
import pickle
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

from torchvideo.models.c3d import C3D

parser = argparse.ArgumentParser(
    description="Convert C3D caffe2 models to pytorch state dicts",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument("c3d_caffe2_weights", type=Path)
parser.add_argument("c3d_pytorch_weights", type=Path)
parser.add_argument(
    "--force",
    action="store_true",
    help="Overwrite c3d_pytorch_weights if it already exists",
)


def read_blobs(blob_path: Path) -> Dict[str, np.ndarray]:
    with open(blob_path, "rb") as f:
        # Python 2 pickles are encoded using latin1 whereas python 3 defaults to
        # ASCII. The VMZ codebase these model checkpoints come from is python 2.
        return pickle.load(f, encoding="latin1")["blobs"]


def convert_c3d_weights(blobs: Dict[str, np.ndarray]) -> Dict[str, torch.Tensor]:
    state_dict = dict()
    for name, blob in blobs.items():
        weight = torch.from_numpy(blob)
        if name.endswith("_w"):
            new_name = re.sub(r"_w$", ".weight", name)
        elif name.endswith("_b"):
            new_name = re.sub(r"_b$", ".bias", name)
        elif name.endswith("_s"):
            new_name = re.sub(r"_s$", ".weight", name)
        elif name.endswith("_rm"):
            new_name = re.sub(r"_rm$", ".running_mean", name)
        # Despite being named riv for running inverse variance, this is in fact the
        # running variance. There is no need to invert it.
        elif name.endswith("_riv"):
            new_name = re.sub(r"_riv$", ".running_var", name)
        else:
            raise NotImplementedError(
                f"Don't know how to convert {name} with param " f"of shape {blob.shape}"
            )

        if new_name.startswith("last_out"):
            new_name = re.sub(r"last_out_L\d+", "last_linear", new_name)

        state_dict[new_name] = weight
    return state_dict


def main(args):
    if args.c3d_pytorch_weights.exists() and not args.force:
        print(
            f"{args.c3d_pytorch_weights} already exists, pass --force to overwrite "
            f"them"
        )
        sys.exit(1)
    blobs = read_blobs(args.c3d_caffe2_weights)
    print("Converting weights")
    state_dict = convert_c3d_weights(blobs)
    model = _make_c3d(blobs)
    print("Checking weights can be loaded")
    model.load_state_dict(state_dict)
    print("Weights loaded successfully")
    torch.save(state_dict, args.c3d_pytorch_weights)
    print(f"state_dict saved to {args.c3d_pytorch_weights}")


def _make_c3d(blobs):
    last_linear_name = [k for k in blobs.keys() if k.startswith("last_out")][0]
    in_channels = blobs["conv1a_w"].shape[1]  # (out_channels, in_channels, 3, 3, 3)
    class_count = int(re.match(r"last_out_L(\d+).*", last_linear_name).groups()[0])
    fc6_dim = blobs["fc6_w"].shape[0]
    fc7_dim = blobs["fc7_w"].shape[0]
    return C3D(class_count, in_channels=in_channels, fc6_dim=fc6_dim, fc7_dim=fc7_dim)


if __name__ == "__main__":
    main(parser.parse_args())
