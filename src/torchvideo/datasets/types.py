import torch
from typing import Any, Callable, Iterator

import numpy as np
from PIL.Image import Image

Label = Any
Transform = Callable[[Any], torch.Tensor]
PILVideoTransform = Callable[[Iterator[Image]], torch.Tensor]
NDArrayVideoTransform = Callable[[np.ndarray], torch.Tensor]


class empty_label:
    pass
