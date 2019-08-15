from pathlib import Path

import hypothesis
import numpy as np
import torch


class TorchRandomState:
    def getstate(self) -> np.ndarray:
        return torch.get_rng_state().numpy()

    def setstate(self, state: np.ndarray) -> None:
        return torch.set_rng_state(torch.from_numpy(state))

    def seed(self, seed) -> None:
        torch.manual_seed(seed)


hypothesis.register_random(TorchRandomState())

_here = Path(__file__).parent
TEST_DATA_ROOT = _here / "data"
