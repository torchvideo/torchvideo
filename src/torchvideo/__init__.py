import warnings

from torchvideo import datasets, transforms, samplers
from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __author__, __author_email__, __license__, __copyright__

_default_backend = "lintel"
_video_backend = _default_backend


def set_video_backend(backend: str) -> None:
    global _video_backend
    backend = backend.lower()
    if backend not in {"lintel", "nvvl"}:
        raise ValueError(
            "Invalid video backend '{}'. Options are: 'lintel', 'pyav', and 'nvvl'".format(
                backend
            )
        )
    if backend == "pyav":
        try:
            import av
        except ImportError:
            warnings.warn(
                "PyAV not installed, defaulting to {}".format(_default_backend),
                RuntimeWarning,
            )
            backend = _default_backend
    if backend == "nvvl":
        try:
            import nvvl
        except ImportError:
            warnings.warn(
                "nvvl not installed, defaulting to {}".format(_default_backend),
                RuntimeWarning,
            )
            backend = _default_backend

    _video_backend = backend


def get_video_backend() -> str:
    return _video_backend
