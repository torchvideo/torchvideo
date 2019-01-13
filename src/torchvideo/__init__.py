from . import datasets, transforms, samplers, tools
from .__version__ import __title__, __description__, __url__, __version__
from .__version__ import __author__, __author_email__, __license__, __copyright__

__all__ = ["get_video_backend", "datasets", "transforms", "samplers", "tools"]

_video_backend = "lintel"


def get_video_backend() -> str:
    return _video_backend
