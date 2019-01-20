import PIL.Image

try:
    from scipy import stats
except ImportError:
    stats = None

pil_interpolation_settings = [
    PIL.Image.NEAREST,
    PIL.Image.BOX,
    PIL.Image.BILINEAR,
    PIL.Image.HAMMING,
    PIL.Image.BICUBIC,
    PIL.Image.LANCZOS,
]
