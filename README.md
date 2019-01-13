# torchvideo
[![Build Status](https://travis-ci.org/willprice/torchvideo.svg?branch=master)](https://travis-ci.org/willprice/torchvideo)
[![PyPI versions](https://img.shields.io/pypi/pyversions/torchvideo.svg)](https://pypi.org/project/torchvideo/)
[![codecov](https://codecov.io/gh/willprice/torchvideo/branch/master/graph/badge.svg)](https://codecov.io/gh/willprice/torchvideo)
[![Documentation Status](https://readthedocs.org/projects/torchvideo/badge/?version=latest)](https://torchvideo.readthedocs.io/en/latest/?badge=latest)


**WARNING: Do not use this library. It is still in development. When this notice is
removed it will be sufficiently stable for usage.**

A [PyTorch](https://pytorch.org/) library for video-based computer vision tasks. `torchvideo` provides
dataset loaders specialised for video, video frame samplers, and transformations specifically for video.

## Get started

```bash
$ pip install git+https://github.com/willprice/torchvideo.git@master
```

## Learn how to use `torchvideo`

Check out the [example notebooks](/examples), you can launch these on binder without
having to install anything locally!

## Acknowledgements

Thanks to the following people and projects

* [yjxiong](https://github.com/yjxiong) for his work on TSN and publicly
  available [pytorch implementation](https://github.com/yjxiong/tsn-pytorch)
  from which many of the transforms in this project started from.
* [dukebw](https://github.com/dukebw) for his excellent
  [lintel](https://github.com/dukebw/lintel) FFmpeg video loading library.
* [hypothesis](https://hypothesis.readthedocs.io) and the team behind it. This
  has been used heavily in testing the project.
