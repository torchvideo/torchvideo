# torchvideo
[![Build Status](https://travis-ci.org/torchvideo/torchvideo.svg?branch=master)](https://travis-ci.org/torchvideo/torchvideo)
[![PyPI versions](https://img.shields.io/pypi/pyversions/torchvideo.svg)](https://pypi.org/project/torchvideo/)
[![codecov](https://codecov.io/gh/torchvideo/torchvideo/branch/master/graph/badge.svg)](https://codecov.io/gh/torchvideo/torchvideo)
[![Documentation Status](https://readthedocs.org/projects/torchvideo/badge/?version=latest)](https://torchvideo.readthedocs.io/en/latest/?badge=latest)

This repo is forked from the original torchvideo repo (https://github.com/torchvideo/torchvideo),
fixing gulpio to gulpio2.

A [PyTorch](https://pytorch.org/) library for video-based computer vision tasks. `torchvideo` provides
dataset loaders specialised for video, video frame samplers, and transformations specifically for video.

## Get started

### Set up an accelerated environment in conda


```console
$ conda env create -f environment.yml -n torchvideo
$ conda activate torchvideo

# The following steps are taken from
# https://docs.fast.ai/performance.html#installation

$ conda uninstall -y --force pillow pil jpeg libtiff
$ pip uninstall -y pillow pil jpeg libtiff
$ conda install -y -c conda-forge libjpeg-turbo
$ CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
$ conda install -y jpeg libtiff
```

NOTE: If the installation of `pillow-simd` fails, you can try installing GCC from
conda-forge and trying the install again:

```bash
$ conda install -y gxx_linux-64
$ export CXX=x86_64-conda_cos6-linux-gnu-g++
$ export CC=x86_64-conda_cos6-linux-gnu-gcc
$ CFLAGS="${CFLAGS} -mavx2" pip install --upgrade --no-cache-dir --force-reinstall --no-binary :all: --compile pillow-simd
$ conda install -y jpeg libtiff
```

If you install any new packages, check that `pillow-simd` hasn't be overwritten
by an alternate `pillow` install by running:

```bash
$ python -c "from PIL import Image; print(Image.PILLOW_VERSION)"
```

You should see something like

```
6.0.0.post0
```

Pillow doesn't release with `post` suffixes, so if you have `post` in the version
name, it's likely you have `pillow-simd` installed.


### Install torchvideo

```console
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
