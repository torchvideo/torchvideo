.. torchvideo documentation master file, created by
   sphinx-quickstart on Wed Dec 26 17:45:11 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

torchvideo
==========

Similar to :mod:`torchvision`, :mod:`torchvideo` is a library for working with video in
pytorch. It contains transforms and dataset classes. It is built on top of
:mod:`torchvision` and designed to be used in conjunction.

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   datasets
   samplers
   transforms
   tools

   bibliography

.. automodule:: torchvideo
   :members:


Installation
------------

Install torchvideo from PyPI with:

.. code-block:: bash

   $ pip install torchvideo


or the cutting edge branch from github with:

.. code-block:: bash

   $ pip install git+https://github.com/willprice/torchvideo.git

We **strongly** advise you to install Pillow-simd to speed up image transformations.
Do this after installing torchvideo.

.. code-block:: bash

   $ pip install pillow-simd
