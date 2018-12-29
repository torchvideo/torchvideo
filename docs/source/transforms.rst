torchvideo.transforms
=====================

.. currentmodule:: torchvideo.transforms

This module contains video transforms similar to those found in
:mod:`torchvision.transforms` specialised for image transformations. Like the transforms
from :mod:`torchvision.transforms` you can chain together successive transforms using
:class:`torchvision.transforms.Compose`.

.. contents:: Contents
    :local:
    :depth: 2


Examples
--------

Typically your transformation pipelines will be compose of a sequence of PIL video
transforms followed by a :class:`CollectFrames` transform and a
:class:`PILVideoToTensor`: transform.


.. code-block:: python

    import torchvideo.transforms as VT
    import torchvision.transforms as IT
    from torchvision.transforms import Compose

    transform = Compose([
        VT.CenterCropVideo((224, 224)),  # (h, w)
        VT.CollectFrames(),
        VT.PILVideoToTensor()
    ])



Video Datatypes
---------------

torchvideo represents videos in a variety of formats:

- PIL video: A list of a PIL Images, this is useful for applying image data
  augmentations
- tensor video: A tensor of shape :math:`(C, T, H, W)` for feeding a network.



Transforms on PIL Videos
------------------------

These transforms all take an iterator/iterable of :class:`PIL.Image.Image` and produce
an iterator of :class:`PIL.Image.Image`. To draw image out of the transform you should
compose your sequence of PIL Video transforms with :class:`CollectFrames`.

CenterCropVideo
~~~~~~~~~~~~~~~

.. autoclass:: CenterCropVideo
    :special-members: __call__

RandomCropVideo
~~~~~~~~~~~~~~~
.. autoclass:: RandomCropVideo
    :special-members: __call__

RandomHorizontalFlipVideo
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandomHorizontalFlipVideo
    :special-members: __call__

ResizeVideo
~~~~~~~~~~~
.. autoclass:: ResizeVideo
    :special-members: __call__

MultiScaleCropVideo
~~~~~~~~~~~~~~~~~~~
.. autoclass:: MultiScaleCropVideo
    :special-members: __call__

RandomResizedCropVideo
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: RandomResizedCropVideo
    :special-members: __call__

TimeApply
~~~~~~~~~
.. autoclass:: TimeApply
    :special-members: __call__



Transforms on Torch.\*Tensor videos
-----------------------------------

The input to these transforms should be a tensor of shape :math:`(C, T, H, W)`

NormalizeVideo
~~~~~~~~~~~~~~
.. autoclass:: NormalizeVideo
    :special-members: __call__


Conversion transforms
---------------------

CollectFrames
~~~~~~~~~~~~~
.. autoclass:: CollectFrames
    :special-members: __call__

PILVideoToTensor
~~~~~~~~~~~~~~~~
.. autoclass:: PILVideoToTensor
    :special-members: __call__
