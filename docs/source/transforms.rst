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

Target parameters
-----------------

All transforms support a `target` parameter. Currently these don't do anything, but
allow you to implement transforms on targets as well as frames. At some point in
future it is the intention that we'll support transforms of things like masks, or
allow you to plug your own target transforms into these classes.


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


Optical flow stored as flattened :math:`(u, v)` pairs like
:math:`(u_0, v_1, u_1, v_1, \ldots, u_n, v_n)` that are then stacked into the channel
dimension would be dealt with like so:

.. code-block:: python

    import torchvideo.transforms as VT
    import torchvision.transforms as IT
    from torchvision.transforms import Compose

    transform = Compose([
        VT.CenterCropVideo((224, 224)),  # (h, w)
        VT.CollectFrames(),
        VT.PILVideoToTensor(),
        VT.TimeToChannel()
    ])



Video Datatypes
---------------

torchvideo represents videos in a variety of formats:

- *PIL video*: A list of a PIL Images, this is useful for applying image data
  augmentations
- *tensor video*: A :class:`torch.Tensor` of shape :math:`(C, T, H, W)` for feeding a
  network.
- *NDArray video*: A :class:`numpy.ndarray` of shape either :math:`(T, H, W, C)` or
  :math:`(C, T, H, W)`. The reason for the multiple channel shapes is that most
  loaders load in :math:`(T, H, W, C)` format, however tensors formatted for input
  into a network typically are formatted in :math:`(C, T, H, W)`. Permuting the
  dimensions is a costly operation, so supporting both format allows for efficient
  implementation of transforms without have to invert the conversion from one format
  to the other.


Composing Transforms
--------------------

Transforms can be composed with :class:`Compose`. This functions in exactly the same
way as torchvision's implementation, however it also supports chaining transforms
that require, or optionally support, or don't support a target parameter. It handles
the marshalling of targets around and into those transforms depending upon their
support allowing you to mix transforms defined in this library (all of which support
a target parameter) and those defined in other libraries.

Additionally, we provide a :class:`IdentityTransform` that has a nicer ``__repr__``
suitable for use as a default transform in :class:`Compose` pipelines.


Compose
~~~~~~~

.. autoclass:: Compose
    :special-members: __call__

IdentityTransform
~~~~~~~~~~~~~~~~~

.. autoclass:: IdentityTransform
    :special-members: __call__

----

Transforms on PIL Videos
------------------------

These transforms all take an iterator/iterable of :class:`PIL.Image.Image` and produce
an iterator of :class:`PIL.Image.Image`. To materialize the iterator the you should
compose your sequence of PIL video transforms with :class:`CollectFrames`.


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

----


Transforms on Torch.\*Tensor videos
-----------------------------------

These transform are applicable to `torch.*Tensor` videos only. The input to these
transforms should be a tensor of shape :math:`(C, T, H, W)`.

NormalizeVideo
~~~~~~~~~~~~~~
.. autoclass:: NormalizeVideo
    :special-members: __call__

TimeToChannel
~~~~~~~~~~~~~
.. autoclass:: TimeToChannel
    :special-members: __call__

----


Conversion transforms
---------------------

These transforms are for converting between different video representations. Typically
your transformation pipeline will operate on iterators of ``PIL`` images which
will then be aggregated by ``CollectFrames`` and then coverted to a tensor via
``PILVideoToTensor``.


CollectFrames
~~~~~~~~~~~~~
.. autoclass:: CollectFrames
    :special-members: __call__

PILVideoToTensor
~~~~~~~~~~~~~~~~
.. autoclass:: PILVideoToTensor
    :special-members: __call__

NDArrayToPILVideo
~~~~~~~~~~~~~~~~~
.. autoclass:: NDArrayToPILVideo
    :special-members: __call__

----


Functional Transforms
---------------------

Functional transforms give you fine-grained control of the transformation pipeline. As
opposed to the transformations above, functional transforms don’t contain a random
number generator for their parameters.

.. currentmodule:: torchvideo.transforms.functional


normalize
~~~~~~~~~
.. autofunction:: normalize

time_to_channel
~~~~~~~~~~~~~~~
.. autofunction:: time_to_channel
