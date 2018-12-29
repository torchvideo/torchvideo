torchvideo.samplers
===================

.. currentmodule:: torchvideo.samplers

Samplers
--------

Different video models use different strategies in sampling frames: some use sparse
sampling strategies (like TSN, TRN) whereas others like 3D CNNs use dense sampling
strategies. In order to accommodate these different architectures we offer a variety
of sampling strategies with the opportunity to implement your own.

.. toctree:: Contents
   :depth: 2

FrameSampler
~~~~~~~~~~~~
.. autoclass:: FrameSampler
    :members:

FullVideoSampler
~~~~~~~~~~~~~~~~
.. autoclass:: FullVideoSampler
    :members:

TemporalSegmentSampler
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TemporalSegmentSampler
    :members:
