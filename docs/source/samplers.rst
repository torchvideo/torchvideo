torchvideo.samplers
===================

.. currentmodule:: torchvideo.samplers

Samplers
--------

Different video models use different strategies in sampling frames: some use sparse
sampling strategies (like TSN, TRN) whereas others like 3D CNNs use dense sampling
strategies. In order to accommodate these different architectures we offer a variety
of sampling strategies with the opportunity to implement your own.

.. contents:: Contents
   :local:
   :depth: 2

FrameSampler
~~~~~~~~~~~~
.. autoclass:: FrameSampler

ClipSampler
~~~~~~~~~~~
.. autoclass:: ClipSampler

FullVideoSampler
~~~~~~~~~~~~~~~~
.. autoclass:: FullVideoSampler

TemporalSegmentSampler
~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: TemporalSegmentSampler

LambdaSampler
~~~~~~~~~~~~~
.. autoclass:: LambdaSampler
