torchvideo.datasets
===================

.. currentmodule:: torchvideo.datasets

Datasets
--------

.. contents:: Contents
   :local:
   :depth: 2

VideoDataset
~~~~~~~~~~~~
.. autoclass:: VideoDataset
    :special-members: __getitem__, __len__


ImageFolderVideoDataset
~~~~~~~~~~~~~~~~~~~~~~~
.. autoclass:: ImageFolderVideoDataset
    :special-members: __getitem__, __len__

VideoFolderDataset
~~~~~~~~~~~~~~~~~~
.. autoclass:: VideoFolderDataset
    :special-members: __getitem__, __len__

GulpVideoDataset
~~~~~~~~~~~~~~~~
.. autoclass:: GulpVideoDataset
    :special-members: __getitem__, __len__

Label Sets
----------

Label sets are an abstraction over how your video data is labelled. This provides
flexibility in swapping out different storage methods and labelling methods. All
datasets optionally take a :class:`LabelSet` that performs the mapping between
example and label.

LabelSet
~~~~~~~~
.. autoclass:: LabelSet
