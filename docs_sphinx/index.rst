.. gimmick documentation master file, created by
   sphinx-quickstart on Fri Jun 11 22:32:58 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

Gimmick
=============
Lets summarize what we are going to generate

Introduction
**************************
Gimmick is a library for generating realastic looking images though use of popular algorithms. The primary focus of this library is to provide easy to use interface for all the interaction. User need to provide a set of images in form of 3D or 4D vector/array for it to learn.

The whole purpose of library is its plug and play mechanism meaning user can switch the underlying algortihm and code will remain same.

Installation
**************************
.. include:: installation.rst


Quick Examples
**************************

.. include:: examples.rst

Documentation
=============

.. include:: documentation_supported_tables.rst

Distributer
***********
.. automodule:: gimmick.distributer
   :members:

Models
******
This section contains class details of models which are using to learn pattern in images for image generation.

.. autoclass:: gimmick.models.autoencoder.AutoEncoder
  :members:

.. autoclass:: gimmick.models.autoencoder_dense.Model
.. autoclass:: gimmick.models.autoencoder_lstm.Model
.. autoclass:: gimmick.models.autoencoder_cnn.Model
.. autoclass:: gimmick.models.autoencoder_cnn_variational.Model

Image Operations
*****************
.. automodule:: gimmick.image_op.functions
   :members:

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
