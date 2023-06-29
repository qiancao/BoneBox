.. BoneBox documentation master file, created by
   sphinx-quickstart on Fri Jun 23 10:29:19 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to BoneBox's documentation!
===================================

BoneBox:
--------

Tools for bone modeling, evaluation and biomarker development.

.. figure:: skeleton.gif
   :alt: alt text


.. toctree::
   :maxdepth: 1
   :caption: Table of Contents:

   example
   toc/randomdropout_docs
   toc/medialaxisutils_docs
   toc/pvutils_docs
   toc/meshutils_docs
   toc/trabeculaephantom_docs
   toc/trabeculaevoronoi_docs
   toc/lsi_docs
   api

.. _installation:

Install Dependencies
--------------------

Clone the repository.

::

   git clone https://github.com/qiancao/BoneBox.git
   cd BoneBox

Create a new conda environment with the needed dependencies.

::

   conda env create -f environment_bonebox.yml
   conda activate bonebox


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
