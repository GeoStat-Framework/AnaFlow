# -*- coding: utf-8 -*-
"""
Purpose
=======

Anaflow provides several analytical and semi-analytical solutions for the
groundwater-flow-equation.

Subpackages
===========

.. autosummary::
    flow
    tools

Solutions
=========

Homogeneous
^^^^^^^^^^^

.. currentmodule:: anaflow.flow.homogeneous

Solutions for homogeneous aquifers

.. autosummary::
   thiem
   theis
   grf_model

Heterogeneous
^^^^^^^^^^^^^

.. currentmodule:: anaflow.flow.heterogeneous

Solutions for heterogeneous aquifers

.. autosummary::
   ext_thiem2D
   ext_theis2D
   ext_thiem3D
   ext_theis3D

Special
^^^^^^^

.. currentmodule:: anaflow.flow.special

Special solutions for special aquifers

.. autosummary::
   grf_disk

Laplace
=======

.. currentmodule:: anaflow.tools.laplace

Helping functions related to the laplace-transformation

.. autosummary::
   get_lap
   get_lap_inv
"""
from __future__ import absolute_import

from anaflow._version import __version__

from anaflow.flow import (
    thiem,
    theis,
    grf_model,
    ext_thiem2D,
    ext_theis2D,
    ext_thiem3D,
    ext_theis3D,
    grf_disk,
)
from anaflow.tools.laplace import get_lap_inv, get_lap

__all__ = ["__version__"]

__all__ += [
    "thiem",
    "theis",
    "ext_thiem2D",
    "ext_theis2D",
    "ext_thiem3D",
    "ext_theis3D",
    "grf_model",
    "grf_disk",
    "get_lap_inv",
    "get_lap",
]
