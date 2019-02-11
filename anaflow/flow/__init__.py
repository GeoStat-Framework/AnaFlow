# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow-solutions for the groundwater flow equation.

Subpackages
^^^^^^^^^^^

.. currentmodule:: anaflow.flow

.. autosummary::
    homogeneous
    heterogeneous
    special
    laplace

Solutions
^^^^^^^^^

Homogeneous
~~~~~~~~~~~

.. currentmodule:: anaflow.flow.homogeneous

Solutions for homogeneous aquifers

.. autosummary::
   thiem
   theis

Heterogeneous
~~~~~~~~~~~~~

.. currentmodule:: anaflow.flow.heterogeneous

Solutions for heterogeneous aquifers

.. autosummary::
   ext_thiem2D
   ext_theis2D
   ext_thiem3D
   ext_theis3D

Special
~~~~~~~

.. currentmodule:: anaflow.flow.special

Special solutions for special aquifers

.. autosummary::
   grf_model
"""
from __future__ import absolute_import

from anaflow.flow.homogeneous import thiem, theis
from anaflow.flow.heterogeneous import (
    ext_thiem2D,
    ext_theis2D,
    ext_thiem3D,
    ext_theis3D,
)
from anaflow.flow.special import grf_model

__all__ = [
    "thiem",
    "theis",
    "ext_thiem2D",
    "ext_theis2D",
    "ext_thiem3D",
    "ext_theis3D",
    "grf_model",
]
