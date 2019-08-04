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
   grf_model

Heterogeneous
~~~~~~~~~~~~~

.. currentmodule:: anaflow.flow.heterogeneous

Solutions for heterogeneous aquifers

.. autosummary::
   ext_thiem2D
   ext_theis2D
   ext_thiem3D
   ext_theis3D
   ext_theis_tpl
   neuman2004
   neuman2004_steady

Special
~~~~~~~

.. currentmodule:: anaflow.flow.special

Special solutions for special aquifers

.. autosummary::
   grf_disk
   grf_stedy
"""
from __future__ import absolute_import

from anaflow.flow.homogeneous import thiem, theis, grf_model
from anaflow.flow.heterogeneous import (
    ext_thiem2D,
    ext_theis2D,
    ext_thiem3D,
    ext_theis3D,
    ext_theis_tpl,
    neuman2004,
    neuman2004_steady,
)
from anaflow.flow.special import grf_disk, grf_steady

__all__ = [
    "thiem",
    "theis",
    "grf_model",
    "ext_thiem2D",
    "ext_theis2D",
    "ext_thiem3D",
    "ext_theis3D",
    "ext_theis_tpl",
    "neuman2004",
    "neuman2004_steady",
    "grf_disk",
    "grf_steady",
]
