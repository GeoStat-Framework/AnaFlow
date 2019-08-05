# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow-solutions for the groundwater flow equation.

Subpackages
^^^^^^^^^^^

.. currentmodule:: anaflow.flow

.. autosummary::
    homogeneous
    heterogeneous
    ext_grf
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
   ext_thiem_2d
   ext_thiem_3d
   ext_theis_2d
   ext_theis_3d
   ext_theis_tpl
   neuman2004
   neuman2004_steady

Extended GRF
~~~~~~~~~~~~

.. currentmodule:: anaflow.flow.ext_grf

The extended general radial flow model.

.. autosummary::
   ext_grf
   ext_grf_steady
"""
from __future__ import absolute_import

from anaflow.flow.homogeneous import thiem, theis, grf
from anaflow.flow.heterogeneous import (
    ext_thiem_2d,
    ext_thiem_3d,
    ext_theis_2d,
    ext_theis_3d,
    ext_theis_tpl,
    neuman2004,
    neuman2004_steady,
)
from anaflow.flow.ext_grf import ext_grf, ext_grf_steady

__all__ = [
    "thiem",
    "theis",
    "grf_model",
    "ext_thiem_2d",
    "ext_thiem_3d",
    "ext_theis_2d",
    "ext_theis_3d",
    "ext_theis_tpl",
    "neuman2004",
    "neuman2004_steady",
    "grf",
    "ext_grf",
    "ext_grf_steady",
]
