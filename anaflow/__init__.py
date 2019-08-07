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
   grf

Heterogeneous
^^^^^^^^^^^^^

.. currentmodule:: anaflow.flow.heterogeneous

Solutions for heterogeneous aquifers

.. autosummary::
   ext_thiem_2d
   ext_thiem_3d
   ext_thiem_tpl
   ext_thiem_tpl_3d
   ext_theis_2d
   ext_theis_3d
   ext_theis_tpl
   ext_thiem_tpl_3d
   neuman2004
   neuman2004_steady

Extended GRF
^^^^^^^^^^^^

.. currentmodule:: anaflow.flow.ext_grf_model

The extended general radial flow model.

.. autosummary::
   ext_grf
   ext_grf_steady

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
    grf,
    ext_thiem_2d,
    ext_thiem_3d,
    ext_thiem_tpl,
    ext_thiem_tpl_3d,
    ext_theis_2d,
    ext_theis_3d,
    ext_theis_tpl,
    ext_theis_tpl_3d,
    neuman2004,
    neuman2004_steady,
    ext_grf,
    ext_grf_steady,
)
from anaflow.tools.laplace import get_lap_inv, get_lap

__all__ = ["__version__"]
__all__ += [
    "thiem",
    "theis",
    "ext_thiem_2d",
    "ext_thiem_3d",
    "ext_thiem_tpl",
    "ext_thiem_tpl_3d",
    "ext_theis_2d",
    "ext_theis_3d",
    "ext_theis_tpl",
    "ext_theis_tpl_3d",
    "neuman2004",
    "neuman2004_steady",
    "grf",
    "ext_grf",
    "ext_grf_steady",
    "get_lap_inv",
    "get_lap",
]
