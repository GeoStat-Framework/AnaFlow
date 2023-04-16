# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow-solutions for the groundwater flow equation.

Subpackages
^^^^^^^^^^^

.. currentmodule:: anaflow.flow

.. autosummary::
    laplace

Solutions
^^^^^^^^^

Homogeneous
~~~~~~~~~~~

Solutions for homogeneous aquifers

.. autosummary::
   :toctree: generated

   thiem
   theis
   grf

Heterogeneous
~~~~~~~~~~~~~

Solutions for heterogeneous aquifers

.. autosummary::
   :toctree: generated

   ext_thiem_2d
   ext_thiem_3d
   ext_thiem_tpl
   ext_thiem_tpl_3d
   ext_theis_2d
   ext_theis_3d
   ext_theis_tpl
   ext_theis_tpl_3d
   neuman2004
   neuman2004_steady

Extended GRF
~~~~~~~~~~~~

The extended general radial flow model.

.. autosummary::
   :toctree: generated

   ext_grf
   ext_grf_steady
"""
from anaflow.flow.homogeneous import thiem, theis, grf
from anaflow.flow.heterogeneous import (
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
)
from anaflow.flow.ext_grf_model import ext_grf, ext_grf_steady
from anaflow.flow.Neuman import neuman_unconfined


__all__ = [
    "thiem",
    "theis",
    "grf",
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
    "neuman_unconfined"
]
