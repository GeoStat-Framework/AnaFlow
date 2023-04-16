"""
Anaflow subpackage providing flow-solutions for the groundwater flow equation.

Subpackages
^^^^^^^^^^^

.. currentmodule:: anaflow.flow

.. autosummary::
   :toctree:

   laplace

Solutions
^^^^^^^^^

Homogeneous
~~~~~~~~~~~

Solutions for homogeneous aquifers

.. autosummary::
   :toctree:

   thiem
   theis
   grf

Heterogeneous
~~~~~~~~~~~~~

Solutions for heterogeneous aquifers

.. autosummary::
   :toctree:

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
   :toctree:

   ext_grf
   ext_grf_steady
"""
from anaflow.flow.ext_grf_model import ext_grf, ext_grf_steady
from anaflow.flow.heterogeneous import (
    ext_theis_2d,
    ext_theis_3d,
    ext_theis_tpl,
    ext_theis_tpl_3d,
    ext_thiem_2d,
    ext_thiem_3d,
    ext_thiem_tpl,
    ext_thiem_tpl_3d,
    neuman2004,
    neuman2004_steady,
)
from anaflow.flow.homogeneous import grf, theis, thiem

__all__ = [
    "thiem",
    "theis",
    "grf_model",
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
]
