"""
Anaflow subpackage providing solutions for the groundwater flow equation.

.. currentmodule:: anaflow.flow

Functions
---------
The following functions are provided

.. autosummary::
   thiem
   ext_thiem2D
   ext_thiem3D
   theis
   ext_theis2D
   ext_theis3D
   diskmodel
   lap_trans_flow_cyl
"""
from __future__ import absolute_import

from anaflow.flow.homogeneous import (
    thiem,
    theis,
)
from anaflow.flow.heterogeneous import (
    ext_thiem2D,
    ext_theis2D,
    ext_thiem3D,
    ext_theis3D,
)
from anaflow.flow.special import (
    diskmodel,
)
from anaflow.flow.laplace import (
    lap_trans_flow_cyl,
)

__all__ = [
    "thiem",
    "theis",
    "ext_thiem2D",
    "ext_theis2D",
    "ext_thiem3D",
    "ext_theis3D",
    "diskmodel",
    "lap_trans_flow_cyl",
]
