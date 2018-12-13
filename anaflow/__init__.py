"""
=======
AnaFlow
=======

Contents
--------
Anaflow provides several analytical and semi-analytical solutions for the
groundwater-flow-equation.

Functions
---------
The following functions are provided directly

.. autosummary::

   thiem
   theis
   ext_thiem2D
   ext_theis2D
   ext_thiem3D
   ext_theis3D
   diskmodel
   get_lap_inv
   stehfest
"""
from __future__ import absolute_import

from anaflow.flow import (
    thiem,
    theis,
    ext_thiem2D,
    ext_theis2D,
    ext_thiem3D,
    ext_theis3D,
    diskmodel,
)
from anaflow.laplace import get_lap_inv, stehfest

__all__ = [
    "thiem",
    "theis",
    "ext_thiem2D",
    "ext_theis2D",
    "ext_thiem3D",
    "ext_theis3D",
    "diskmodel",
    "get_lap_inv",
    "stehfest",
]

__version__ = "0.2.5"
