# -*- coding: utf-8 -*-
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
   stehfest

Subpackages
-----------
Using any of these subpackages requires an explicit import.  For example,
``import anaflow.helper``.

.. autosummary::

   gwsolutions - Solutions for the groundwater flow equation
   laplace - Functions concerning the laplace-transform
   helper - Several helper-functions

"""
from __future__ import absolute_import

from anaflow.gwsolutions import (thiem, theis,
                                 ext_thiem2D, ext_theis2D,
                                 ext_thiem3D, ext_theis3D,
                                 diskmodel)
from anaflow.laplace import (stehfest)

__all__ = ["thiem", "theis",
           "ext_thiem2D",
           "ext_theis2D",
           "ext_thiem3D",
           "ext_theis3D",
           "diskmodel",
           "stehfest"]
