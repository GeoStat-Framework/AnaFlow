# -*- coding: utf-8 -*-
"""
AnaFlow: A python-package containing analytical solutions for groundwater flow
==============================================================================

Contents
--------
Anaflow provides several analytical and semi-analytical solutions for the
groundwater-flow-equation.

Functions
---------
The following functions are provided directly

::

 thiem                        --- Thiem solution for steady state pumping
 theis                        --- Theis solution for transient pumping
 ext_thiem2D                  --- extended Thiem solution in 2D
 ext_theis2D                  --- extended Theis solution in 2D
 ext_thiem3D                  --- extended Thiem solution in 3D
 ext_theis3D                  --- extended Theis solution in 3D
 diskmodel                    --- Solution for a diskmodel
 stehfest                     --- Stehfest algorithm for laplace inversion

Subpackages
-----------
Using any of these subpackages requires an explicit import.  For example,
``import anaflow.helper``.

::

 gwsolutions                  --- Solutions for the groundwater flow equation
 laplace                      --- Functions concerning the laplace-transform
 helper                       --- Several helper-functions

Utility tools
-------------
::

 __version__       --- Anaflow version string

"""
from __future

from anaflow.gwsolutions import (thiem, theis,
                                 ext_thiem2D, ext_theis2D,
                                 ext_thiem3D, ext_theis3D,
                                 diskmodel)
from anflow.laplace import (stehfest)

__all__ = ["thiem", "theis",
           "ext_thiem2D", "ext_theis2D",
           "ext_thiem3D", "ext_theis3D",
           "diskmodel",
           "stehfest"]
