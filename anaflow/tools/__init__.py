# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing miscellaneous tools.

Subpackages
^^^^^^^^^^^

.. currentmodule:: anaflow.tools

.. autosummary::
    laplace
    mean
    special
    coarse_graining

Functions
^^^^^^^^^

Annular mean
~~~~~~~~~~~~

.. currentmodule:: anaflow.tools.mean

Functions to calculate dimension dependent annular means of a function.

.. autosummary::
   annular_fmean
   annular_amean
   annular_gmean
   annular_hmean
   annular_pmean

Coarse Graining solutions
~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: anaflow.tools.coarse_graining

Effective Coarse Graining conductivity/transmissivity solutions.

.. autosummary::
   T_CG
   K_CG
   TPL_CG

Special
~~~~~~~

.. currentmodule:: anaflow.tools.special

Special functions.

.. autosummary::
   step_f
   specialrange
   specialrange_cut
   neuman2004_trans
   aniso

Laplace
~~~~~~~

.. currentmodule:: anaflow.tools.laplace

Helping functions related to the laplace-transformation

.. autosummary::
   get_lap
   get_lap_inv
"""
from anaflow.tools.mean import (
    annular_fmean,
    annular_amean,
    annular_gmean,
    annular_hmean,
    annular_pmean,
)
from anaflow.tools.special import (
    step_f,
    specialrange,
    specialrange_cut,
    neuman2004_trans,
    aniso,
)
from anaflow.tools.coarse_graining import T_CG, K_CG, TPL_CG
from anaflow.tools.laplace import get_lap, get_lap_inv

__all__ = [
    "get_lap",
    "get_lap_inv",
    "annular_fmean",
    "annular_amean",
    "annular_gmean",
    "annular_hmean",
    "annular_pmean",
    "step_f",
    "specialrange",
    "specialrange_cut",
    "neuman2004_trans",
    "aniso",
    "T_CG",
    "K_CG",
    "TPL_CG",
]
