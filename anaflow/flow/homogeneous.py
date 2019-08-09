# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow solutions in homogeneous aquifers.

.. currentmodule:: anaflow.flow.homogeneous

The following functions are provided

.. autosummary::
   thiem
   theis
   grf
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function

import numpy as np

from anaflow.tools.special import well_solution, grf_solution
from anaflow.flow.ext_grf_model import ext_grf, ext_grf_steady

__all__ = ["thiem", "theis", "grf"]


###############################################################################
# Thiem-solution
###############################################################################


def thiem(rad, r_ref, transmissivity, rate=-1e-4, h_ref=0.0):
    """
    The Thiem solution.

    The Thiem solution for steady-state flow under a pumping condition
    in a confined and homogeneous aquifer.
    This solution was presented in [Thiem1906]_.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated.
    r_ref : :class:`float`
        Reference radius with known head (see `h_ref`).
    transmissivity : :class:`float`
        Transmissivity of the aquifer.
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_ref : :class:`float`, optional
        Reference head at the reference-radius `r_ref`. Default: ``0.0``

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii.

    References
    ----------
    .. [Thiem1906] Thiem, G.,
       ''Hydrologische Methoden, J.M. Gebhardt'', Leipzig, 1906.

    Notes
    -----
    The parameters ``rad``, ``r_ref`` and ``transmissivity``
    will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> thiem([1,2,3], 10, 0.001, -0.001)
    array([-0.3664678 , -0.25615   , -0.19161822])
    """
    return ext_grf_steady(rad, r_ref, transmissivity, 2, 1, rate, h_ref)


###############################################################################
# Theis-solution
###############################################################################


def theis(
    time,
    rad,
    storage,
    transmissivity,
    rate=-1e-4,
    r_well=0.0,
    r_bound=np.inf,
    h_bound=0.0,
    struc_grid=True,
    lap_kwargs=None,
):
    """
    The Theis solution.

    The Theis solution for transient flow under a pumping condition
    in a confined and homogeneous aquifer.
    This solution was presented in [Theis35]_.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    storage : :class:`float`
        Storage coefficient of the aquifer.
    conductivity : :class:`float`
        Conductivity of the aquifer.
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundariy of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary, as well as initial condition.
        Default: ``0.0``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    References
    ----------
    .. [Theis35] Theis, C.,
       ''The relation between the lowering of the piezometric surface and the
       rate and duration of discharge of a well using groundwater storage'',
       Trans. Am. Geophys. Union, 16, 519-524, 1935
    """
    if np.isclose(r_well, 0) and np.isposinf(r_bound) and lap_kwargs is None:
        return well_solution(time, rad, storage, transmissivity, rate)
    return ext_grf(
        time=time,
        rad=rad,
        S_part=[storage],
        K_part=[transmissivity],
        R_part=[r_well, r_bound],
        dim=2,
        lat_ext=1,
        rate=rate,
        K_well=None,
        h_bound=h_bound,
        lap_kwargs=lap_kwargs,
        struc_grid=struc_grid,
    )


def grf(
    time,
    rad,
    storage,
    conductivity,
    dim=2,
    lat_ext=1.0,
    rate=-1e-4,
    r_well=0.0,
    r_bound=np.inf,
    h_bound=0.0,
    struc_grid=True,
    lap_kwargs=None,
):
    """
    The general radial flow (GRF) model for a pumping test.

    This solution was  presented in [Barker88]_.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated.
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated.
    storage : :class:`float`
        Storage coefficient of the aquifer.
    conductivity : :class:`float`
        Conductivity of the aquifer.
    dim : :class:`float`, optional
        Fractional dimension of the aquifer. Default: ``2.0``
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer. Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary, as well as initial condition.
        Default: ``0.0``
    struc_grid : :class:`bool`, optional
        If this is set to "False", the "rad" and "time" array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    References
    ----------
    .. [Barker88] Barker, J.
       ''A generalized radial flow model for hydraulic tests
       in fractured rock.'',
       Water Resources Research 24.10, 1796-1804, 1988
    """
    if np.isclose(r_well, 0) and np.isposinf(r_bound) and lap_kwargs is None:
        return grf_solution(
            time=time,
            rad=rad,
            storage=storage,
            conductivity=conductivity,
            dim=dim,
            lat_ext=lat_ext,
            rate=rate,
            h_bound=h_bound,
            struc_grid=struc_grid,
        )
    return ext_grf(
        time=time,
        rad=rad,
        S_part=[storage],
        K_part=[conductivity],
        R_part=[r_well, r_bound],
        dim=dim,
        lat_ext=lat_ext,
        rate=rate,
        K_well=None,
        h_bound=h_bound,
        struc_grid=struc_grid,
        lap_kwargs=lap_kwargs,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
