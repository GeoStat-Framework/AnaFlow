# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow solutions in homogeneous aquifers.

.. currentmodule:: anaflow.flow.homogeneous

The following functions are provided

.. autosummary::
   thiem
   theis
   grf_model
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function

import numpy as np

from anaflow.tools.laplace import get_lap_inv
from anaflow.tools.special import well_solution, Shaper, grf
from anaflow.flow.laplace import grf_laplace

__all__ = ["thiem", "theis", "grf_model"]


###############################################################################
# Thiem-solution
###############################################################################


def thiem(rad, Rref, T, Qw, href=0.0):
    """
    The Thiem solution.

    The Thiem solution for steady-state flow under a pumping condition
    in a confined and homogeneous aquifer.
    This solution was presented in [Thiem1906]_.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    Rref : :class:`float`
        Reference radius with known head (see `href`)
    T : :class:`float`
        Given transmissivity of the aquifer
    Qw : :class:`float`
        Pumpingrate at the well
    href : :class:`float`, optional
        Reference head at the reference-radius `Rref`. Default: ``0.0``

    Returns
    -------
    thiem : :class:`numpy.ndarray`
        Array with all heads at the given radii.

    References
    ----------
    .. [Thiem1906] Thiem, G.,
       ''Hydrologische Methoden, J.M. Gebhardt'', Leipzig, 1906.

    Notes
    -----
    The parameters ``rad``, ``Rref`` and ``T`` will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> thiem([1,2,3], 10, 0.001, -0.001)
    array([-0.3664678 , -0.25615   , -0.19161822])
    """
    rad = np.squeeze(rad)

    # check the input
    if Rref <= 0.0:
        raise ValueError("The reference-radius needs to be greater than 0")
    if np.any(rad <= 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if T <= 0.0:
        raise ValueError("The Transmissivity needs to be positiv")

    return -Qw / (2.0 * np.pi * T) * np.log(rad / Rref) + href


###############################################################################
# Theis-solution
###############################################################################


def theis(
    time,
    rad,
    T,
    S,
    Qw,
    struc_grid=True,
    rwell=0.0,
    rinf=np.inf,
    hinf=0.0,
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
    T : :class:`float`
        Given transmissivity of the aquifer
    S : :class:`float`
        Given storativity of the aquifer
    Qw : :class:`float`
        Pumpingrate at the well
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    rwell : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    rinf : :class:`float`, optional
        Radius of the outer boundariy of the aquifer. Default: ``np.inf``
    hinf : :class:`float`, optional
        Reference head at the outer boundary `rinf`. Default: ``0.0``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    References
    ----------
    .. [Theis35] Theis, C.,
       ''The relation between the lowering of the piezometric surface and the
       rate and duration of discharge of a well using groundwater storage'',
       Trans. Am. Geophys. Union, 16, 519-524, 1935
    """
    Input = Shaper(time, rad, struc_grid)
    lap_kwargs = {} if lap_kwargs is None else lap_kwargs

    # check the input
    if rwell < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if rinf <= rwell:
        raise ValueError(
            "The upper boundary needs to be greater than the wellradius"
        )
    if Input.rad_min < rwell:
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if T <= 0.0:
        raise ValueError("The Transmissivity needs to be positiv")
    if S <= 0.0:
        raise ValueError("The Storage needs to be positiv")

    if rwell == 0.0 and rinf == np.inf:
        return well_solution(time, rad, T, S, Qw)

    rpart = np.array([rwell, rinf])
    Tpart = np.array([T])
    Spart = np.array([S])

    # write the paramters in kwargs to use the stehfest-algorithm
    kwargs = {
        "rad": Input.rad,
        "Qw": Qw,
        "dim": 2,
        "lat_ext": 1,
        "rpart": rpart,
        "Spart": Spart,
        "Kpart": Tpart,
    }
    kwargs.update(lap_kwargs)

    res = np.zeros((Input.time_no, Input.rad_no))
    # call the grf-model
    lap_inv = get_lap_inv(grf_laplace, **kwargs)
    res[Input.time_gz, :] = lap_inv(Input.time[Input.time_gz])
    res = Input.reshape(res)
    if Qw > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += hinf
    return res


def grf_model(
    time,
    rad,
    K,
    S,
    Qw,
    dim=2.0,
    lat_ext=1.0,
    struc_grid=True,
    rwell=0.0,
    rinf=np.inf,
    hinf=0.0,
    lap_kwargs=None,
):
    """
    The general radial flow (GRF) model for a pumping test.

    This solution was  presented in [Barker88]_.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    K : :class:`float`
        Given conductivity of the aquifer
    S : :class:`float`
        Given storativity of the aquifer
    Qw : :class:`float`
        Pumpingrate at the well
    dim : :class:`float`
        Fractional dimension of the aquifer. Default: ``2.0``
    lat_ext : :class:`float`
        Lateral extend of the aquifer. Default: ``1.0``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    rwell : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    rinf : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    hinf : :class:`float`, optional
        Reference head at the outer boundary `rinf`. Default: ``0.0``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    References
    ----------
    .. [Barker88] Barker, J.
       ''A generalized radial flow model for hydraulic tests
       in fractured rock.'',
       Water Resources Research 24.10, 1796-1804, 1988
    """
    Input = Shaper(time, rad, struc_grid)
    lap_kwargs = {} if lap_kwargs is None else lap_kwargs

    # check the input
    if rwell < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if rinf <= rwell:
        raise ValueError(
            "The upper boundary needs to be greater than the wellradius"
        )
    if Input.rad_min < rwell:
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if K <= 0.0:
        raise ValueError("The Transmissivity needs to be positiv")
    if S <= 0.0:
        raise ValueError("The Storage needs to be positiv")
    if dim <= 0.0 or dim > 3.0:
        raise ValueError("The dimension needs to be positiv and <= 3")
    if lat_ext <= 0.0:
        raise ValueError("The lateral extend needs to be positiv")

    if rwell == 0.0 and rinf == np.inf:
        return grf(time, rad, K, S, Qw, dim, lat_ext, struc_grid, hinf)

    rpart = np.array([rwell, rinf])
    Kpart = np.array([K])
    Spart = np.array([S])

    # write the paramters in kwargs to use the stehfest-algorithm
    kwargs = {
        "rad": Input.rad,
        "Qw": Qw,
        "dim": dim,
        "lat_ext": lat_ext,
        "rpart": rpart,
        "Spart": Spart,
        "Kpart": Kpart,
    }
    kwargs.update(lap_kwargs)

    res = np.zeros((Input.time_no, Input.rad_no))
    # call the grf-model
    lap_inv = get_lap_inv(grf_laplace, **kwargs)
    res[Input.time_gz, :] = lap_inv(Input.time[Input.time_gz])
    res = Input.reshape(res)
    if Qw > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += hinf
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
