# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow solutions in heterogeneous aquifers.

.. currentmodule:: anaflow.flow.heterogeneous

The following functions are provided

.. autosummary::
   ext_thiem2D
   ext_thiem3D
   ext_theis2D
   ext_theis3D
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.special import exp1, expi

from anaflow.tools.laplace import get_lap_inv
from anaflow.tools.special import aniso, specialrange_cut, Shaper
from anaflow.tools.mean import annular_hmean
from anaflow.tools.coarse_graining import T_CG, T_CG_error, K_CG, K_CG_error
from anaflow.flow.laplace import grf_laplace

__all__ = ["ext_thiem2D", "ext_thiem3D", "ext_theis2D", "ext_theis3D"]


###############################################################################
# 2D version of extended Thiem
###############################################################################


def ext_thiem2D(rad, Rref, TG, sig2, corr, Qw, href=0.0, Twell=None, prop=1.6):
    """
    The extended Thiem solution in 2D.

    The extended Thiem solution for steady-state flow under
    a pumping condition in a confined aquifer.
    The type curve is describing the effective drawdown
    in a 2D statistical framework, where the transmissivity distribution is
    following a log-normal distribution with a gaussian correlation function.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    Rref : :class:`float`
        Reference radius with known head (see `href`)
    TG : :class:`float`
        Geometric-mean transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
    Qw : :class:`float`
        Pumpingrate at the well
    href : :class:`float`, optional
        Reference head at the reference-radius `Rref`. Default: ``0.0``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    ext_thiem2D : :class:`numpy.ndarray`
        Array with all heads at the given radii.

    References
    ----------
    .. [Zech2013] Zech, A.
       ''Impact of Aqifer Heterogeneity on Subsurface Flow and Salt
       Transport at Different Scales: from a method determine parameters
       of heterogeneous permeability at local scale to a large-scale model
       for the sedimentary basin of Thuringia.''
       PhD thesis, Friedrich-Schiller-Universität Jena, 2013

    Notes
    -----
    The parameters ``rad``, ``Rref``, ``TG``, ``sig2``, ``corr``, ``Twell``
    and ``prop`` will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_thiem2D([1,2,3], 10, 0.001, 1, 10, -0.001)
    array([-0.53084596, -0.35363029, -0.25419375])
    """
    rad = np.array(rad, dtype=float)
    # check the input
    if Rref <= 0.0:
        raise ValueError(
            "The upper boundary needs to be greater than the wellradius"
        )
    if np.any(rad <= 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if TG <= 0.0:
        raise ValueError("The Transmissivity needs to be positiv")
    if Twell is not None and Twell <= 0.0:
        raise ValueError("The Transmissivity at the well needs to be positiv")
    if sig2 <= 0.0:
        raise ValueError("The variance needs to be positiv")
    if corr <= 0.0:
        raise ValueError("The correlationlength needs to be positiv")
    if prop <= 0.0:
        raise ValueError("The proportionalityfactor needs to be positiv")

    # define some substitions to shorten the result
    if Twell is not None:
        chi = np.log(Twell) - np.log(TG)
    else:
        chi = -sig2 / 2.0

    Q = -Qw / (4.0 * np.pi * TG)
    C = (prop / corr) ** 2

    # derive the result
    res = -expi(-chi / (1.0 + C * rad ** 2))
    res -= np.exp(-chi) * exp1(chi / (1.0 + C * rad ** 2) - chi)
    res += expi(-chi / (1.0 + C * Rref ** 2))
    res += np.exp(-chi) * exp1(chi / (1.0 + C * Rref ** 2) - chi)
    res *= Q
    res += href

    return res


###############################################################################
# 3D version of extended Thiem
###############################################################################


def ext_thiem3D(
    rad, Rref, KG, sig2, corr, e, Qw, L, href=0.0, Kwell="KH", prop=1.6
):
    """
    The extended Thiem solution in 3D.

    The extended Thiem solution for steady-state flow under
    a pumping condition in a confined aquifer.
    The type curve is describing the effective drawdown
    in a 3D statistical framework, where the conductivity distribution is
    following a log-normal distribution with a gaussian correlation function
    and taking vertical anisotropy into account.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    Rref : :class:`float`
        Reference radius with known head (see `href`)
    KG : :class:`float`
        Geometric-mean conductivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the conductivity-distribution
    corr : :class:`float`
        corralation-length of conductivity-distribution
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    Qw : :class:`float`
        Pumpingrate at the well
    L : :class:`float`
        Thickness of the aquifer
    href : :class:`float`, optional
        Reference head at the reference-radius `Rref`. Default: ``0.0``
    Kwell : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    ext_thiem3D : :class:`numpy.ndarray`
        Array with all heads at the given radii.

    References
    ----------
    .. [Zech2013] Zech, A.
       ''Impact of Aqifer Heterogeneity on Subsurface Flow and Salt
       Transport at Different Scales: from a method determine parameters
       of heterogeneous permeability at local scale to a large-scale model
       for the sedimentary basin of Thuringia.''
       PhD thesis, Friedrich-Schiller-Universität Jena, 2013

    Notes
    -----
    The parameters ``rad``, ``Rref``, ``KG``, ``sig2``, ``corr``, ``Kwell``
    and ``prop`` will be checked for positivity.
    The anisotropy ``e`` factor must be greater 0 and less or equal 1.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_thiem3D([1,2,3], 10, 0.001, 1, 10, 1, -0.001, 1)
    array([-0.48828026, -0.31472059, -0.22043022])
    """
    rad = np.array(rad, dtype=float)
    # check the input
    if Rref <= 0.0:
        raise ValueError(
            "The upper boundary needs to be greater than the wellradius"
        )
    if np.any(rad <= 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if Kwell != "KA" and Kwell != "KH" and not isinstance(Kwell, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(Kwell, float) and Kwell <= 0.0:
        raise ValueError("The well-conductivity needs to be positiv")
    if KG <= 0.0:
        raise ValueError("The Transmissivity needs to be positiv")
    if sig2 <= 0.0:
        raise ValueError("The variance needs to be positiv")
    if corr <= 0.0:
        raise ValueError("The correlationlength needs to be positiv")
    if L <= 0.0:
        raise ValueError("The aquifer-thickness needs to be positiv")
    if not 0.0 < e <= 1.0:
        raise ValueError("The anisotropy-ratio must be > 0 and <= 1")
    if prop <= 0.0:
        raise ValueError("The proportionalityfactor needs to be positiv")

    # define some substitions to shorten the result
    Kefu = KG * np.exp(sig2 * (0.5 - aniso(e)))
    if Kwell == "KH":
        chi = sig2 * (aniso(e) - 1.0)
    elif Kwell == "KA":
        chi = sig2 * aniso(e)
    else:
        chi = np.log(Kwell) - np.log(Kefu)

    Q = -Qw / (2.0 * np.pi * Kefu)
    C = (prop / corr / e ** (1.0 / 3.0)) ** 2

    sub11 = np.sqrt(1.0 + C * Rref ** 2)
    sub12 = np.sqrt(1.0 + C * rad ** 2)

    sub21 = np.log(sub12 + 1.0) - np.log(sub11 + 1.0)
    sub21 -= 1.0 / sub12 - 1.0 / sub11

    sub22 = np.log(sub12) - np.log(sub11)
    sub22 -= 0.50 / sub12 ** 2 - 0.50 / sub11 ** 2
    sub22 -= 0.25 / sub12 ** 4 - 0.25 / sub11 ** 4

    # derive the result
    res = np.exp(-chi) * (np.log(rad) - np.log(Rref))
    res += sub21 * np.sinh(chi) + sub22 * (1.0 - np.cosh(chi))
    res *= Q
    res += href

    return res


###############################################################################
# 2D version of extended Theis
###############################################################################


def ext_theis2D(
    time,
    rad,
    TG,
    sig2,
    corr,
    S,
    Qw,
    struc_grid=True,
    rwell=0.0,
    rinf=np.inf,
    hinf=0.0,
    Twell=None,
    T_err=0.01,
    prop=1.6,
    parts=30,
    lap_kwargs=None,
):
    """
    The extended Theis solution in 2D.

    The extended Theis solution for transient flow under
    a pumping condition in a confined aquifer.
    The type curve is describing the effective drawdown
    in a 2D statistical framework, where the transmissivity distribution is
    following a log-normal distribution with a gaussian correlation function.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    TG : :class:`float`
        Geometric-mean transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
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
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    hinf : :class:`float`, optional
        Reference head at the outer boundary ``rinf``. Default: ``0.0``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``
    T_err : :class:`float`, optional
        Absolute error for the farfield transmissivity for calculating the
        cutoff-point of the solution. Default: ``0.01``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    ext_theis2D : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    The parameters ``rad``, ``Rref``, ``TG``, ``sig2``, ``corr``, ``S``,
    ``Twell`` and ``prop`` will be checked for positivity.
    ``T_err`` must be greater 0 and less or equal 1.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_theis2D([10,100], [1,2,3], 0.001, 1, 10, 0.001, -0.001)
    array([[-0.33737576, -0.17400123, -0.09489812],
           [-0.58443489, -0.40847176, -0.31095166]])
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
    if TG <= 0.0:
        raise ValueError("The Transmissivity needs to be positiv")
    if Twell is not None and Twell <= 0.0:
        raise ValueError("The Transmissivity at the well needs to be positiv")
    if sig2 <= 0.0:
        raise ValueError("The variance needs to be positiv")
    if corr <= 0.0:
        raise ValueError("The correlationlength needs to be positiv")
    if S <= 0.0:
        raise ValueError("The Storage needs to be positiv")
    if prop <= 0.0:
        raise ValueError("The proportionalityfactor needs to be positiv")
    if parts <= 1:
        raise ValueError("The numbor of partitions needs to be at least 2")
    if not 0.0 < T_err < 1.0:
        raise ValueError(
            "The relative error of Transmissivity needs to be within (0,1)"
        )

    # genearte rlast from a given relativ-error to farfield-transmissivity
    rlast = T_CG_error(T_err, TG, sig2, corr, prop, Twell)
    # generate the partition points
    if rlast > rwell:
        rpart = specialrange_cut(rwell, rinf, parts + 1, rlast)
    else:
        rpart = np.array([rwell, rinf])
    # calculate the harmonic mean transmissivity values within each partition
    Tpart = annular_hmean(
        T_CG, rpart, TG=TG, sig2=sig2, corr=corr, prop=prop, Twell=Twell
    )

    # write the paramters in kwargs to use the grf-model
    kwargs = {
        "rad": Input.rad,
        "Qw": Qw,
        "dim": 2,
        "lat_ext": 1,
        "rpart": rpart,
        "Spart": np.full_like(Tpart, S),
        "Kpart": Tpart,
        "Kwell": T_CG(rwell, TG, sig2, corr, prop, Twell),
    }
    kwargs.update(lap_kwargs)

    res = np.zeros((Input.time_no, Input.rad_no))
    # call the grf-model
    lap_inv = get_lap_inv(grf_laplace, **kwargs)
    res[Input.time > 0, :] = lap_inv(Input.time[Input.time > 0])
    res = Input.reshape(res)
    if Qw > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += hinf
    return res


###############################################################################
# 3D version of extended Theis
###############################################################################


def ext_theis3D(
    time,
    rad,
    KG,
    sig2,
    corr,
    e,
    S,
    Qw,
    L,
    struc_grid=True,
    rwell=0.0,
    rinf=np.inf,
    hinf=0.0,
    Kwell="KH",
    K_err=0.01,
    prop=1.6,
    parts=30,
    lap_kwargs=None,
):
    """
    The extended Theis solution in 3D.

    The extended Theis solution for transient flow under
    a pumping condition in a confined aquifer.
    The type curve is describing the effective drawdown
    in a 3D statistical framework, where the transmissivity distribution is
    following a log-normal distribution with a gaussian correlation function
    and taking vertical anisotropy into account.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    KG : :class:`float`
        Geometric-mean conductivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the conductivity-distribution
    corr : :class:`float`
        corralation-length of conductivity-distribution
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    S : :class:`float`
        Given storativity of the aquifer
    Qw : :class:`float`
        Pumpingrate at the well
    L : :class:`float`
        Thickness of the aquifer
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
    Kwell : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    K_err : :class:`float`, optional
        Absolute error for the farfield conductivity for calculating the
        cutoff-point of the solution, if ``rinf=inf``. Default: ``0.01``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    ext_theis3D : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    The parameters ``rad``, ``Rref``, ``KG``, ``sig2``, ``corr``, ``S``,
    ``Twell`` and ``prop`` will be checked for positivity.
    The Anisotropy-ratio ``e`` must be greater 0 and less or equal 1.
    ``T_err`` must be greater 0 and less or equal 1.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_theis3D([10,100], [1,2,3], 0.001, 1, 10, 1, 0.001, -0.001, 1)
    array([[-0.32756786, -0.16717569, -0.09141211],
           [-0.5416396 , -0.36982684, -0.27798614]])
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
    if Kwell != "KA" and Kwell != "KH" and not isinstance(Kwell, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(Kwell, float) and Kwell <= 0.0:
        raise ValueError("The well-conductivity needs to be positiv")
    if KG <= 0.0:
        raise ValueError("The conductivity needs to be positiv")
    if sig2 <= 0.0:
        raise ValueError("The variance needs to be positiv")
    if corr <= 0.0:
        raise ValueError("The correlationlength needs to be positiv")
    if S <= 0.0:
        raise ValueError("The Storage needs to be positiv")
    if L <= 0.0:
        raise ValueError("The aquifer-thickness needs to be positiv")
    if prop <= 0.0:
        raise ValueError("The proportionalityfactor needs to be positiv")
    if parts <= 1:
        raise ValueError("The numbor of partitions needs to be at least 2")
    if not 0.0 < K_err < 1.0:
        raise ValueError(
            "The relative error of Transmissivity needs to be within (0,1)"
        )

    # genearte rlast from a given relativ-error to farfield-conductivity
    rlast = K_CG_error(K_err, KG, sig2, corr, e, prop, Kwell=Kwell)
    # generate the partition points
    if rlast > rwell:
        rpart = specialrange_cut(rwell, rinf, parts + 1, rlast)
    else:
        rpart = np.array([rwell, rinf])
    # calculate the harmonic mean conductivity values within each partition
    Kpart = annular_hmean(
        K_CG, rpart, KG=KG, sig2=sig2, corr=corr, e=e, prop=prop, Kwell=Kwell
    )

    # write the paramters in kwargs to use the stehfest-algorithm
    kwargs = {
        "rad": Input.rad,
        "Qw": Qw,
        "dim": 2,
        "lat_ext": L,
        "rpart": rpart,
        "Spart": np.full_like(Kpart, S),
        "Kpart": Kpart,
        "Kwell": K_CG(rwell, KG, sig2, corr, e, prop, Kwell),
    }
    kwargs.update(lap_kwargs)

    res = np.zeros((Input.time_no, Input.rad_no))
    # call the grf-model in laplace space
    lap_inv = get_lap_inv(grf_laplace, **kwargs)
    res[Input.time > 0, :] = lap_inv(Input.time[Input.time > 0])
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
