# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow solutions in heterogeneous aquifers.

.. currentmodule:: anaflow.flow.heterogeneous

The following functions are provided

.. autosummary::
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
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function

import functools as ft

import numpy as np
from scipy.special import exp1, expi

from anaflow.tools.special import aniso, specialrange_cut, neuman2004_trans
from anaflow.tools.mean import annular_hmean
from anaflow.tools.coarse_graining import (
    T_CG,
    T_CG_error,
    K_CG,
    K_CG_error,
    TPL_CG,
    TPL_CG_error,
)
from anaflow.flow.ext_grf_model import ext_grf, ext_grf_steady

__all__ = [
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
]


###############################################################################
# 2D version of extended Thiem
###############################################################################


def ext_thiem_2d(
    rad,
    r_ref,
    trans_gmean,
    var,
    len_scale,
    rate=-1e-4,
    h_ref=0.0,
    T_well=None,
    prop=1.6,
):
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
    r_ref : :class:`float`
        Radius of the reference head.
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_ref : :class:`float`, optional
        Reference head at the reference-radius `r_ref`. Default: ``0.0``
    T_well : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    head : :class:`numpy.ndarray`
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
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_thiem_2d([1,2,3], 10, 0.001, 1, 10, -0.001)
    array([-0.53084596, -0.35363029, -0.25419375])
    """
    rad = np.array(rad, dtype=float)
    # check the input
    if r_ref <= 0.0:
        raise ValueError(
            "The upper boundary needs to be greater than the wellradius"
        )
    if np.any(rad <= 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if trans_gmean <= 0.0:
        raise ValueError("The Transmissivity needs to be positive.")
    if T_well is not None and T_well <= 0.0:
        raise ValueError("The well Transmissivity needs to be positive.")
    if var <= 0.0:
        raise ValueError("The variance needs to be positive.")
    if len_scale <= 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if prop <= 0.0:
        raise ValueError("The proportionalityfactor needs to be positive.")

    # define some substitions to shorten the result
    chi = -var / 2.0 if T_well is None else np.log(T_well / trans_gmean)
    C = (prop / len_scale) ** 2
    # derive the result
    res = -expi(-chi / (1.0 + C * rad ** 2))
    res -= np.exp(-chi) * exp1(chi / (1.0 + C * rad ** 2) - chi)
    res += expi(-chi / (1.0 + C * r_ref ** 2))
    res += np.exp(-chi) * exp1(chi / (1.0 + C * r_ref ** 2) - chi)
    res *= -rate / (4.0 * np.pi * trans_gmean)
    res += h_ref
    return res


###############################################################################
# 3D version of extended Thiem
###############################################################################


def ext_thiem_3d(
    rad,
    r_ref,
    cond_gmean,
    var,
    len_scale,
    anis=1.0,
    lat_ext=1.0,
    rate=-1e-4,
    h_ref=0.0,
    K_well="KH",
    prop=1.6,
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
    r_ref : :class:`float`
        Reference radius with known head (see `h_ref`)
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
    var : :class:`float`
        Variance of the log-conductivity.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    anis : :class:`float`, optional
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
        Default: 1.0
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer (thickness). Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_ref : :class:`float`, optional
        Reference head at the reference-radius `r_ref`. Default: ``0.0``
    K_well : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    head : :class:`numpy.ndarray`
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
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_thiem_3d([1,2,3], 10, 0.001, 1, 10, 1, 1, -0.001)
    array([-0.48828026, -0.31472059, -0.22043022])
    """
    rad = np.array(rad, dtype=float)
    # check the input
    if not r_ref > 0.0:
        raise ValueError("The reference radius needs to be positive.")
    if not np.min(rad) > 0.0:
        raise ValueError("The given radii need to be positive.")
    if K_well != "KA" and K_well != "KH" and not isinstance(K_well, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(K_well, float) and K_well <= 0.0:
        raise ValueError("The well-conductivity needs to be positive.")
    if not cond_gmean > 0.0:
        raise ValueError("The gmean conductivity needs to be positive.")
    if var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if not lat_ext > 0.0:
        raise ValueError("The aquifer-thickness needs to be positive.")
    if not 0.0 < anis <= 1.0:
        raise ValueError("The anisotropy-ratio must be > 0 and <= 1")
    if not prop > 0.0:
        raise ValueError("The proportionalityfactor needs to be positive.")

    # define some substitions to shorten the result
    K_efu = cond_gmean * np.exp(var * (0.5 - aniso(anis)))
    if K_well == "KH":
        chi = var * (aniso(anis) - 1.0)
    elif K_well == "KA":
        chi = var * aniso(anis)
    else:
        chi = np.log(K_well / K_efu)

    C = (prop / len_scale / anis ** (1.0 / 3.0)) ** 2

    sub11 = np.sqrt(1.0 + C * r_ref ** 2)
    sub12 = np.sqrt(1.0 + C * rad ** 2)

    sub21 = np.log(sub12 + 1.0) - np.log(sub11 + 1.0)
    sub21 -= 1.0 / sub12 - 1.0 / sub11

    sub22 = np.log(sub12) - np.log(sub11)
    sub22 -= 0.50 / sub12 ** 2 - 0.50 / sub11 ** 2
    sub22 -= 0.25 / sub12 ** 4 - 0.25 / sub11 ** 4

    # derive the result
    res = np.exp(-chi) * (np.log(rad) - np.log(r_ref))
    res += sub21 * np.sinh(chi) + sub22 * (1.0 - np.cosh(chi))
    res *= -rate / (2.0 * np.pi * K_efu * lat_ext)
    res += h_ref

    return res


###############################################################################
# 2D version of extended Theis
###############################################################################


def ext_theis_2d(
    time,
    rad,
    storage,
    trans_gmean,
    var,
    len_scale,
    rate=-1e-4,
    r_well=0.0,
    r_bound=np.inf,
    h_bound=0.0,
    T_well=None,
    prop=1.6,
    struc_grid=True,
    far_err=0.01,
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
    storage : :class:`float`
        Storage of the aquifer.
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary as well as initial condition.
        Default: ``0.0``
    T_well : :class:`float`, optional
        Explicit transmissivity value at the well. Harmonic mean by default.
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    far_err : :class:`float`, optional
        Relative error for the farfield transmissivity for calculating the
        cutoff-point of the solution. Default: ``0.01``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissivity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_theis_2d([10,100], [1,2,3], 0.001, 0.001, 1, 10, -0.001)
    array([[-0.33737576, -0.17400123, -0.09489812],
           [-0.58443489, -0.40847176, -0.31095166]])
    """
    lap_kwargs = {} if lap_kwargs is None else lap_kwargs
    # check the input
    if r_well < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if not r_bound > r_well:
        raise ValueError("The upper boundary needs to be > well radius")
    if not storage > 0.0:
        raise ValueError("The Storage needs to be positive.")
    if not trans_gmean > 0.0:
        raise ValueError("The Transmissivity needs to be positive.")
    if var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if T_well is not None and not T_well > 0.0:
        raise ValueError("The well Transmissivity needs to be positive.")
    if not prop > 0.0:
        raise ValueError("The proportionality factor needs to be positive.")
    if parts <= 1:
        raise ValueError("The numbor of partitions needs to be at least 2")
    if not 0.0 < far_err < 1.0:
        raise ValueError(
            "The relative error of Transmissivity needs to be within (0,1)"
        )
    # genearte rlast from a given relativ-error to farfield-transmissivity
    r_last = T_CG_error(far_err, trans_gmean, var, len_scale, T_well, prop)
    # generate the partition points
    if r_last > r_well:
        R_part = specialrange_cut(r_well, r_bound, parts + 1, r_last)
    else:
        R_part = np.array([r_well, r_bound])
    # calculate the harmonic mean transmissivity values within each partition
    T_part = annular_hmean(
        T_CG,
        R_part,
        trans_gmean=trans_gmean,
        var=var,
        len_scale=len_scale,
        T_well=T_well,
        prop=prop,
    )
    T_well = T_CG(r_well, trans_gmean, var, len_scale, T_well, prop)
    return ext_grf(
        time=time,
        rad=rad,
        S_part=np.full_like(T_part, storage),
        K_part=T_part,
        R_part=R_part,
        dim=2,
        lat_ext=1,
        rate=rate,
        h_bound=h_bound,
        K_well=T_well,
        struc_grid=struc_grid,
        lap_kwargs=lap_kwargs,
    )


def ext_theis_3d(
    time,
    rad,
    storage,
    cond_gmean,
    var,
    len_scale,
    anis=1.0,
    lat_ext=1.0,
    rate=-1e-4,
    r_well=0.0,
    r_bound=np.inf,
    h_bound=0.0,
    K_well="KH",
    prop=1.6,
    far_err=0.01,
    struc_grid=True,
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
    storage : :class:`float`
        Storage of the aquifer.
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
    var : :class:`float`
        Variance of the log-conductivity.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    anis : :class:`float`, optional
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
        Default: 1.0
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer (thickness). Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary as well as initial condition.
        Default: ``0.0``
    K_well : :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    far_err : :class:`float`, optional
        Relative error for the farfield transmissivity for calculating the
        cutoff-point of the solution. Default: ``0.01``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissivity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> ext_theis_3d([10,100], [1,2,3], 0.001, 0.001, 1, 10, 1, 1, -0.001)
    array([[-0.32756786, -0.16717569, -0.09141211],
           [-0.5416396 , -0.36982684, -0.27798614]])
    """
    # check the input
    if r_well < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if not r_bound > r_well:
        raise ValueError("The upper boundary needs to be > well radius")
    if not storage > 0.0:
        raise ValueError("The storage needs to be positive.")
    if not cond_gmean > 0.0:
        raise ValueError("The gmean conductivity needs to be positive.")
    if var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if K_well != "KA" and K_well != "KH" and not isinstance(K_well, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(K_well, float) and not K_well > 0.0:
        raise ValueError("The well-conductivity needs to be positive.")
    if not cond_gmean > 0.0:
        raise ValueError("The conductivity needs to be positive.")
    if not prop > 0.0:
        raise ValueError("The proportionality factor needs to be positive.")
    if parts <= 1:
        raise ValueError("The numbor of partitions needs to be at least 2")
    if not 0.0 < far_err < 1.0:
        raise ValueError(
            "The relative error of Conductivity needs to be within (0,1)"
        )
    # genearte rlast from a given relativ-error to farfield-conductivity
    r_last = K_CG_error(
        far_err, cond_gmean, var, len_scale, anis, K_well, prop
    )
    # generate the partition points
    if r_last > r_well:
        R_part = specialrange_cut(r_well, r_bound, parts + 1, r_last)
    else:
        R_part = np.array([r_well, r_bound])
    # calculate the harmonic mean conductivity values within each partition
    K_part = annular_hmean(
        K_CG,
        R_part,
        cond_gmean=cond_gmean,
        var=var,
        len_scale=len_scale,
        anis=anis,
        K_well=K_well,
        prop=prop,
    )
    K_well = K_CG(r_well, cond_gmean, var, len_scale, anis, K_well, prop)
    return ext_grf(
        time=time,
        rad=rad,
        S_part=np.full_like(K_part, storage),
        K_part=K_part,
        R_part=R_part,
        dim=2,
        lat_ext=lat_ext,
        rate=rate,
        h_bound=h_bound,
        K_well=K_well,
        struc_grid=struc_grid,
        lap_kwargs=lap_kwargs,
    )


###############################################################################
# TPL version of extended Theis
###############################################################################


def ext_theis_tpl(
    time,
    rad,
    storage,
    cond_gmean,
    len_scale,
    hurst,
    var=None,
    c=1.0,
    dim=2.0,
    lat_ext=1.0,
    rate=-1e-4,
    r_well=0.0,
    r_bound=np.inf,
    h_bound=0.0,
    K_well="KH",
    prop=1.6,
    far_err=0.01,
    struc_grid=True,
    parts=30,
    lap_kwargs=None,
):
    """
    The extended Theis solution for truncated power-law fields.

    The extended Theis solution for transient flow under
    a pumping condition in a confined aquifer.
    The type curve is describing the effective drawdown
    in a d-dimensional statistical framework,
    where the conductivity distribution is
    following a log-normal distribution with a truncated power-law
    correlation function build on superposition of gaussian modes.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    storage : :class:`float`
        Storage of the aquifer.
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
        You can also treat this as transmissivity by leaving 'lat_ext=1'.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    hurst: :class:`float`
        Hurst coefficient of the TPL model. Should be in (0, 1).
    var : :class:`float`
        Variance of the log-conductivity.
        If var is given, c will be calculated accordingly.
        Default: :any:`None`
    c : :class:`float`, optional
        Intensity of variation in the TPL model.
        Is overwritten if var is given.
        Default: ``1.0``
    dim: :class:`float`, optional
        Dimension of space.
        Default: ``2.0``
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer:

            * sqare-root of cross-section in 1D
            * thickness in 2D
            * meaningless in 3D

        Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary as well as initial condition.
        Default: ``0.0``
    K_well : :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    far_err : :class:`float`, optional
        Relative error for the farfield transmissivity for calculating the
        cutoff-point of the solution. Default: ``0.01``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissivity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``
    """
    # check the input
    if r_well < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if not r_bound > r_well:
        raise ValueError("The upper boundary needs to be > well radius")
    if not storage > 0.0:
        raise ValueError("The storage needs to be positive.")
    if not cond_gmean > 0.0:
        raise ValueError("The gmean conductivity needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if not 0 < hurst < 1:
        raise ValueError("Hurst coefficient needs to be in (0,1)")
    if var is not None and var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if var is None and not c > 0.0:
        raise ValueError("The intensity of variation needs to be positive.")
    if K_well != "KA" and K_well != "KH" and not isinstance(K_well, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(K_well, float) and not K_well > 0.0:
        raise ValueError("The well-conductivity needs to be positive.")
    if not prop > 0.0:
        raise ValueError("The proportionality factor needs to be positive.")
    if parts <= 1:
        raise ValueError("The numbor of partitions needs to be at least 2")
    if not 0.0 < far_err < 1.0:
        raise ValueError(
            "The relative error of Conductivity needs to be within (0,1)"
        )
    # genearte rlast from a given relativ-error to farfield-conductivity
    r_last = TPL_CG_error(
        far_err, cond_gmean, len_scale, hurst, var, c, 1, dim, K_well, prop
    )
    # generate the partition points
    if r_last > r_well:
        R_part = specialrange_cut(r_well, r_bound, parts + 1, r_last)
    else:
        R_part = np.array([r_well, r_bound])
    # calculate the harmonic mean conductivity values within each partition
    K_part = annular_hmean(
        TPL_CG,
        R_part,
        ann_dim=dim,
        cond_gmean=cond_gmean,
        len_scale=len_scale,
        hurst=hurst,
        var=var,
        c=c,
        anis=1,
        dim=dim,
        K_well=K_well,
        prop=prop,
    )
    K_well = TPL_CG(
        r_well, cond_gmean, len_scale, hurst, var, c, 1, dim, K_well, prop
    )
    return ext_grf(
        time=time,
        rad=rad,
        S_part=np.full_like(K_part, storage),
        K_part=K_part,
        R_part=R_part,
        dim=dim,
        lat_ext=lat_ext,
        rate=rate,
        h_bound=h_bound,
        K_well=K_well,
        struc_grid=struc_grid,
        lap_kwargs=lap_kwargs,
    )


def ext_theis_tpl_3d(
    time,
    rad,
    storage,
    cond_gmean,
    len_scale,
    hurst,
    var=None,
    c=1.0,
    anis=1,
    lat_ext=1.0,
    rate=-1e-4,
    r_well=0.0,
    r_bound=np.inf,
    h_bound=0.0,
    K_well="KH",
    prop=1.6,
    far_err=0.01,
    struc_grid=True,
    parts=30,
    lap_kwargs=None,
):
    """
    The extended Theis solution for truncated power-law fields in 3D.

    The extended Theis solution for transient flow under
    a pumping condition in a confined aquifer with anisotropy in 3D.
    The type curve is describing the effective drawdown
    in a 3-dimensional statistical framework,
    where the conductivity distribution is
    following a log-normal distribution with a truncated power-law
    correlation function build on superposition of gaussian modes.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    storage : :class:`float`
        Storage of the aquifer.
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    hurst: :class:`float`
        Hurst coefficient of the TPL model. Should be in (0, 1).
    var : :class:`float`
        Variance of the log-conductivity.
        If var is given, c will be calculated accordingly.
        Default: :any:`None`
    c : :class:`float`, optional
        Intensity of variation in the TPL model.
        Is overwritten if var is given.
        Default: ``1.0``
    anis : :class:`float`, optional
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
        Default: 1.0
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer (thickness).
        Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary as well as initial condition.
        Default: ``0.0``
    K_well : :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    far_err : :class:`float`, optional
        Relative error for the farfield transmissivity for calculating the
        cutoff-point of the solution. Default: ``0.01``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissivity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``
    """
    # check the input
    if r_well < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if not r_bound > r_well:
        raise ValueError("The upper boundary needs to be > well radius")
    if not storage > 0.0:
        raise ValueError("The storage needs to be positive.")
    if not cond_gmean > 0.0:
        raise ValueError("The gmean conductivity needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if not 0 < hurst < 1:
        raise ValueError("Hurst coefficient needs to be in (0,1)")
    if var is not None and var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if var is None and not c > 0.0:
        raise ValueError("The intensity of variation needs to be positive.")
    if K_well != "KA" and K_well != "KH" and not isinstance(K_well, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(K_well, float) and not K_well > 0.0:
        raise ValueError("The well-conductivity needs to be positive.")
    if not prop > 0.0:
        raise ValueError("The proportionality factor needs to be positive.")
    if parts <= 1:
        raise ValueError("The numbor of partitions needs to be at least 2")
    if not 0.0 < far_err < 1.0:
        raise ValueError(
            "The relative error of Conductivity needs to be within (0,1)"
        )
    # genearte rlast from a given relativ-error to farfield-conductivity
    r_last = TPL_CG_error(
        far_err, cond_gmean, len_scale, hurst, var, c, anis, 3, K_well, prop
    )
    # generate the partition points
    if r_last > r_well:
        R_part = specialrange_cut(r_well, r_bound, parts + 1, r_last)
    else:
        R_part = np.array([r_well, r_bound])
    # calculate the harmonic mean conductivity values within each partition
    K_part = annular_hmean(
        TPL_CG,
        R_part,
        ann_dim=2,
        cond_gmean=cond_gmean,
        len_scale=len_scale,
        hurst=hurst,
        var=var,
        c=c,
        anis=anis,
        dim=3,
        K_well=K_well,
        prop=prop,
    )
    K_well = TPL_CG(
        r_well, cond_gmean, len_scale, hurst, var, c, anis, 3, K_well, prop
    )
    return ext_grf(
        time=time,
        rad=rad,
        S_part=np.full_like(K_part, storage),
        K_part=K_part,
        R_part=R_part,
        dim=2,
        lat_ext=lat_ext,
        rate=rate,
        h_bound=h_bound,
        K_well=K_well,
        struc_grid=struc_grid,
        lap_kwargs=lap_kwargs,
    )


def ext_thiem_tpl(
    rad,
    r_ref,
    cond_gmean,
    len_scale,
    hurst,
    var=None,
    c=1.0,
    dim=2.0,
    lat_ext=1.0,
    rate=-1e-4,
    h_ref=0.0,
    K_well="KH",
    prop=1.6,
):
    """
    The extended Thiem solution for truncated power-law fields.

    The extended Theis solution for steady flow under
    a pumping condition in a confined aquifer.
    The type curve is describing the effective drawdown
    in a d-dimensional statistical framework,
    where the conductivity distribution is
    following a log-normal distribution with a truncated power-law
    correlation function build on superposition of gaussian modes.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    r_ref : :class:`float`
        Reference radius with known head (see `h_ref`)
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
        You can also treat this as transmissivity by leaving 'lat_ext=1'.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    hurst: :class:`float`
        Hurst coefficient of the TPL model. Should be in (0, 1).
    var : :class:`float`
        Variance of the log-conductivity.
        If var is given, c will be calculated accordingly.
        Default: :any:`None`
    c : :class:`float`, optional
        Intensity of variation in the TPL model.
        Is overwritten if var is given.
        Default: ``1.0``
    dim: :class:`float`, optional
        Dimension of space.
        Default: ``2.0``
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer:

            * sqare-root of cross-section in 1D
            * thickness in 2D
            * meaningless in 3D

        Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_ref : :class:`float`, optional
        Reference head at the reference-radius `r_ref`. Default: ``0.0``
    K_well : :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``
    """
    # check the input
    if not cond_gmean > 0.0:
        raise ValueError("The gmean conductivity needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if not 0 < hurst < 1:
        raise ValueError("Hurst coefficient needs to be in (0,1)")
    if var is not None and var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if var is None and not c > 0.0:
        raise ValueError("The intensity of variation needs to be positive.")
    if K_well != "KA" and K_well != "KH" and not isinstance(K_well, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(K_well, float) and not K_well > 0.0:
        raise ValueError("The well-conductivity needs to be positive.")
    if not prop > 0.0:
        raise ValueError("The proportionality factor needs to be positive.")
    cond = ft.partial(
        TPL_CG,
        cond_gmean=cond_gmean,
        len_scale=len_scale,
        hurst=hurst,
        var=var,
        c=c,
        anis=1,
        dim=dim,
        K_well=K_well,
        prop=prop,
    )
    return ext_grf_steady(
        rad=rad,
        r_ref=r_ref,
        conductivity=cond,
        dim=dim,
        lat_ext=lat_ext,
        rate=rate,
        h_ref=h_ref,
    )


def ext_thiem_tpl_3d(
    rad,
    r_ref,
    cond_gmean,
    len_scale,
    hurst,
    var=None,
    c=1.0,
    anis=1,
    lat_ext=1.0,
    rate=-1e-4,
    h_ref=0.0,
    K_well="KH",
    prop=1.6,
):
    """
    The extended Theis solution for truncated power-law fields in 3D.

    The extended Theis solution for transient flow under
    a pumping condition in a confined aquifer with anisotropy in 3D.
    The type curve is describing the effective drawdown
    in a 3-dimensional statistical framework,
    where the conductivity distribution is
    following a log-normal distribution with a truncated power-law
    correlation function build on superposition of gaussian modes.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    storage : :class:`float`
        Storage of the aquifer.
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    hurst: :class:`float`
        Hurst coefficient of the TPL model. Should be in (0, 1).
    var : :class:`float`
        Variance of the log-conductivity.
        If var is given, c will be calculated accordingly.
        Default: :any:`None`
    c : :class:`float`, optional
        Intensity of variation in the TPL model.
        Is overwritten if var is given.
        Default: ``1.0``
    anis : :class:`float`, optional
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
        Default: 1.0
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer (thickness).
        Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary as well as initial condition.
        Default: ``0.0``
    K_well : :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    far_err : :class:`float`, optional
        Relative error for the farfield transmissivity for calculating the
        cutoff-point of the solution. Default: ``0.01``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissivity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    head : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``
    """
    # check the input
    if not cond_gmean > 0.0:
        raise ValueError("The gmean conductivity needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if not 0 < hurst < 1:
        raise ValueError("Hurst coefficient needs to be in (0,1)")
    if var is not None and var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if var is None and not c > 0.0:
        raise ValueError("The intensity of variation needs to be positive.")
    if K_well != "KA" and K_well != "KH" and not isinstance(K_well, float):
        raise ValueError(
            "The well-conductivity should be given as float or 'KA' resp 'KH'"
        )
    if isinstance(K_well, float) and not K_well > 0.0:
        raise ValueError("The well-conductivity needs to be positive.")
    if not prop > 0.0:
        raise ValueError("The proportionality factor needs to be positive.")
    cond = ft.partial(
        TPL_CG,
        cond_gmean=cond_gmean,
        len_scale=len_scale,
        hurst=hurst,
        var=var,
        c=c,
        anis=anis,
        dim=3,
        K_well=K_well,
        prop=prop,
    )
    return ext_grf_steady(
        rad=rad,
        r_ref=r_ref,
        conductivity=cond,
        dim=2,
        lat_ext=lat_ext,
        rate=rate,
        h_ref=h_ref,
    )


###############################################################################
# solution for apparent transmissivity from Neuman 2004
###############################################################################


def neuman2004(
    time,
    rad,
    storage,
    trans_gmean,
    var,
    len_scale,
    rate=-1e-4,
    r_well=0.0,
    r_bound=np.inf,
    h_bound=0.0,
    struc_grid=True,
    parts=30,
    lap_kwargs=None,
):
    """
    The transient solution for the apparent transmissivity from [Neuman2004].

    This solution is build on the apparent transmissivity from Neuman 2004,
    which represents a mean drawdown in an ensemble of pumping tests in
    heterogeneous transmissivity fields following an exponential covariance.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated.
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated.
    storage : :class:`float`
        Storage of the aquifer.
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    r_well : :class:`float`, optional
        Radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    h_bound : :class:`float`, optional
        Reference head at the outer boundary as well as initial condition.
        Default: ``0.0``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    parts : :class:`int`, optional
        Since the solution is calculated by setting the transmissivity to local
        constant values, one needs to specify the number of partitions of the
        transmissivity. Default: ``30``
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
    .. [Neuman2004] Neuman, Shlomo P., Alberto Guadagnini, and Monica Riva.
       ''Type-curve estimation of statistical heterogeneity.''
       Water resources research 40.4, 2004
    """
    # check the input
    if r_well < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if not r_bound > r_well:
        raise ValueError("The upper boundary needs to be > well radius")
    if not storage > 0.0:
        raise ValueError("The Storage needs to be positive.")
    if not trans_gmean > 0.0:
        raise ValueError("The Transmissivity needs to be positive.")
    if var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")
    if parts <= 1:
        raise ValueError("The numbor of partitions needs to be at least 2")
    # genearte rlast from a given relativ-error to farfield-transmissivity
    r_last = 2 * len_scale
    # generate the partition points
    if r_last > r_well:
        R_part = specialrange_cut(r_well, r_bound, parts + 1, r_last)
    else:
        R_part = np.array([r_well, r_bound])
    # calculate the harmonic mean transmissivity values within each partition
    T_part = annular_hmean(
        neuman2004_trans,
        R_part,
        trans_gmean=trans_gmean,
        var=var,
        len_scale=len_scale,
    )
    T_well = neuman2004_trans(r_well, trans_gmean, var, len_scale)
    return ext_grf(
        time=time,
        rad=rad,
        S_part=np.full_like(T_part, storage),
        K_part=T_part,
        R_part=R_part,
        dim=2,
        lat_ext=1,
        rate=rate,
        h_bound=h_bound,
        K_well=T_well,
        struc_grid=struc_grid,
        lap_kwargs=lap_kwargs,
    )


def neuman2004_steady(
    rad, r_ref, trans_gmean, var, len_scale, rate=-1e-4, h_ref=0.0
):
    """
    The steady solution for the apparent transmissivity from [Neuman2004].

    This solution is build on the apparent transmissivity from Neuman 1994,
    which represents a mean drawdown in an ensemble of pumping tests in
    heterogeneous transmissivity fields following an exponential covariance.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    r_ref : :class:`float`
        Radius of the reference head.
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
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
    .. [Neuman2004] Neuman, Shlomo P., Alberto Guadagnini, and Monica Riva.
       ''Type-curve estimation of statistical heterogeneity.''
       Water resources research 40.4, 2004
    """
    # check the input
    if not trans_gmean > 0.0:
        raise ValueError("The Transmissivity needs to be positive.")
    if var < 0.0:
        raise ValueError("The variance needs to be positive.")
    if not len_scale > 0.0:
        raise ValueError("The correlationlength needs to be positive.")

    return ext_grf_steady(
        rad=rad,
        r_ref=r_ref,
        conductivity=neuman2004_trans,
        dim=2,
        lat_ext=1,
        rate=rate,
        h_ref=h_ref,
        trans_gmean=trans_gmean,
        var=var,
        len_scale=len_scale,
    )


if __name__ == "__main__":
    import doctest

    doctest.testmod()
