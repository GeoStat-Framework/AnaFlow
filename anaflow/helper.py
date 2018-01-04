#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
========================================
Helper functions (:mod:`anaflow.helper`)
========================================

.. currentmodule:: anaflow.helper

Anaflow subpackage providing several helper functions.

Functions
---------
The following functions are provided

.. autosummary::
   :toctree: generated/

   radii - Distribution of diskseparations
   specialrange - several special range functions
   T_CG - coarse graining transmissivity
   T_CG_inverse - inverse coarse graining transmissivity
   T_CG_error - error calculation for the coarse graining transmissivity
   K_CG - coarse graining conductivity
   K_CG_inverse - inverse coarse graining conductivity
   K_CG_error - error calculation for the coarse graining conductivity
   aniso - the anisotropy function by dagan
   well_solution - The classic Theis well function

"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.special import exp1

__all__ = ["radii", "specialrange",
           "T_CG", "T_CG_inverse", "T_CG_error",
           "K_CG", "K_CG_inverse", "K_CG_error",
           "aniso", "well_solution"]


def radii(parts, rwell=0.0, rinf=np.inf, rlast=500.0, typ="log"):
    '''
    Calculation of specific point distributions for the diskmodel.

    Parameters
    ----------
    parts : int
        Number of partitions.
    rwell : float, optional
        Radius at the well. Default: 0.0
    rinf : float, optional
        Radius at the outer boundary. Default: inf
    rlast : float, optional
        Setting the last radius befor the outer boundary. Default: 500.0
    typ : string, optional
        Setting the distribution type of the radii. Default: "log"

    Returns
    -------
    p_rad : ndarray
        Array containing the separating radii
    f_rad : ndarray
        Array containing the function-evaluation points within each disk

    Example
    -------
    >>> radii(2)
    (array([   0.,  500.,   inf]), array([  0.,  inf]))
    '''

    if typ == "log":
        if rinf == np.inf:
            # define the partition radii
            p_rad = np.expm1(np.linspace(np.log1p(rwell),
                                         np.log1p(rlast), parts))
            p_rad = np.append(p_rad, [np.inf])

            # define the points within the partitions to evaluate the function
            f_rad = np.expm1(np.linspace(np.log1p(rwell),
                                         np.log1p(rlast), parts-1))
            f_rad = np.append(f_rad, [np.inf])
        else:
            p_rad = np.expm1(np.linspace(np.log1p(rwell),
                                         np.log1p(rinf), parts+1))
            f_rad = np.expm1(np.linspace(np.log1p(rwell),
                                         np.log1p(rinf), parts))

    else:
        if rinf == np.inf:
            # define the partition radii
            p_rad = np.linspace(rwell, rlast, parts)
            p_rad = np.append(p_rad, [np.inf])

            # define the points within the partitions to evaluate the function
            f_rad = np.linspace(rwell, rlast, parts-1)
            f_rad = np.append(f_rad, [np.inf])
        else:
            p_rad = np.linspace(rwell, rinf, parts+1)
            f_rad = np.linspace(rwell, rinf, parts)

    return p_rad, f_rad


def specialrange(val_min, val_max, steps, typ="log"):
    '''
    Calculation of special point ranges.

    Parameters
    ----------
    val_min : float
        Starting value.
    val_max : float
        Ending value
    steps : int
        Number of steps.
    typ : string, optional
        Setting the kind of range-distribution. One can choose between
        "log" for logarithmic behavior,
        "lin" for linear behavior,
        "quad" for quadratic behavior,
        "cub" for cubic behavior,
        Default: "log"

    Returns
    -------
    rng : ndarray
        Array containing the special range

    Example
    -------
    >>> specialrange(1,10,4)
    array([  1.        ,   2.53034834,   5.23167968,  10.        ])
    '''

    if typ in ["logarithmic", "log"]:
        rng = np.expm1(np.linspace(np.log1p(val_min),
                                   np.log1p(val_max), steps))
    elif typ in ["linear", "lin"]:
        rng = np.linspace(val_min, val_max, steps)
    elif typ in ["quadratic", "quad"]:
        rng = (np.linspace(np.sqrt(val_min), np.sqrt(val_max), steps))**2
    elif typ in ["cubic", "cub"]:
        rng = (np.linspace(np.power(val_min, 1/3.),
                           np.power(val_max, 1/3.), steps))**3
    else:
        rng = np.linspace(val_min, val_max, steps)

    return rng


def T_CG(rad, TG, sig2, corr, prop, Twell=None):
    '''
    The coarse-graining Transmissivity. This functions gives an effective
    transmissivity for 2D pumpingtests in heterogenous aquifers, where the
    transmissivity is following a log-normal distribution and a gaussian
    correlation function.

    Parameters
    ----------
    rad : ndarray
        Array with all radii where the function should be evaluated
    TG : float
        Geometric-mean of the transmissivity-distribution
    sig2 : float
        log-normal-variance of the transmissivity-distribution
    corr : float
        corralation-length of transmissivity-distribution
    prop: float, optional
        Proportionality factor used within the upscaling procedure.
        Default: 1.6
    Twell : float, optional
        Explicit transmissivity value at the well. Default: None

    Returns
    -------
    T_CG : ndarray
        Array containing the effective transmissivity values.

    Example
    -------
    >>> T_CG([1,2,3], 0.001, 1, 10, 2)
    array([ 0.00061831,  0.00064984,  0.00069236])
    '''

    rad = np.squeeze(rad)

    if Twell is not None:
        chi = np.log(Twell) - np.log(TG)
    else:
        chi = -sig2/2.0

    return TG*np.exp(chi/(1.0+(prop*rad/corr)**2))


def T_CG_inverse(T, TG, sig2, corr, prop, Twell=None):
    '''
    The inverse function of the coarse-graining Transmissivity. See: "T_CG"

    Parameters
    ----------
    T : ndarray
        Array with all transmissivity values
        where the function should be evaluated
    TG : float
        Geometric-mean of the transmissivity-distribution
    sig2 : float
        log-normal-variance of the transmissivity-distribution
    corr : float
        corralation-length of transmissivity-distribution
    prop: float, optional
        Proportionality factor used within the upscaling procedure.
        Default: 1.6
    Twell : float, optional
        Explicit transmissivity value at the well. Default: None

    Returns
    -------
    rad : ndarray
        Array containing the radii belonging to the given transmissivity values

    Example
    -------
    >>> T_CG_inverse([7e-4,8e-4,9e-4], 0.001, 1, 10, 2)
    array([ 3.16952925,  5.56935826,  9.67679026])
    '''

    T = np.squeeze(T)

    if Twell is not None:
        chi = np.log(Twell) - np.log(TG)
    else:
        chi = -sig2/2.0

    return (corr/prop)*np.sqrt(chi/np.log(T/TG) - 1.0)


def T_CG_error(err, TG, sig2, corr, prop, Twell=None):
    '''
    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance. See: "T_CG"

    Parameters
    ----------
    err : float
        Given relative error for the farfield transmissivity
    TG : float
        Geometric-mean of the transmissivity-distribution
    sig2 : float
        log-normal-variance of the transmissivity-distribution
    corr : float
        corralation-length of transmissivity-distribution
    prop: float, optional
        Proportionality factor used within the upscaling procedure.
        Default: 1.6
    Twell : float, optional
        Explicit transmissivity value at the well. Default: None

    Returns
    -------
    rad : float
        Radial point, where the relative error is less than the given one.

    Example
    -------
    >>> T_CG_error(0.01, 0.001, 1, 10, 2)
    34.910450167790387
    '''

    if Twell is not None:
        chi = np.log(Twell) - np.log(TG)
    else:
        chi = -sig2/2.0

    if chi > 0.0:
        if chi/np.log(1.+err) >= 1.0:
            return (corr/prop)*np.sqrt(chi/np.log(1.+err) - 1.0)
        # standard value if the error is less then the variation
        return 1.
    else:
        if chi/np.log(1.-err) >= 1.0:
            return (corr/prop)*np.sqrt(chi/np.log(1.-err) - 1.0)
        # standard value if the error is less then the variation
        return 1.


def K_CG(rad, KG, sig2, corr, e, prop, Kwell="KH"):
    '''
    The coarse-graining conductivity. This functions gives an effective
    conductivity for 3D pumpingtests in heterogenous aquifers, where the
    conductivity is following a log-normal distribution and a gaussian
    correlation function and taking vertical anisotropy into account.

    Parameters
    ----------
    rad : ndarray
        Array with all radii where the function should be evaluated
    KG : float
        Geometric-mean conductivity-distribution
    sig2 : float
        log-normal-variance of the conductivity-distribution
    corr : float
        corralation-length of conductivity-distribution
    e : float
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: float, optional
        Proportionality factor used within the upscaling procedure.
        Default: 1.6
    Kwell : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean ("KH"), the arithmetic mean ("KA") or an arbitrary float
        value. Default: "KH"

    Returns
    -------
    K_CG : ndarray
        Array containing the effective conductivity values.

    Example
    -------
    >>> K_CG([1,2,3], 0.001, 1, 10, 1, 2)
    array([ 0.00063008,  0.00069285,  0.00077595])
    '''

    rad = np.squeeze(rad)

    Kefu = KG*np.exp(sig2*(0.5 - aniso(e)))
    if Kwell == "KH":
        chi = sig2*(aniso(e)-1.)
    elif Kwell == "KA":
        chi = sig2*aniso(e)
    else:
        chi = np.log(Kwell) - np.log(Kefu)

    return Kefu*np.exp(chi/np.sqrt(1.0+(prop*rad/(corr*e**(1./3.)))**2)**3)


def K_CG_inverse(K, KG, sig2, corr, e, prop, Kwell="KH"):
    '''
    The inverse function of the coarse-graining conductivity. See: "K_CG"

    Parameters
    ----------
    K : ndarray
        Array with all conductivity values
        where the function should be evaluated
    KG : float
        Geometric-mean conductivity-distribution
    sig2 : float
        log-normal-variance of the conductivity-distribution
    corr : float
        corralation-length of conductivity-distribution
    e : float
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: float, optional
        Proportionality factor used within the upscaling procedure.
        Default: 1.6
    Kwell : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean ("KH"), the arithmetic mean ("KA") or an arbitrary float
        value. Default: "KH"

    Returns
    -------
    rad : ndarray
        Array containing the radii belonging to the given conductivity values

    Example
    -------
    >>> K_CG_inverse([7e-4,8e-4,9e-4], 0.001, 1, 10, 1, 2)
    array([ 2.09236867,  3.27914996,  4.52143956])
    '''

    K = np.squeeze(K)

    Kefu = KG*np.exp(sig2*(0.5 - aniso(e)))
    if Kwell == "KH":
        chi = sig2*(aniso(e)-1.)
    elif Kwell == "KA":
        chi = sig2*aniso(e)
    else:
        chi = np.log(Kwell) - np.log(Kefu)

    return corr*e**(1./3.)/prop*np.sqrt((chi/np.log(K/Kefu))**(2./3.) - 1.0)


def K_CG_error(err, KG, sig2, corr, e, prop, Kwell="KH"):
    '''
    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance. See: "K_CG"

    Parameters
    ----------
    err : float
        Given relative error for the farfield conductivity
    KG : float
        Geometric-mean conductivity-distribution
    sig2 : float
        log-normal-variance of the conductivity-distribution
    corr : float
        corralation-length of conductivity-distribution
    e : float
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: float, optional
        Proportionality factor used within the upscaling procedure.
        Default: 1.6
    Kwell : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean ("KH"), the arithmetic mean ("KA") or an arbitrary float
        value. Default: "KH"

    Returns
    -------
    rad : float
        Radial point, where the relative error is less than the given one.

    Example
    -------
    >>> K_CG_error(0.01, 0.001, 1, 10, 1, 2)
    19.612796453639845
    '''

    Kefu = KG*np.exp(sig2*(0.5 - aniso(e)))
    if Kwell == "KH":
        chi = sig2*(aniso(e)-1.)
    elif Kwell == "KA":
        chi = sig2*aniso(e)
    else:
        chi = np.log(Kwell) - np.log(Kefu)

    coef = corr*e**(1./3.)/prop

    if chi > 0.0:
        if chi/np.log(1.+err) >= 1.0:
            return coef*np.sqrt((chi/np.log(1.+err))**(2./3.)-1.0)
        # standard value if the error is less then the variation
        return 1.
    else:
        if chi/np.log(1.-err) >= 1.0:
            return coef*np.sqrt((chi/np.log(1.-err))**(2./3.)-1.0)
        # standard value if the error is less then the variation
        return 1.


def aniso(e):
    '''
    The anisotropy function, known from 'Dagan [1989]'.

    Parameters
    ----------
    e : float
        Anisotropy-ratio of the vertical and horizontal corralation-lengths

    Returns
    -------
    aniso : float
        Value of the anisotropy function for the given value.

    Example
    -------
    >>> aniso(0.5)
    0.23639985871871511
    '''

    if not (0.0 <= e <= 1.0):
        raise ValueError(
            "Anisotropieratio 'e' must be within 0 and 1")

    if e == 1.0:
        res = 1./3.
    elif e == 0.0:
        res = 0.0
    else:
        res = e/(2*(1.-e**2))
        res *= 1./np.sqrt(1.-e**2)*np.arctan(np.sqrt(1./e**2 - 1.)) - e

    return res


def well_solution(rad, time, T, S, Qw,
                  struc_grid=True, hinf=0.0):
    '''
    The classical Theis solution for transient flow under a pumping condition
    in a confined and homogeneous aquifer.

    Parameters
    ----------
    rad : ndarray
        Array with all radii where the function should be evaluated
    time : ndarray
        Array with all time-points where the function should be evaluated
    T : float
        Given transmissivity of the aquifer
    S : float
        Given storativity of the aquifer
    Qw : float
        Pumpingrate at the well
    struc_grid : bool, optional
        If this is set to "False", the "rad" and "time" array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: True
    hinf : float, optional
        Reference head at the outer boundary "rinf". Default: 0.0

    Returns
    -------
    well_solution : ndarray
        Array with all heads at the given radii and time-points.

    Notes
    -----
    The parameters "rad", "T" and "S" will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    r = sqrt(x**2 + y**2)

    Example
    -------
    >>> well_solution([1,2,3], [10,100], 0.001, 0.001, -0.001)
    array([[-0.24959541, -0.14506368, -0.08971485],
           [-0.43105106, -0.32132823, -0.25778313]])
    '''

    rad = np.squeeze(rad)
    time = np.array(time).reshape(-1)

    if not struc_grid:
        grid_shape = rad.shape
        rad = rad.reshape(-1)

    if not (np.all(rad > 0.0)):
        raise ValueError(
            "The given radii need to be greater than the wellradius")
    if not (np.all(time > 0.0)):
        raise ValueError(
            "The given times need to be > 0")
    if not struc_grid and not (rad.shape == time.shape):
        raise ValueError(
            "For unstructured grid the number of time- & radii-pts must equal")
    if not (T > 0.0):
        raise ValueError(
            "The Transmissivity needs to be positiv")
    if not (S > 0.0):
        raise ValueError(
            "The Storage needs to be positiv")

    res = np.zeros(time.shape + rad.shape)

    for ti, te in np.ndenumerate(time):
        for ri, re in np.ndenumerate(rad):
            res[ti+ri] = Qw/(4.0*np.pi*T)*exp1(re**2*S/(4*T*te))

    if not struc_grid and len(grid_shape) > 0:
        res = np.copy(np.diag(res).reshape(grid_shape))

    # add the reference head
    res += hinf

    return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()
