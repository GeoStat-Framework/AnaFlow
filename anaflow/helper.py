#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing several helper functions.

.. currentmodule:: anaflow.helper

Functions
---------
The following functions are provided

.. autosummary::

   rad_amean_func
   rad_gmean_func
   rad_hmean_func
   rad_pmean_func
   radii
   specialrange
   specialrange_cut
   T_CG
   T_CG_inverse
   T_CG_error
   K_CG
   K_CG_inverse
   K_CG_error
   aniso
   well_solution
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.special import exp1
from scipy.integrate import quad as integ

__all__ = ["rad_amean_func",
           "rad_gmean_func",
           "rad_hmean_func",
           "rad_pmean_func",
           "radii", "specialrange", "specialrange_cut",
           "T_CG", "T_CG_inverse", "T_CG_error",
           "K_CG", "K_CG_inverse", "K_CG_error",
           "aniso", "well_solution"]


def rad_amean_func(func, val_arr, arg_dict=None, **kwargs):
    '''
    Calculating the arithmetic mean of a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\\frac{2}{r_{i+1}^2-r_i^2}
       \\intop^{r_{i+1}}_{r_i} r\\cdot f(r)\\, dr

    Parameters
    ----------
    func : :any:`callable`
        function that should be used
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        given radii defining the disks
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``rad_amean_func`` and
        ``func``.
        Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all calculated arithmetic means

    Raises
    ------
    ValueError
        If `func` is not callable.
    ValueError
        If ``val_arr`` has less than 2 values.
    ValueError
        If ``val_arr`` is not sorted in incresing order.

    Notes
    -----
    If the last value in val_arr is "inf", the given function should provide
    a value for "inf" as input: ``func(float("inf"))``

    Example
    -------
    >>> f = lambda x: x**2
    >>> rad_amean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    '''

    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError(
            "The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError(
            "To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i+1] for i in range(len(val_arr)-1)):
        raise ValueError(
            "The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """
        dummy function providing the integrand
        """
        def integrand(val):
            '''
            Integrand for the geometric mean ``r*log(f(r))``
            '''
            return 2*val*func(val, **kwargs)
        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i+1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(_step(func, kwargs),
                                val_arr[i], val_arr[i+1])[0]
            func_arr[i] = func_arr[i]/(val_arr[i+1]**2 - val_arr[i]**2)

    return func_arr


def rad_gmean_func(func, val_arr, arg_dict=None, **kwargs):
    '''
    Calculating the geometric mean of a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\\exp\\left(\\frac{2}{r_{i+1}^2-r_i^2}
       \\intop^{r_{i+1}}_{r_i} r\\cdot\\ln(f(r))\\, dr\\right)

    Parameters
    ----------
    func : :any:`callable`
        function that should be used
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        given radii defining the disks
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``rad_gmean_func`` and
        ``func``.
        Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all calculated geometric means

    Raises
    ------
    ValueError
        If `func` is not callable.
    ValueError
        If ``val_arr`` has less than 2 values.
    ValueError
        If ``val_arr`` is not sorted in incresing order.

    Notes
    -----
    If the last value in val_arr is "inf", the given function should provide
    a value for "inf" as input: ``func(float("inf"))``

    Example
    -------
    >>> f = lambda x: x**2
    >>> rad_gmean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    '''

    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError(
            "The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError(
            "To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i+1] for i in range(len(val_arr)-1)):
        raise ValueError(
            "The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """
        dummy function providing the integrand
        """
        def integrand(val):
            '''
            Integrand for the geometric mean ``r*log(f(r))``
            '''
            return 2*val*np.log(func(val, **kwargs))
        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i+1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(_step(func, kwargs),
                                val_arr[i], val_arr[i+1])[0]
            func_arr[i] = np.exp(func_arr[i]/(val_arr[i+1]**2 - val_arr[i]**2))

    return func_arr


def rad_hmean_func(func, val_arr, arg_dict=None, **kwargs):
    '''
    Calculating the harmonic mean of a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\\left(\\frac{2}{r_{i+1}^2-r_i^2}
       \\intop^{r_{i+1}}_{r_i} r\\cdot(f(r))^{-1}\\, dr\\right)^{-1}

    Parameters
    ----------
    func : :any:`callable`
        function that should be used
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        given radii defining the disks
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``rad_gmean_func`` and
        ``func``.
        Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all calculated geometric means

    Raises
    ------
    ValueError
        If `func` is not callable.
    ValueError
        If ``val_arr`` has less than 2 values.
    ValueError
        If ``val_arr`` is not sorted in incresing order.

    Notes
    -----
    If the last value in val_arr is "inf", the given function should provide
    a value for "inf" as input: ``func(float("inf"))``

    Example
    -------
    >>> f = lambda x: x**2
    >>> rad_hmean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    '''

    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError(
            "The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError(
            "To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i+1] for i in range(len(val_arr)-1)):
        raise ValueError(
            "The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """
        dummy function providing the integrand
        """
        def integrand(val):
            '''
            Integrand for the geometric mean ``r*log(f(r))``
            '''
            return 2*val/func(val, **kwargs)
        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i+1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(_step(func, kwargs),
                                val_arr[i], val_arr[i+1])[0]
            func_arr[i] = 1.0/(func_arr[i]/(val_arr[i+1]**2 - val_arr[i]**2))

    return func_arr


def rad_pmean_func(func, val_arr, p=1.0, arg_dict=None, **kwargs):
    '''
    Calculating the p-mean of a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\\left(\\frac{2}{r_{i+1}^2-r_i^2}
       \\intop^{r_{i+1}}_{r_i} r\\cdot(f(r))^p\\, dr\\right)^{\\frac{1}{p}}

    Parameters
    ----------
    func : :any:`callable`
        function that should be used
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        given radii defining the disks
    p : :class:`float`, optional
        The potency defining the p-mean.
        Default: ``1.0``
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``rad_pmean_func`` and
        ``func``.
        Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all calculated p-means

    Raises
    ------
    ValueError
        If `func` is not callable.
    ValueError
        If ``val_arr`` has less than 2 values.
    ValueError
        If ``val_arr`` is not sorted in incresing order.

    Notes
    -----
    If the last value in val_arr is "inf", the given function should provide
    a value for "inf" as input: ``func(float("inf"))``

    Example
    -------
    >>> f = lambda x: x**2
    >>> rad_pmean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    '''

    # if p is 0, the limit-case of the geometric mean is returned
    if p == 0:
        return rad_gmean_func(func, val_arr, arg_dict, **kwargs)

    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError(
            "The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError(
            "To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i+1] for i in range(len(val_arr)-1)):
        raise ValueError(
            "The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """
        dummy function providing the integrand
        """
        def integrand(val):
            '''
            Integrand for the geometric mean ``r*log(f(r))``
            '''
            return 2*val*func(val, **kwargs)**p
        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i+1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(_step(func, kwargs),
                                val_arr[i], val_arr[i+1])[0]
            func_arr[i] = (func_arr[i] /
                           (val_arr[i+1]**2 - val_arr[i]**2))**(1.0/p)

    return func_arr


def radii(parts, rwell=0.0, rinf=np.inf, rlast=500.0, typ="log"):
    '''
    Calculation of specific point distributions for the diskmodel.

    Parameters
    ----------
    parts : :class:`int`
        Number of partitions.
    rwell : :class:`float`, optional
        Radius at the well. Default: ``0.0``
    rinf : :class:`float`, optional
        Radius at the outer boundary. Default: :any:``np.inf``
    rlast : :class:`float`, optional
        Setting the last radius befor the outer boundary. Default: ``500.0``
    typ : :class:`str`, optional
        Setting the distribution type of the radii. Default: ``"log"``

    Returns
    -------
    p_rad : :class:`numpy.ndarray`
        Array containing the separating radii
    f_rad : :class:`numpy.ndarray`
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
    val_min : :class:`float`
        Starting value.
    val_max : :class:`float`
        Ending value
    steps : :class:`int`
        Number of steps.
    typ : :class:`str` or :class:`float`, optional
        Setting the kind of range-distribution. One can choose between

        * ``"log"``: for logarithmic behavior
        * ``"lin"``: for linear behavior
        * ``"quad"``: for quadratic behavior
        * ``"cub"``: for cubic behavior
        * :class:`float`: here you can specifi any exponent ("quad" would be
          equivalent to 2)

        Default: ``"log"``

    Returns
    -------
    :class:`numpy.ndarray`
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
    elif isinstance(typ, (float, int)):
        rng = (np.linspace(np.power(val_min, 1./typ),
                           np.power(val_max, 1./typ), steps))**typ
    else:
        rng = np.linspace(val_min, val_max, steps)

    return rng


def specialrange_cut(val_min, val_max, steps, val_cut=np.inf, typ="log"):
    '''
    Calculation of special point ranges.

    Parameters
    ----------
    val_min : :class:`float`
        Starting value.
    val_max : :class:`float`
        Ending value
    steps : :class:`int`
        Number of steps.
    val_cut : :class:`float`
        Cutting value, if val_max is bigger than this value, the last interval
        will be between val_cut and val_max
    typ : :class:`str` or :class:`float`, optional
        Setting the kind of range-distribution. One can choose between

        * ``"log"``: for logarithmic behavior
        * ``"lin"``: for linear behavior
        * ``"quad"``: for quadratic behavior
        * ``"cub"``: for cubic behavior
        * :class:`float`: here you can specifi any exponent ("quad" would be
          equivalent to 2)

        Default: ``"log"``

    Returns
    -------
    :class:`numpy.ndarray`
        Array containing the special range

    Example
    -------
    >>> specialrange_cut(1,10,4)
    array([  1.        ,   2.53034834,   5.23167968,  10.        ])
    '''

    if val_max > val_cut:
        rng = specialrange(val_min, val_cut, steps-1, typ)
        return np.hstack((rng, val_max))

    return specialrange(val_min, val_max, steps, typ)


def T_CG(rad, TG, sig2, corr, prop=1.6, Twell=None):
    '''
    The coarse-graining Transmissivity.

    This solution was presented in ''Schneider & Attinger 2008''[R3]_.

    This functions gives an effective
    transmissivity for 2D pumpingtests in heterogenous aquifers, where the
    transmissivity is following a log-normal distribution and a gaussian
    correlation function.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    TG : :class:`float`
        Geometric-mean of the transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``

    Returns
    -------
    T_CG : :class:`numpy.ndarray`
        Array containing the effective transmissivity values.

    References
    ----------
    .. [R3] Schneider, C. and Attinger, S.,
       ''Beyond thiem: A new method for interpreting large scale
       pumping tests in heterogeneous aquifers.''
       Water resources research, 44(4), 2008

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


def T_CG_inverse(T, TG, sig2, corr, prop=1.6, Twell=None):
    '''
    The inverse function of the coarse-graining Transmissivity.
    See: :func:`T_CG`

    Parameters
    ----------
    T : :class:`numpy.ndarray`
        Array with all transmissivity values
        where the function should be evaluated
    TG : :class:`float`
        Geometric-mean of the transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``

    Returns
    -------
    rad : :class:`numpy.ndarray`
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


def T_CG_error(err, TG, sig2, corr, prop=1.6, Twell=None):
    '''
    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance.
    See: :func:`T_CG`

    Parameters
    ----------
    err : :class:`float`
        Given relative error for the farfield transmissivity
    TG : :class:`float`
        Geometric-mean of the transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``

    Returns
    -------
    rad : :class:`float`
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


def K_CG(rad, KG, sig2, corr, e, prop=1.6, Kwell="KH"):
    '''
    The coarse-graining conductivity.

    This solution was presented in ''Zech 2013''[R8]_.

    This functions gives an effective
    conductivity for 3D pumpingtests in heterogenous aquifers, where the
    conductivity is following a log-normal distribution and a gaussian
    correlation function and taking vertical anisotropy into account.

    Parameters
    ----------
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
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Kwell :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``),
        the arithmetic mean (``"KA"``) or an arbitrary float
        value. Default: ``"KH"``

    Returns
    -------
    K_CG : :class:`numpy.ndarray`
        Array containing the effective conductivity values.

    References
    ----------
    .. [R8] Zech, A.
       ''Impact of Aqifer Heterogeneity on Subsurface Flow and Salt
       Transport at Different Scales: from a method determine parameters
       of heterogeneous permeability at local scale to a large-scale model
       for the sedimentary basin of Thuringia.''
       PhD thesis, Friedrich-Schiller-Universität Jena, 2013

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


def K_CG_inverse(K, KG, sig2, corr, e, prop=1.6, Kwell="KH"):
    '''
    The inverse function of the coarse-graining conductivity.
    See: :func:`K_CG`

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        Array with all conductivity values
        where the function should be evaluated
    KG : :class:`float`
        Geometric-mean conductivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the conductivity-distribution
    corr : :class:`float`
        corralation-length of conductivity-distribution
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Kwell :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean ("KH"), the arithmetic mean ("KA") or an arbitrary float
        value. Default: ``"KH"``

    Returns
    -------
    rad : :class:`numpy.ndarray`
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


def K_CG_error(err, KG, sig2, corr, e, prop=1.6, Kwell="KH"):
    '''
    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance.
    See: :func:`K_CG`

    Parameters
    ----------
    err : :class:`float`
        Given relative error for the farfield conductivity
    KG : :class:`float`
        Geometric-mean conductivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the conductivity-distribution
    corr : :class:`float`
        corralation-length of conductivity-distribution
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Kwell :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean ("KH"), the arithmetic mean ("KA") or an arbitrary float
        value. Default: ``"KH"``

    Returns
    -------
    rad : :class:`float`
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
    The anisotropy function.

    Known from ''Dagan (1989)''[R2]_.

    Parameters
    ----------
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths

    Returns
    -------
    aniso : :class:`float`
        Value of the anisotropy function for the given value.

    Raises
    ------
    ValueError
        If the Anisotropy-ratio ``e`` is not within 0 and 1.


    References
    ----------
    .. [R2] Dagan, G., ''Flow and Transport on Porous Formations'',
           Springer Verlag, New York, 1989.

    Example
    -------
    >>> aniso(0.5)
    0.23639985871871511
    '''

    if not 0.0 <= e <= 1.0:
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

    This solution was presented in ''Theis 1935''[R9]_.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    T : :class:`float`
        Given transmissivity of the aquifer
    S : :class:`float`
        Given storativity of the aquifer
    Qw : :class:`float`
        Pumpingrate at the well
    struc_grid : :class:`bool`, optional
        If this is set to "False", the "rad" and "time" array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    hinf : :class:`float`, optional
        Reference head at the outer boundary "rinf". Default: ``0.0``

    Returns
    -------
    well_solution : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Raises
    ------
    ValueError
        If ``rad`` is not positiv.
    ValueError
        If ``time`` is not positiv.
    ValueError
        If shape of ``rad`` and ``time`` differ in case of
        ``struc_grid`` is ``True``.
    ValueError
        If ``T`` is not positiv.
    ValueError
        If ``S`` is not positiv.

    References
    ----------
    .. [R9] Theis, C.,
       ''The relation between the lowering of the piezometric surface and the
       rate and duration of discharge of a well using groundwater storage'',
       Trans. Am. Geophys. Union, 16, 519–524, 1935

    Notes
    -----
    The parameters ``rad``, ``T`` and ``S`` will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

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

    if not np.all(rad > 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius")
    if not np.all(time > 0.0):
        raise ValueError(
            "The given times need to be > 0")
    if not struc_grid and not rad.shape == time.shape:
        raise ValueError(
            "For unstructured grid the number of time- & radii-pts must equal")
    if not T > 0.0:
        raise ValueError(
            "The Transmissivity needs to be positiv")
    if not S > 0.0:
        raise ValueError(
            "The Storage needs to be positiv")

    res = np.zeros(time.shape + rad.shape)

    for ti, te in np.ndenumerate(time):
        for ri, re in np.ndenumerate(rad):
            res[ti+ri] = Qw/(4.0*np.pi*T)*exp1(re**2*S/(4*T*te))

    if not struc_grid and grid_shape:
        res = np.copy(np.diag(res).reshape(grid_shape))

    # add the reference head
    res += hinf

    return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()
