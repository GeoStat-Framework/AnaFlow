# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing several mean calculating routines.

.. currentmodule:: anaflow.tools.mean

The following functions are provided

.. autosummary::

   annular_fmean
   annular_amean
   annular_gmean
   annular_hmean
   annular_pmean
"""
# pylint: disable=E1137, C0103
from __future__ import absolute_import, division, print_function

import numpy as np

from scipy.integrate import quad as integ

__all__ = [
    "annular_fmean",
    "annular_amean",
    "annular_gmean",
    "annular_hmean",
    "annular_pmean",
]


def annular_fmean(
    func, val_arr, f_def, f_inv, ann_dim=2, arg_dict=None, **kwargs
):
    r"""
    Calculating the annular generalized f-mean.

    Calculating the annular generalized f-mean of a radial symmetric function
    for given consecutive radii defining annuli by the following formula

    .. math::
       \varphi_i=f^{-1}\left(\frac{d}{r_{i+1}^d-r_i^d}
       \intop^{r_{i+1}}_{r_i} r^{d-1}\cdot f(\varphi(r))\, dr \right)

    Parameters
    ----------
    func : :any:`callable`
        Function that should be used (:math:`\varphi` in the formula).
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        Radii defining the annuli.
    ann_dim : :class:`float`, optional
        The dimension of the annuli.
        Default: ``2.0``
    f_def : :any:`callable`
        Function defining the f-mean.
    f_inv : :any:`callable`
        Inverse of the function defining the f-mean.
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``annular_fmean`` and
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
        If `f_def` is not callable.
    ValueError
        If `f_inv` is not callable.
    ValueError
        If ``val_arr`` has less than 2 values.
    ValueError
        If ``val_arr`` is not sorted in incresing order.

    Notes
    -----
    If the last value in val_arr is "inf", the given function should provide
    a value for "inf" as input: ``func(float("inf"))``
    """
    arg_dict = {} if arg_dict is None else arg_dict
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=np.double).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError("The given function needs to be callable")
    if not callable(f_def):
        raise ValueError("The f-mean function needs to be callable")
    if not callable(f_inv):
        raise ValueError("The inverse f-mean function needs to be callable")
    if not np.all(
        np.isclose(
            f_inv(f_def(func(val_arr, **kwargs))), func(val_arr, **kwargs)
        )
    ):
        raise ValueError("f_def and f_inv need to be inverse to each other")
    if len(val_arr) < 2:
        raise ValueError("To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i + 1] for i in range(len(val_arr) - 1)):
        raise ValueError("The input values need to be sorted")

    def integrand(val):
        """Integrand for the f-mean ``r^(d-1)*f(phi(r))``."""
        return val ** (ann_dim - 1) * f_def(func(val, **kwargs))

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=np.double)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i + 1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = f_inv(
                ann_dim
                * integ(integrand, val_arr[i], val_arr[i + 1])[0]
                / (val_arr[i + 1] ** ann_dim - val_arr[i] ** ann_dim)
            )

    return func_arr


def annular_amean(func, val_arr, ann_dim=2, arg_dict=None, **kwargs):
    r"""
    Calculating the annular arithmetic mean.

    Calculating the annular arithmetic mean of a radial symmetric function
    for given consecutive radii defining annuli by the following formula

    .. math::
       \varphi_i=\frac{d}{r_{i+1}^d-r_i^d}
       \intop^{r_{i+1}}_{r_i} r^{d-1}\cdot \varphi(r)\, dr

    Parameters
    ----------
    func : :any:`callable`
        Function that should be used (:math:`\varphi` in the formula).
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        Radii defining the annuli.
    ann_dim : :class:`float`, optional
        The dimension of the annuli.
        Default: ``2.0``
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``annular_amean`` and
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
    """
    return annular_fmean(
        func=func,
        val_arr=val_arr,
        f_def=lambda x: x,
        f_inv=lambda x: x,
        ann_dim=ann_dim,
        arg_dict=arg_dict,
        **kwargs
    )


def annular_gmean(func, val_arr, ann_dim=2, arg_dict=None, **kwargs):
    r"""
    Calculating the annular geometric mean.

    Calculating the annular geometric mean of a radial symmetric function
    for given consecutive radii defining annuli by the following formula

    .. math::
       \varphi_i=\exp\left(\frac{d}{r_{i+1}^d-r_i^d}
       \intop^{r_{i+1}}_{r_i} r^{d-1}\cdot \ln(\varphi(r))\, dr \right)

    Parameters
    ----------
    func : :any:`callable`
        Function that should be used (:math:`\varphi` in the formula).
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        Radii defining the annuli.
    ann_dim : :class:`float`, optional
        The dimension of the annuli.
        Default: ``2.0``
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``annular_gmean`` and
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

    Examples
    --------
    >>> f = lambda x: x**2
    >>> annular_gmean(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    """
    return annular_fmean(
        func=func,
        val_arr=val_arr,
        f_def=np.log,
        f_inv=np.exp,
        ann_dim=ann_dim,
        arg_dict=arg_dict,
        **kwargs
    )


def annular_hmean(func, val_arr, ann_dim=2, arg_dict=None, **kwargs):
    r"""
    Calculating the annular harmonic mean.

    Calculating the annular harmonic mean of a radial symmetric function
    for given consecutive radii defining annuli by the following formula

    .. math::
       \varphi_i=\left(\frac{d}{r_{i+1}^d-r_i^d}
       \intop^{r_{i+1}}_{r_i} r^{d-1}\cdot \varphi(r)^{-1}\, dr \right)^{-1}

    Parameters
    ----------
    func : :any:`callable`
        Function that should be used (:math:`\varphi` in the formula).
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        Radii defining the annuli.
    ann_dim : :class:`float`, optional
        The dimension of the annuli.
        Default: ``2.0``
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``annular_hmean`` and
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
    """
    return annular_fmean(
        func=func,
        val_arr=val_arr,
        f_def=lambda x: 1.0 / x,
        f_inv=lambda x: 1.0 / x,
        ann_dim=ann_dim,
        arg_dict=arg_dict,
        **kwargs
    )


def annular_pmean(func, val_arr, p=2.0, ann_dim=2, arg_dict=None, **kwargs):
    r"""
    Calculating the annular p-mean.

    Calculating the annular p-mean of a radial symmetric function
    for given consecutive radii defining annuli by the following formula

    .. math::
       \varphi_i=\left(\frac{d}{r_{i+1}^d-r_i^d}
       \intop^{r_{i+1}}_{r_i} r^{d-1}\cdot\varphi(r)^p\, dr
       \right)^{\frac{1}{p}}

    Parameters
    ----------
    func : :any:`callable`
        Function that should be used (:math:`\varphi` in the formula).
        The first argument needs to be the radial variable:
        ``func(r, **kwargs)``
    val_arr : :class:`numpy.ndarray`
        Radii defining the annuli.
    p : :class:`float`, optional
        The potency defining the p-mean.
        Default: ``2.0``
    ann_dim : :class:`float`, optional
        The dimension of the annuli.
        Default: ``2.0``
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``annular_pmean`` and
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
    """
    # if p is 0, the limit-case of the geometric mean is returned
    if np.isclose(p, 0):
        return annular_gmean(func, val_arr, ann_dim, arg_dict, **kwargs)

    return annular_fmean(
        func=func,
        val_arr=val_arr,
        f_def=lambda x: np.power(x, p),
        f_inv=lambda x: np.power(x, 1.0 / p),
        ann_dim=ann_dim,
        arg_dict=arg_dict,
        **kwargs
    )
