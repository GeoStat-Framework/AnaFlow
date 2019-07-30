# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing several mean calculating routines.

.. currentmodule:: anaflow.tools.mean

The following functions are provided

.. autosummary::

   rad_amean_func
   rad_gmean_func
   rad_hmean_func
   rad_pmean_func
"""
# pylint: disable=E1137, C0103
from __future__ import absolute_import, division, print_function

import numpy as np

from scipy.integrate import quad as integ

__all__ = [
    "rad_amean_func",
    "rad_gmean_func",
    "rad_hmean_func",
    "rad_pmean_func",
]


def rad_amean_func(func, val_arr, arg_dict=None, **kwargs):
    r"""
    Calculating the arithmetic mean.

    Calculating the arithmetic mean of a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\frac{2}{r_{i+1}^2-r_i^2}
       \intop^{r_{i+1}}_{r_i} r\cdot f(r)\, dr

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

    Examples
    --------
    >>> f = lambda x: x**2
    >>> rad_amean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    """
    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError("The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError("To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i + 1] for i in range(len(val_arr) - 1)):
        raise ValueError("The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """Dummy function providing the integrand."""

        def integrand(val):
            """Integrand for the geometric mean ``r*log(f(r))``."""
            return 2 * val * func(val, **kwargs)

        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i + 1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(
                _step(func, kwargs), val_arr[i], val_arr[i + 1]
            )[0]
            func_arr[i] /= val_arr[i + 1] ** 2 - val_arr[i] ** 2

    return func_arr


def rad_gmean_func(func, val_arr, arg_dict=None, **kwargs):
    r"""
    Calculating the geometric mean.

    Calculating the geometric meanof a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\exp\left(\frac{2}{r_{i+1}^2-r_i^2}
       \intop^{r_{i+1}}_{r_i} r\cdot\ln(f(r))\, dr\right)

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

    Examples
    --------
    >>> f = lambda x: x**2
    >>> rad_gmean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    """
    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError("The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError("To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i + 1] for i in range(len(val_arr) - 1)):
        raise ValueError("The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """Dummy function providing the integrand."""

        def integrand(val):
            """Integrand for the geometric mean ``r*log(f(r))``."""
            return 2 * val * np.log(func(val, **kwargs))

        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i + 1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(
                _step(func, kwargs), val_arr[i], val_arr[i + 1]
            )[0]
            func_arr[i] = np.exp(
                func_arr[i] / (val_arr[i + 1] ** 2 - val_arr[i] ** 2)
            )

    return func_arr


def rad_hmean_func(func, val_arr, arg_dict=None, **kwargs):
    r"""
    Calculating the harmonic mean.

    Calculating the harmonic mean of a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\left(\frac{2}{r_{i+1}^2-r_i^2}
       \intop^{r_{i+1}}_{r_i} r\cdot(f(r))^{-1}\, dr\right)^{-1}

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

    Examples
    --------
    >>> f = lambda x: x**2
    >>> rad_hmean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    """
    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError("The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError("To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i + 1] for i in range(len(val_arr) - 1)):
        raise ValueError("The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """Dummy function providing the integrand."""

        def integrand(val):
            """Integrand for the geometric mean ``r*log(f(r))``."""
            return 2 * val / func(val, **kwargs)

        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i + 1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(
                _step(func, kwargs), val_arr[i], val_arr[i + 1]
            )[0]
            func_arr[i] = 1.0 / (
                func_arr[i] / (val_arr[i + 1] ** 2 - val_arr[i] ** 2)
            )

    return func_arr


def rad_pmean_func(func, val_arr, p=1.0, arg_dict=None, **kwargs):
    r"""
    Calculating the p-mean.

    Calculating the p-mean of a radial symmetric function
    for given consecutive radii defining 2D disks by the following formula

    .. math::
       f_i=\left(\frac{2}{r_{i+1}^2-r_i^2}
       \intop^{r_{i+1}}_{r_i} r\cdot(f(r))^p\, dr\right)^{\frac{1}{p}}

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

    Examples
    --------
    >>> f = lambda x: x**2
    >>> rad_pmean_func(f, [1,2,3])
    array([ 2.33588885,  6.33423311])
    """
    # if p is 0, the limit-case of the geometric mean is returned
    if p == 0:
        return rad_gmean_func(func, val_arr, arg_dict, **kwargs)

    if arg_dict is None:
        arg_dict = {}
    kwargs.update(arg_dict)

    val_arr = np.array(val_arr, dtype=float).reshape(-1)
    parts = len(val_arr) - 1

    if not callable(func):
        raise ValueError("The given function needs to be callable")
    if len(val_arr) < 2:
        raise ValueError("To few input values in val_arr. Need at least 2.")
    if not all(val_arr[i] < val_arr[i + 1] for i in range(len(val_arr) - 1)):
        raise ValueError("The input values need to be sorted")

    # dummy function defining the needed integrand
    def _step(func, kwargs):
        """Dummy function providing the integrand."""

        def integrand(val):
            """Integrand for the geometric mean ``r*log(f(r))``."""
            return 2 * val * func(val, **kwargs) ** p

        return integrand

    # creating the output array
    func_arr = np.zeros_like(val_arr[:-1], dtype=float)

    # iterating over the input values
    for i in range(parts):
        # if one side is infinity, the function is evaluated at infinity
        if val_arr[i + 1] == np.inf:
            func_arr[i] = func(np.inf, **kwargs)
        else:
            func_arr[i] = integ(
                _step(func, kwargs), val_arr[i], val_arr[i + 1]
            )[0]
            func_arr[i] = (
                func_arr[i] / (val_arr[i + 1] ** 2 - val_arr[i] ** 2)
            ) ** (1.0 / p)

    return func_arr


if __name__ == "__main__":
    import doctest

    doctest.testmod()
