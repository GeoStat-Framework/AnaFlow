# -*- coding: utf-8 -*-
"""
===============================================
Laplace transformation (:mod:`anaflow.laplace`)
===============================================

.. currentmodule:: anaflow.laplace

Anaflow subpackage providing functions concerning the laplace transformation.

Functions
---------
The following functions are provided

.. autosummary::
   :toctree: generated/

   stehfest - The stehfest-algorithm for numerical laplace inversion.

"""

from __future__ import absolute_import, division, print_function

from math import floor, factorial
import numpy as np

__all__ = ["stehfest"]


C_LOOKUP = {2: np.array([2.000000000000000000e+00,
                         -2.000000000000000000e+00]),
            4: np.array([-2.000000000000000000e+00,
                         2.600000000000000000e+01,
                         -4.800000000000000000e+01,
                         2.400000000000000000e+01]),
            6: np.array([1.000000000000000000e+00,
                         -4.900000000000000000e+01,
                         3.660000000000000000e+02,
                         -8.580000000000000000e+02,
                         8.100000000000000000e+02,
                         -2.700000000000000000e+02]),
            8: np.array([-3.333333333333333148e-01,
                         4.833333333333333570e+01,
                         -9.060000000000000000e+02,
                         5.464666666666666060e+03,
                         -1.437666666666666606e+04,
                         1.873000000000000000e+04,
                         -1.194666666666666606e+04,
                         2.986666666666666515e+03]),
            10: np.array([8.333333333333332871e-02,
                          -3.208333333333333570e+01,
                          1.279000000000000000e+03,
                          -1.562366666666666606e+04,
                          8.424416666666665697e+04,
                          -2.369575000000000000e+05,
                          3.759116666666666861e+05,
                          -3.400716666666666861e+05,
                          1.640625000000000000e+05,
                          -3.281250000000000000e+04]),
            12: np.array([-1.666666666666666644e-02,
                          1.601666666666666572e+01,
                          -1.247000000000000000e+03,
                          2.755433333333333212e+04,
                          -2.632808333333333139e+05,
                          1.324138699999999953e+06,
                          -3.891705533333333209e+06,
                          7.053286333333333023e+06,
                          -8.005336500000000000e+06,
                          5.552830500000000000e+06,
                          -2.155507200000000186e+06,
                          3.592512000000000116e+05]),
            14: np.array([2.777777777777777884e-03,
                          -6.402777777777778567e+00,
                          9.240499999999999545e+02,
                          -3.459792777777777519e+04,
                          5.403211111111111240e+05,
                          -4.398346366666667163e+06,
                          2.108759177777777612e+07,
                          -6.394491304444444180e+07,
                          1.275975795499999970e+08,
                          -1.701371880833333433e+08,
                          1.503274670333333313e+08,
                          -8.459216150000000000e+07,
                          2.747888476666666567e+07,
                          -3.925554966666666791e+06]),
            16: np.array([-3.968253968253968251e-04,
                          2.133730158730158699e+00,
                          -5.510166666666666515e+02,
                          3.350016111111111240e+04,
                          -8.126651111111111240e+05,
                          1.007618376666666567e+07,
                          -7.324138297777777910e+07,
                          3.390596320730158687e+08,
                          -1.052539536278571367e+09,
                          2.259013328583333492e+09,
                          -3.399701984433333397e+09,
                          3.582450461699999809e+09,
                          -2.591494081366666794e+09,
                          1.227049828766666651e+09,
                          -3.427345554285714030e+08,
                          4.284181942857142538e+07])}


def stehfest(func, time, bound=12, kwargs=None):
    '''
    The stehfest-algorithm for numerical laplace inversion.

    Parameters
    ----------
    func : function
        function in laplace-space that shall be inverted.
        The first argument needs to be the laplace-variable: func(s, **kwargs)
    time : float
        time-points to evaluate the function at
    bound : int, optional
        Here you can specify the number of interations within this
        algorithm. Default: 12
    kwargs : dict, optional
        Keyword-arguments that are forwarded to the function given in "func".
        Default: None

    Returns
    -------
    stehfest : ndarray
        Array with all evaluations in Time-space.

    Notes
    -----
    The parameter "time" needs to be strictly positiv.
    The algorithm gets unstable for "bound" values above 20.

    Example
    -------
    >>> f = lambda x: x**-1
    >>> stehfest(f, [1,10,100])
    array([ 1.,  1.,  1.])
    '''

    if kwargs is None:
        kwargs = {}

    # check and save if 't' is scalar
    is_scal = np.isscalar(time)

    # ensure that t is handled as an 1d-array
    time = np.array(time).reshape(-1)

    # check the input
    if not (np.all(time > 0.0)):
        raise ValueError(
            "The time-values need to be positiv for the stehfest-algorithm")
    if not (bound > 1):
        raise ValueError(
            "The boundary needs to be >1 for the stehfest-algorithm")
    if not (bound % 2 == 0):
        raise ValueError(
            "The boundary needs to be even for the stehfest-algorithm")

    # get all coefficient factors at once
    c_fac = _c_array(bound)
    t_fac = np.log(2.0)/time

    # store every function-argument needed in one array
    fargs = np.outer(t_fac, np.arange(1, bound+1))

    # get every function-value needed with one call of 'func'
    lap_val = func(fargs.reshape(-1), **kwargs)
    lap_val = lap_val.reshape(*(fargs.shape + lap_val.shape[1:]))

    # do all the sumation with fancy indexing in numpy
    res = np.tensordot(lap_val, c_fac, axes=(1, 0))
    res = np.rollaxis(np.multiply(np.rollaxis(res, 0, res.ndim), t_fac), -1, 0)

    # reformat the result according to the input
    res = np.squeeze(res)
    if np.ndim(res) == 0 and is_scal:
        res = np.asscalar(res)

    return res


def _c_array(bound):
    '''
    Array of coefficients for the stehfest-algorithm.

    Parameters
    ----------
    bound : int, optional
        The number of interations. Default: 12

    Returns
    -------
    c_array : ndarray
        Array with all coefficinets needed.
    '''

    if bound in C_LOOKUP:
        return C_LOOKUP[bound]

    return _carr(bound)


def _carr(bound):
    res = np.zeros(bound)
    for l in range(1, bound+1):
        res[l-1] = _c(l, bound)
    return res


def _c(l, bound):
    res = 0.0
    for k in range(int(floor((l+1)/2.0)), min(l, bound//2)+1):
        res += _d(k, l, bound)
    res *= (-1)**(l+bound/2)
    return res


def _d(k, l, bound):
    res = ((float(k))**(bound/2+1))*(factorial(2*k))
    res /= (factorial(bound/2-k))*(factorial(l-k))*(factorial(2*k-l))
    res /= (factorial(k)**2)
    return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()
