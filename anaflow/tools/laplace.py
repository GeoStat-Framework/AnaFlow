# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing functions concerning the laplace transformation.

.. currentmodule:: anaflow.tools.laplace

The following functions are provided

.. autosummary::

   get_lap
   get_lap_inv
   lap_trans
   stehfest
"""
from math import floor, factorial
import numpy as np
from scipy.integrate import quad

__all__ = ["get_lap", "lap_trans", "get_lap_inv", "stehfest"]


def get_lap(func, arg_dict=None, **kwargs):
    """
    Callable Laplace transform.

    Get the Laplace transform of a given function as a callable function.

    Parameters
    ----------
    func : :any:`callable`
        function that shall be transformed.
        The first argument needs to be the time-variable:
        ``func(t, **kwargs)``

        `func` should be capable of taking numpy arrays as input for `s` and
        the first shape component of the output of `func` should match the
        shape of `s`.
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``
        This is designed for overlapping keywords. Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``.

    Returns
    -------
    :any:`callable`
        The Laplace transformed of the given function.

    Raises
    ------
    ValueError
        If `func` is not callable.
    """
    # check the input
    if not callable(func):
        raise ValueError("The given function needs to be callable")

    # define the returned function
    def ret_func(phase):
        """Return function for the Laplace transformed."""
        return lap_trans(func, phase, arg_dict=arg_dict, **kwargs)

    return ret_func


def lap_trans(func, phase, arg_dict=None, **kwargs):
    """
    The laplace transform.

    Parameters
    ----------
    func : :any:`callable`
        function that shall be transformed.
        The first argument needs to be the time-variable:
        ``func(s, **kwargs)``

        `func` should be capable of taking numpy arrays as input for `s` and
        the first shape component of the output of `func` should match the
        shape of `s`.
    phase : :class:`float` or :class:`numpy.ndarray`
        phase-points to evaluate the transformed function at
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``
        This is designed for overlapping keywords in ``stehfest`` and
        ``func``.Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all evaluations in phase-space.

    Raises
    ------
    ValueError
        If `func` is not callable.
    """
    # check the input
    if not callable(func):
        raise ValueError("The given function needs to be callable")

    arg_dict = {} if arg_dict is None else arg_dict
    kwargs.update(arg_dict)

    # check and save if phase is scalar
    is_scal = np.isscalar(phase)
    phase = np.array(phase, dtype=float)
    result = np.zeros_like(phase)

    def make_integrand(phase):
        def integrand(val):
            """Integrand for the laplace transform."""
            return np.exp(-phase * val) * func(val, **kwargs)

        return integrand

    for phase_i, phase_e in np.ndenumerate(phase):

        integ = make_integrand(phase_e)
        result[phase_i] = quad(integ, 0, np.inf)[0]

    if is_scal:
        result = np.asscalar(result)

    return result


def get_lap_inv(
    func, method="stehfest", method_dict=None, arg_dict=None, **kwargs
):
    """
    Callable Laplace inversion.

    Get the Laplace inversion of a given function as a callable function.

    Parameters
    ----------
    func : :any:`callable`
        function in laplace-space that shall be inverted.
        The first argument needs to be the laplace-variable:
        ``func(s, **kwargs)``

        `func` should be capable of taking numpy arrays as input for `s` and
        the first shape component of the output of `func` should match the
        shape of `s`.
    method : :class:`str`
        Method that should be used to calculate the inverse.
        One can choose between

        * ``"stehfest"``: for the stehfest algorithm

        Default: ``"stehfest"``
    method_dict : :class:`dict` or :any:`None`, optional
        Keyword arguments for the used method.
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``
        This is designed for overlapping keywords. Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``.

    Returns
    -------
    :any:`callable`
        The Laplace inverse of the given function.

    Raises
    ------
    ValueError
        If `func` is not callable.
    ValueError
        If `method` is unknown.
    """
    # dict with all implemented methods
    method_avail = {"stehfest": stehfest}
    # check the input
    if not callable(func):
        raise ValueError("The given function needs to be callable")
    if method not in method_avail:
        raise ValueError("The given method is unknown: " + str(method))
    # assign the used method
    used_meth = method_avail[method]
    # update kwargs
    if method_dict is None:
        method_dict = {}
    kwargs.update(method_dict)
    kwargs["arg_dict"] = arg_dict

    # define the returned function
    def ret_func(time):
        """Return function for the Laplace inversion."""
        return used_meth(func, time, **kwargs)

    return ret_func


def stehfest(func, time, bound=12, arg_dict=None, **kwargs):
    r"""
    The stehfest-algorithm for numerical laplace inversion.

    The Inversion was derivide in ''Stehfest 1970''[R1]_
    and is given by the formula

    .. math::
       f\left(t\right) &=\frac{\ln2}{t}\sum_{n=1}^{N}c_{n}\cdot\tilde{f}
       \left(n\cdot\frac{\ln2}{t}\right)\\
       c_{n} &=\left(-1\right)^{n+\frac{N}{2}}\cdot
       \sum_{k=\left\lfloor \frac{n+1}{2}\right\rfloor }
       ^{\min\left\{ n,\frac{N}{2}\right\} }
       \frac{k^{\frac{N}{2}+1}\cdot\binom{2k}{k}}
       {\left(\frac{N}{2}-k\right)!\cdot\left(n-k\right)!
       \cdot\left(2k-n\right)!}

    In the algorithm
    :math:`N` corresponds to ``bound``,
    :math:`\tilde{f}` to ``func`` and
    :math:`t` to ``time``.

    Parameters
    ----------
    func : :any:`callable`
        function in laplace-space that shall be inverted.
        The first argument needs to be the laplace-variable:
        ``func(s, **kwargs)``

        `func` should be capable of taking numpy arrays as input for `s` and
        the first shape component of the output of `func` should match the
        shape of `s`.
    time : :class:`float` or :class:`numpy.ndarray`
        time-points to evaluate the function at
    bound : :class:`int`, optional
        Here you can specify the number of interations within this
        algorithm. Default: ``12``
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        function given in ``func``. Will be merged with ``**kwargs``
        This is designed for overlapping keywords in ``stehfest`` and
        ``func``.Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the function given in ``func``.
        Will be merged with ``arg_dict``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all evaluations in Time-space.

    Raises
    ------
    ValueError
        If `func` is not callable.
    ValueError
        If `time` is not positive.
    ValueError
        If `bound` is not positive.
    ValueError
        If `bound` is not even.

    References
    ----------
    .. [R1] Stehfest, H., ''Algorithm 368:
       Numerical inversion of laplace transforms [d5].''
       Communications of the ACM, 13(1):47-49, 1970

    Notes
    -----
    The parameter ``time`` needs to be strictly positiv.

    The algorithm gets unstable for ``bound`` values above 20.

    Examples
    --------
    >>> f = lambda x: x**-1
    >>> stehfest(f, [1,10,100])
    array([ 1.,  1.,  1.])
    """
    arg_dict = {} if arg_dict is None else arg_dict
    kwargs.update(arg_dict)

    # ensure that t is handled as an 1d-array
    time = np.array(time, dtype=float).reshape(-1)

    # check the input
    if not callable(func):
        raise ValueError("The given function needs to be callable")
    if not np.all(time > 0.0):
        raise ValueError(
            "The time-values need to be positiv for the stehfest-algorithm"
        )
    if bound <= 1:
        raise ValueError(
            "The boundary needs to be >1 for the stehfest-algorithm"
        )
    if bound % 2 != 0:
        raise ValueError(
            "The boundary needs to be even for the stehfest-algorithm"
        )

    # get all coefficient factors at once
    c_fac = c_array(bound)
    t_fac = np.log(2.0) / time
    # store every function-argument needed in one array
    fargs = np.einsum("i,j->ij", t_fac, np.arange(1, bound + 1))
    # get every function-value needed with one call of 'func'
    lap_val = func(fargs.reshape(-1), **kwargs)
    # reshape again for further summation
    lap_val = lap_val.reshape(fargs.shape + lap_val.shape[1:])
    # sumation of c*f
    res = np.einsum("ij...,j->i...", lap_val, c_fac)
    # multiply with ln(2)/t
    res = np.einsum("i...,i->i...", res, t_fac)

    return res


def c_array(bound=12):
    """
    Array of coefficients for the stehfest-algorithm.

    Parameters
    ----------
    bound : :class:`int`, optional
        The number of interations. Default: ``12``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all coefficinets needed.
    """
    c_lookup = {
        2: np.array([2.000000000000000000e00, -2.000000000000000000e00]),
        4: np.array(
            [
                -2.000000000000000000e00,
                2.600000000000000000e01,
                -4.800000000000000000e01,
                2.400000000000000000e01,
            ]
        ),
        6: np.array(
            [
                1.000000000000000000e00,
                -4.900000000000000000e01,
                3.660000000000000000e02,
                -8.580000000000000000e02,
                8.100000000000000000e02,
                -2.700000000000000000e02,
            ]
        ),
        8: np.array(
            [
                -3.333333333333333148e-01,
                4.833333333333333570e01,
                -9.060000000000000000e02,
                5.464666666666666060e03,
                -1.437666666666666606e04,
                1.873000000000000000e04,
                -1.194666666666666606e04,
                2.986666666666666515e03,
            ]
        ),
        10: np.array(
            [
                8.333333333333332871e-02,
                -3.208333333333333570e01,
                1.279000000000000000e03,
                -1.562366666666666606e04,
                8.424416666666665697e04,
                -2.369575000000000000e05,
                3.759116666666666861e05,
                -3.400716666666666861e05,
                1.640625000000000000e05,
                -3.281250000000000000e04,
            ]
        ),
        12: np.array(
            [
                -1.666666666666666644e-02,
                1.601666666666666572e01,
                -1.247000000000000000e03,
                2.755433333333333212e04,
                -2.632808333333333139e05,
                1.324138699999999953e06,
                -3.891705533333333209e06,
                7.053286333333333023e06,
                -8.005336500000000000e06,
                5.552830500000000000e06,
                -2.155507200000000186e06,
                3.592512000000000116e05,
            ]
        ),
        14: np.array(
            [
                2.777777777777777884e-03,
                -6.402777777777778567e00,
                9.240499999999999545e02,
                -3.459792777777777519e04,
                5.403211111111111240e05,
                -4.398346366666667163e06,
                2.108759177777777612e07,
                -6.394491304444444180e07,
                1.275975795499999970e08,
                -1.701371880833333433e08,
                1.503274670333333313e08,
                -8.459216150000000000e07,
                2.747888476666666567e07,
                -3.925554966666666791e06,
            ]
        ),
        16: np.array(
            [
                -3.968253968253968251e-04,
                2.133730158730158699e00,
                -5.510166666666666515e02,
                3.350016111111111240e04,
                -8.126651111111111240e05,
                1.007618376666666567e07,
                -7.324138297777777910e07,
                3.390596320730158687e08,
                -1.052539536278571367e09,
                2.259013328583333492e09,
                -3.399701984433333397e09,
                3.582450461699999809e09,
                -2.591494081366666794e09,
                1.227049828766666651e09,
                -3.427345554285714030e08,
                4.284181942857142538e07,
            ]
        ),
    }
    if bound in c_lookup:
        return c_lookup[bound]
    return _carr(bound)


def _carr(bound):
    res = np.zeros(bound)
    for i in range(1, bound + 1):
        res[i - 1] = _c(i, bound)
    return res


def _c(i, bound):
    res = 0.0
    for k in range(int(floor((i + 1) / 2.0)), min(i, bound // 2) + 1):
        res += _d(k, i, bound)
    res *= (-1) ** (i + bound / 2)
    return res


def _d(k, i, bound):
    res = ((float(k)) ** (bound / 2 + 1)) * (factorial(2 * k))
    res /= (
        (factorial(bound / 2 - k))
        * (factorial(i - k))
        * (factorial(2 * k - i))
    )
    res /= factorial(k) ** 2
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
