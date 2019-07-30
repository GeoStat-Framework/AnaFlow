# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing special functions.

.. currentmodule:: anaflow.tools.special

The following functions are provided

.. autosummary::

   Shaper
   sph_surf
   specialrange
   specialrange_cut
   aniso
   well_solution
   inc_gamma
   tpl_hyp
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.special import gamma, gammaincc, exp1, expn, hyp2f1

__all__ = [
    "Shaper",
    "sph_surf",
    "specialrange",
    "specialrange_cut",
    "aniso",
    "well_solution",
    "inc_gamma",
    "tpl_hyp",
]


class Shaper(object):
    """A class to reshape radius-time input-output in a good way."""

    def __init__(self, time, rad, struc_grid):
        self.time_scalar = np.isscalar(time)
        self.rad_scalar = np.isscalar(rad)
        self.time = np.array(time)
        self.rad = np.array(rad)
        self.struc_grid = struc_grid
        self.time_shape = self.time.shape
        self.rad_shape = self.rad.shape
        self.time = self.time.reshape(-1)
        self.rad = self.rad.reshape(-1)
        self.time_no = self.time.shape[0]
        self.rad_no = self.rad.shape[0]

        self.time_min = np.min(self.time)
        self.time_max = np.max(self.time)
        self.rad_min = np.min(self.rad)
        self.rad_max = np.max(self.rad)

        self.time_gz = self.time > 0
        self.time_mat = np.outer(
            self.time[self.time_gz], np.ones_like(self.rad)
        )
        self.rad_mat = np.outer(
            np.ones_like(self.time[self.time_gz]), self.rad
        )

        if not self.struc_grid and not self.rad_shape == self.time_shape:
            raise ValueError("No struc_grid: shape of time & radius differ")
        if np.any(self.time < 0.0):
            raise ValueError("The given times need to be positive.")
        if np.any(self.rad <= 0.0):
            raise ValueError("The given radii need to be non-negative.")

    def reshape(self, result):
        """Reshape a time-rad result according to the input shape."""
        np.asanyarray(result)
        if self.struc_grid:
            result = result.reshape(self.time_shape + self.rad_shape)
        elif self.rad_shape:
            result = np.diag(result).reshape(self.rad_shape)
        if self.time_scalar and self.rad_scalar:
            result = np.asscalar(result)
        return result


def sph_surf(dim):
    """Surface of the d-dimensional sphere."""
    return 2.0 * np.sqrt(np.pi) ** dim / gamma(dim / 2.0)


def specialrange(val_min, val_max, steps, typ="log"):
    """
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

    Examples
    --------
    >>> specialrange(1,10,4)
    array([  1.        ,   2.53034834,   5.23167968,  10.        ])
    """
    if typ in ["logarithmic", "log"]:
        rng = np.expm1(
            np.linspace(np.log1p(val_min), np.log1p(val_max), steps)
        )
    elif typ in ["linear", "lin"]:
        rng = np.linspace(val_min, val_max, steps)
    elif typ in ["quadratic", "quad"]:
        rng = (np.linspace(np.sqrt(val_min), np.sqrt(val_max), steps)) ** 2
    elif typ in ["cubic", "cub"]:
        rng = (
            np.linspace(
                np.power(val_min, 1 / 3.0), np.power(val_max, 1 / 3.0), steps
            )
        ) ** 3
    elif isinstance(typ, (float, int)):
        rng = (
            np.linspace(
                np.power(val_min, 1.0 / typ),
                np.power(val_max, 1.0 / typ),
                steps,
            )
        ) ** typ
    else:
        rng = np.linspace(val_min, val_max, steps)

    return rng


def specialrange_cut(val_min, val_max, steps, val_cut=np.inf, typ="log"):
    """
    Calculation of special point ranges.

    Calculation of special point ranges with a cut-off value.

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

    Examples
    --------
    >>> specialrange_cut(1,10,4)
    array([  1.        ,   2.53034834,   5.23167968,  10.        ])
    """
    if val_max > val_cut:
        rng = specialrange(val_min, val_cut, steps - 1, typ)
        return np.hstack((rng, val_max))

    return specialrange(val_min, val_max, steps, typ)


def aniso(e):
    """
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

    Examples
    --------
    >>> aniso(0.5)
    0.23639985871871511
    """
    if e < 0 or e > 1:
        raise ValueError("Anisotropy ratio 'e' must be within 0 and 1")

    if np.isclose(e, 1):
        res = 1.0 / 3.0
    elif np.isclose(e, 0):
        res = 0.0
    else:
        res = e / (2 * (1.0 - e ** 2))
        res *= (
            1.0
            / np.sqrt(1.0 - e ** 2)
            * np.arctan(np.sqrt(1.0 / e ** 2 - 1.0))
            - e
        )

    return res


def well_solution(time, rad, T, S, Qw, struc_grid=True, hinf=0.0):
    """
    The classical Theis solution.

    The classical Theis solution for transient flow under a pumping condition
    in a confined and homogeneous aquifer.

    This solution was presented in ''Theis 1935''[R9]_.

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
        If ``time`` is negative.
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
       Trans. Am. Geophys. Union, 16, 519-524, 1935

    Notes
    -----
    The parameters ``rad``, ``T`` and ``S`` will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> well_solution([1,2,3], [10,100], 0.001, 0.001, -0.001)
    array([[-0.24959541, -0.14506368, -0.08971485],
           [-0.43105106, -0.32132823, -0.25778313]])
    """
    Input = Shaper(time, rad, struc_grid)

    if not T > 0.0:
        raise ValueError("The Transmissivity needs to be positiv")
    if not S > 0.0:
        raise ValueError("The Storage needs to be positiv")

    time_mat = Input.time_mat
    rad_mat = Input.rad_mat

    res = np.zeros((Input.time_no, Input.rad_no))
    res[Input.time_gz, :] = (
        Qw / (4.0 * np.pi * T) * exp1(rad_mat ** 2 * S / (4 * T * time_mat))
    )
    res = Input.reshape(res)
    if Qw > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += hinf
    return res


def grf(time, rad, K, S, Qw, dim=2, lat_ext=1.0, struc_grid=True, hinf=0.0):
    """
    The general radial flow (GRF) model for a pumping test.

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
        If ``time`` is negative.
    ValueError
        If shape of ``rad`` and ``time`` differ in case of
        ``struc_grid`` is ``True``.
    ValueError
        If ``K`` is not positiv.
    ValueError
        If ``S`` is not positiv.
    """
    Input = Shaper(time, rad, struc_grid)

    if not K > 0.0:
        raise ValueError("The Conductivity needs to be positiv")
    if not S > 0.0:
        raise ValueError("The Storage needs to be positiv")

    time_mat = Input.time_mat
    rad_mat = Input.rad_mat
    u = S * rad_mat ** 2 / (4 * K * time_mat)
    nu = 1.0 - dim / 2.0

    res = np.zeros((Input.time_no, Input.rad_no))
    res[Input.time_gz, :] = inc_gamma(-nu, u)
    res[Input.time_gz, :] *= (
        Qw
        / (4.0 * np.pi ** (1 - nu) * K * lat_ext ** (3.0 - dim))
        * rad_mat ** (2 * nu)
    )
    res = Input.reshape(res)
    if Qw > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += hinf
    return res


def inc_gamma(s, x):
    r"""The (upper) incomplete gamma function.

    Given by: :math:`\Gamma(s,x) = \int_x^{\infty} t^{s-1}\,e^{-t}\,{\rm d}t`

    Parameters
    ----------
    s : :class:`float`
        exponent in the integral
    x : :class:`numpy.ndarray`
        input values
    """
    if np.isclose(s, 0):
        return exp1(x)
    if np.isclose(s, np.around(s)) and s < -0.5:
        return x ** (s - 1) * expn(int(1 - np.around(s)), x)
    if s < 0:
        return (inc_gamma(s + 1, x) - x ** s * np.exp(-x)) / s
    return gamma(s) * gammaincc(s, x)


def tpl_hyp(rad, dim, hurst, corr, prop):
    """Hyp_2F1 for the TPL CG model."""
    x = 1.0 / (1.0 + (prop * rad / corr) ** 2)
    return x ** (dim / 2.0) * hyp2f1(dim / 2.0, 1, dim / 2.0 + 1 + hurst, x)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
