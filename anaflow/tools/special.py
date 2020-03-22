# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing special functions.

.. currentmodule:: anaflow.tools.special

The following functions are provided

.. autosummary::

   Shaper
   step_f
   sph_surf
   specialrange
   specialrange_cut
   aniso
   well_solution
   grf_solution
   inc_gamma
   tpl_hyp
   neuman2004_trans
"""

import numpy as np
from scipy.special import gamma, gammaincc, exp1, expn, hyp2f1

__all__ = [
    "Shaper",
    "step_f",
    "sph_surf",
    "specialrange",
    "specialrange_cut",
    "aniso",
    "well_solution",
    "grf_solution",
    "inc_gamma",
    "tpl_hyp",
    "neuman2004_trans",
]


class Shaper(object):
    """
    A class to reshape radius-time input-output in a good way.

    Parameters
    ----------
    time : :class:`numpy.ndarray` or :class:`float`, optional
        Array with all time-points where the function should be evaluated.
        Default: 0
    rad : :class:`numpy.ndarray` or :class:`float`, optional
        Array with all radii where the function should be evaluated.
        Default: 0
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured t-r grid is created.
        Default: ``True``
    """

    def __init__(self, time=0, rad=0, struc_grid=True):
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


def step_f(rad, r_part, f_part):
    """Callalbe step function."""
    return np.piecewise(
        np.array(rad),
        [
            np.logical_and(r1 <= rad, rad < r2)
            for r1, r2 in zip(r_part[:-1], r_part[1:])
        ],
        f_part,
    )


def sph_surf(dim):
    """Surface of the d-dimensional sphere."""
    return 2.0 * np.sqrt(np.pi) ** dim / gamma(dim / 2.0)


def specialrange(val_min, val_max, steps, typ="exp"):
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

        * ``"exp"``: for exponential behavior
        * ``"log"``: for logarithmic behavior
        * ``"geo"``: for geometric behavior
        * ``"lin"``: for linear behavior
        * ``"quad"``: for quadratic behavior
        * ``"cub"``: for cubic behavior
        * :class:`float`: here you can specifi any exponent ("quad" would be
          equivalent to 2)

        Default: ``"exp"``

    Returns
    -------
    :class:`numpy.ndarray`
        Array containing the special range

    Examples
    --------
    >>> specialrange(1,10,4)
    array([ 1.        ,  2.53034834,  5.23167968, 10.        ])
    """
    if typ in ["exponential", "exp"]:
        rng = np.expm1(
            np.linspace(np.log1p(val_min), np.log1p(val_max), steps)
        )
    elif typ in ["logarithmic", "log"]:
        rng = np.log(np.linspace(np.exp(val_min), np.exp(val_max), steps))
    elif typ in ["geometric", "geo", "geom"]:
        rng = np.geomspace(val_min, val_max, steps)
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
        print("specialrange: unknown typ '{}'. Using linear range".format(typ))
        rng = np.linspace(val_min, val_max, steps)

    return rng


def specialrange_cut(val_min, val_max, steps, val_cut=None, typ="exp"):
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
    val_cut : :class:`float`, optional
        Cutting value, if val_max is bigger than this value, the last interval
        will be between val_cut and val_max.
        Default: 100.0
    typ : :class:`str` or :class:`float`, optional
        Setting the kind of range-distribution. One can choose between

        * ``"exp"``: for exponential behavior
        * ``"log"``: for logarithmic behavior
        * ``"geo"``: for geometric behavior
        * ``"lin"``: for linear behavior
        * ``"quad"``: for quadratic behavior
        * ``"cub"``: for cubic behavior
        * :class:`float`: here you can specifi any exponent ("quad" would be
          equivalent to 2)

        Default: ``"exp"``

    Returns
    -------
    :class:`numpy.ndarray`
        Array containing the special range

    Examples
    --------
    >>> specialrange_cut(1,10,4)
    array([ 1.        ,  2.53034834,  5.23167968, 10.        ])
    """
    val_cut = 100.0 if val_cut is None else val_cut
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
    0.2363998587187151
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


def well_solution(
    time,
    rad,
    storage,
    transmissivity,
    rate=-1e-4,
    h_bound=0.0,
    struc_grid=True,
):
    """
    The classical Theis solution.

    The classical Theis solution for transient flow under a pumping condition
    in a confined and homogeneous aquifer.

    This solution was presented in ''Theis 1935''[R9]_.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated.
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated.
    storage : :class:`float`
        Storage of the aquifer.
    transmissivity : :class:`float`
        Transmissivity of the aquifer.
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_bound : :class:`float`, optional
        Reference head at the outer boundary at infinity. Default: ``0.0``
    struc_grid : :class:`bool`, optional
        If this is set to "False", the "rad" and "time" array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``

    Returns
    -------
    head : :class:`numpy.ndarray`
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
        If ``transmissivity`` is not positiv.
    ValueError
        If ``storage`` is not positiv.

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
    >>> well_solution([10,100], [1,2,3], 0.001, 0.001, -0.001)
    array([[-0.24959541, -0.14506368, -0.08971485],
           [-0.43105106, -0.32132823, -0.25778313]])
    """
    Input = Shaper(time, rad, struc_grid)

    if not transmissivity > 0.0:
        raise ValueError("The Transmissivity needs to be positive.")
    if not storage > 0.0:
        raise ValueError("The Storage needs to be positive.")

    time_mat = Input.time_mat
    rad_mat = Input.rad_mat

    res = np.zeros((Input.time_no, Input.rad_no))
    res[Input.time_gz, :] = (
        rate
        / (4.0 * np.pi * transmissivity)
        * exp1(rad_mat ** 2 * storage / (4 * transmissivity * time_mat))
    )
    res = Input.reshape(res)
    if rate > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += h_bound
    return res


def grf_solution(
    time,
    rad,
    storage,
    conductivity,
    dim=2,
    lat_ext=1.0,
    rate=-1e-4,
    h_bound=0.0,
    struc_grid=True,
):
    """
    The general radial flow (GRF) model for a pumping test.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated.
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated.
    storage : :class:`float`
        Storage coefficient of the aquifer.
    conductivity : :class:`float`
        Conductivity of the aquifer.
    dim : :class:`float`, optional
        Fractional dimension of the aquifer. Default: ``2.0``
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer. Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_bound : :class:`float`, optional
        Reference head at the outer boundary at infinity. Default: ``0.0``
    struc_grid : :class:`bool`, optional
        If this is set to "False", the "rad" and "time" array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``

    Returns
    -------
    head : :class:`numpy.ndarray`
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
        If ``conductivity`` is not positiv.
    ValueError
        If ``storage`` is not positiv.
    """
    Input = Shaper(time, rad, struc_grid)

    if not conductivity > 0.0:
        raise ValueError("The Conductivity needs to be positive.")
    if not storage > 0.0:
        raise ValueError("The Storage needs to be positive.")

    time_mat = Input.time_mat
    rad_mat = Input.rad_mat
    u = storage * rad_mat ** 2 / (4 * conductivity * time_mat)
    nu = 1.0 - dim / 2.0

    res = np.zeros((Input.time_no, Input.rad_no))
    res[Input.time_gz, :] = inc_gamma(-nu, u)
    res[Input.time_gz, :] *= (
        rate
        / (4.0 * np.pi ** (1 - nu) * conductivity * lat_ext ** (3.0 - dim))
        * rad_mat ** (2 * nu)
    )
    res = Input.reshape(res)
    if rate > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += h_bound
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


def neuman2004_trans(rad, trans_gmean, var, len_scale):
    r"""The apparent transmissivity from Neuman 2004.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
    """
    t_h = trans_gmean * np.exp(-var / 2)
    return t_h + (trans_gmean - t_h) * neuman2004_phi(0.5 * rad / len_scale)


def neuman2004_phi(alpha):
    r"""The phi function from Neuman 2004.

    Parameters
    ----------
    alpha : :class:`numpy.ndarray`
        The ratio r/(2l)
    """
    alpha = np.array(np.abs(alpha), dtype=np.double)
    a_lo = alpha < 1
    res = np.ones_like(alpha)
    res[a_lo] = 3 * alpha[a_lo] ** 2 - 2 * alpha[a_lo] ** 3
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
