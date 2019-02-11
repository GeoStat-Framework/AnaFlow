# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing special functions

.. currentmodule:: anaflow.tools.special

The following functions are provided

.. autosummary::

   sph_surf
   radii
   specialrange
   specialrange_cut
   aniso
   well_solution
   inc_gamma
"""

from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.special import gamma, gammaincc, exp1, expn

__all__ = [
    "sph_surf",
    "radii",
    "specialrange",
    "specialrange_cut",
    "aniso",
    "well_solution",
    "inc_gamma",
]


def sph_surf(dim):
    """surface of the d-dimensional sphere"""
    return 2.0 * np.sqrt(np.pi) ** dim / gamma(dim / 2.0)


def radii(parts, rwell=0.0, rinf=np.inf, rlast=500.0, typ="log"):
    """
    Calculation of specific point distributions

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
    """

    if typ == "log":
        if rinf == np.inf:
            # define the partition radii
            p_rad = np.expm1(
                np.linspace(np.log1p(rwell), np.log1p(rlast), parts)
            )
            p_rad = np.append(p_rad, [np.inf])

            # define the points within the partitions to evaluate the function
            f_rad = np.expm1(
                np.linspace(np.log1p(rwell), np.log1p(rlast), parts - 1)
            )
            f_rad = np.append(f_rad, [np.inf])
        else:
            p_rad = np.expm1(
                np.linspace(np.log1p(rwell), np.log1p(rinf), parts + 1)
            )
            f_rad = np.expm1(
                np.linspace(np.log1p(rwell), np.log1p(rinf), parts)
            )

    else:
        if rinf == np.inf:
            # define the partition radii
            p_rad = np.linspace(rwell, rlast, parts)
            p_rad = np.append(p_rad, [np.inf])

            # define the points within the partitions to evaluate the function
            f_rad = np.linspace(rwell, rlast, parts - 1)
            f_rad = np.append(f_rad, [np.inf])
        else:
            p_rad = np.linspace(rwell, rinf, parts + 1)
            f_rad = np.linspace(rwell, rinf, parts)

    return p_rad, f_rad


def specialrange(val_min, val_max, steps, typ="log"):
    """
    Calculation of special point ranges

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
    Calculation of special point ranges

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

    Example
    -------
    >>> specialrange_cut(1,10,4)
    array([  1.        ,   2.53034834,   5.23167968,  10.        ])
    """

    if val_max > val_cut:
        rng = specialrange(val_min, val_cut, steps - 1, typ)
        return np.hstack((rng, val_max))

    return specialrange(val_min, val_max, steps, typ)


def aniso(e):
    """
    The anisotropy function

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
    """

    if not 0.0 <= e <= 1.0:
        raise ValueError("Anisotropieratio 'e' must be within 0 and 1")

    if e == 1.0:
        res = 1.0 / 3.0
    elif e == 0.0:
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


def well_solution(rad, time, T, S, Qw, struc_grid=True, hinf=0.0):
    """
    The classical Theis solution

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
       Trans. Am. Geophys. Union, 16, 519-524, 1935

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
    """

    rad = np.squeeze(rad)
    time = np.array(time).reshape(-1)

    if not struc_grid:
        grid_shape = rad.shape
        rad = rad.reshape(-1)

    if not np.all(rad > 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if not np.all(time > 0.0):
        raise ValueError("The given times need to be > 0")
    if not struc_grid and not rad.shape == time.shape:
        raise ValueError(
            "For unstructured grid the number of time- & radii-pts must equal"
        )
    if not T > 0.0:
        raise ValueError("The Transmissivity needs to be positiv")
    if not S > 0.0:
        raise ValueError("The Storage needs to be positiv")

    res = np.zeros(time.shape + rad.shape)

    for ti, te in np.ndenumerate(time):
        for ri, re in np.ndenumerate(rad):
            res[ti + ri] = (
                Qw / (4.0 * np.pi * T) * exp1(re ** 2 * S / (4 * T * te))
            )

    if not struc_grid and grid_shape:
        res = np.copy(np.diag(res).reshape(grid_shape))

    # add the reference head
    res += hinf

    return res


def inc_gamma(s, x):
    r"""The (upper) incomplete gamma function

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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
