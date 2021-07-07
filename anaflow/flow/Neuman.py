# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing the Neuman equation for homogeneous aquifer.

.. currentmodule:: anaflow.flow.Neuman

The following functions are provided

.. autosummary::
   neuman_unconfined_laplace
   neuman_unconfined
"""
# pylint: disable=C0103
import numpy as np
from scipy.special import k0
from anaflow.tools.laplace import get_lap_inv
from anaflow.tools.special import Shaper
from scipy.optimize import fsolve
from scipy import optimize

__all__ = []


def eps_func(eps,value=0):
    return eps * np.tan(eps) - value


def interval_of_nth_root(n, v):
    """Correct interval."""
    a, b = n * np.pi, (n + 0.5) * np.pi - 0.001
    if v < 0:  # just for completeness (shift to negative branch)
        a += 0.5 * np.pi
        b += 0.5 * np.pi
    return a, b


def nth_root(n, v):
    """Get nth root of x*tan(x) = v."""
    a, b = interval_of_nth_root(n, v)
    if abs(v) > 100:
        return (n+0.5) * np.pi
    return optimize.bisect(eps_func, a, b, (v,))


def neuman_unconfined_laplace(
    s,
    rad,
    storage,
    transmissivity,
    rate,
    sat_thickness=52.0,
    screen_size=11.88,
    well_depth=12.6,
    kd=0.61,
    specific_yield=0.26,
    n_numbers=300,
):
    """
        The Neuman solution.


        Parameters
        ----------
    s : :class:`numpy.ndarray`
        Array with all "Laplace-space-points" where the function should
        be evaluated in the Laplace space.
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated.
    storage : :class:`float`
        Storage of the aquifer.
    transmissivity : :class:`float`
        Geometric-mean transmissivity.
    sat_thickness : :class:`float`, optional
        Saturated thickness of the aquifer.
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    screen_size : :class:`float`, optional
        Vertical length of the observation screen
    well_depth : :class:`float`, optional
        Vertical distance from initial water table to bottom
        of pumped well screen
    kd : :class:`float`, optional
        Dimensionless parameter for the conductivity.
        Kz/Kr : vertical conductivity divided by horizontal conductivity
    specific_yield: :class:`float`, optional
        specific yield
    """
    z = sat_thickness - well_depth
    d = well_depth - screen_size
    s = np.squeeze(s).reshape(-1)
    rad = np.squeeze(rad).reshape(-1)
    res = np.zeros(s.shape + rad.shape)

    for si, se in enumerate(s):
        for ri, re in enumerate(rad):
            if re < np.inf:
                rd = re / sat_thickness
                betaw = kd * (rd ** 2)
                righthand = se / (((storage/specific_yield) * betaw) + se / 1e9)
                roots = [nth_root(n, righthand) for n in range(n_numbers)]
                for i_n in range(len(roots)):
                    epsilon_n = roots[i_n]
                    xn = (betaw * (epsilon_n ** 2) + se) ** 0.5
                    res[si, ri] = (2 * k0(xn) * (np.sin(epsilon_n * (1 - (d / sat_thickness))) - np.sin(epsilon_n *
                                    (1 - (well_depth / sat_thickness)))) * np.cos(epsilon_n * (z / sat_thickness))) / \
                                  (se * ((well_depth - d) / sat_thickness) * (0.5 * epsilon_n + 0.25 * np.sin(2 * epsilon_n)))
    res *= rate / (2 * np.pi * transmissivity)
    return res


def neuman_unconfined(
    time,
    rad,
    storage,
    transmissivity,
    rate,
    h_bound=0.0,
    struc_grid=True,
    lap_kwargs=None,
):
    """Neuman solution form laplace solution."""
    Input = Shaper(time, rad, struc_grid)
    lap_kwargs = {} if lap_kwargs is None else lap_kwargs

    if not transmissivity > 0.0:
        raise ValueError("The Transmissivity needs to be positive.")
    if not storage > 0.0:
        raise ValueError("The Storage needs to be positive.")
    kwargs = {
        "rad": rad,
        "storage": storage,
        "transmissivity": transmissivity,
        "rate": rate,
    }
    kwargs.update(lap_kwargs)
    res = np.zeros((Input.time_no, Input.rad_no))
    lap_inv = get_lap_inv(neuman_unconfined_laplace, **kwargs)
    # call the laplace inverse function (only at time points > 0)
    res[Input.time_gz, :] = lap_inv(Input.time[Input.time_gz])
    # reshaping results
    res = Input.reshape(res)
    # add the reference head
    res += h_bound
    return res
