# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing the Neuman equation for homogeneous aquifer.

.. currentmodule:: anaflow.flow.Neuman

The following functions are provided

.. autosummary::
    get_f_df
    nth_root
    neuman_unconfined_partially_penetrating_laplace
    neuman_unconfined_fully_penetrating_laplace
    neuman_unconfined
"""
# pylint: disable=C0103
import numpy as np
from scipy.special import k0
from anaflow.tools.laplace import get_lap_inv
from anaflow.tools.special import Shaper
from scipy.optimize import root
from scipy.optimize import root_scalar

__all__ = []

def get_f_df_df2(value=0):
    """Get target function and its first two derivatives."""
    if value < 0:
        raise ValueError("Neuman: epsilon for root finding needs to be >0.")

    def _f_df_df2(x):
        return (
            np.subtract(np.multiply(x, np.tan(x)), value),
            np.tan(x) + np.divide(x, np.cos(x) ** 2),
            2 * (np.multiply(x, np.tan(x)) + 1.0) * np.cos(x) ** -2,
        )

    return _f_df_df2

def nth_root(n, v):
    """Get n-th root of x*tan(x) = v."""
    x0 = np.sqrt(v) if (v < 1 and n < 1) else np.arctan(v) + n * np.pi
    f = get_f_df_df2(v)
    sol = root_scalar(f, x0=x0, fprime=True, fprime2=True)
    if not sol.converged:
        raise ValueError(f"Neuman: couldn't find {n}-th root for eps={v}")
    return sol.root

def neuman_unconfined_partially_penetrating_laplace(
    s,
    rad,
    storage,
    transmissivity,
    rate,
    sat_thickness=49,
    screen_size=11.88,
    well_depth=12.6,
    kd=0.61,
    specific_yield=0.26,
    n_numbers=25,
):
    """
    Neuman solution for unconfined aquifers in Laplace space.

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
        Specific yield
    """
    z = sat_thickness - well_depth
    d = well_depth - screen_size
    s = np.squeeze(s).reshape(-1)
    rad = np.squeeze(rad).reshape(-1)
    res = np.zeros(s.shape + rad.shape)

    for si, se in enumerate(s):
        for ri, re in enumerate(rad):
            if re == np.inf:
                continue
            rd = re / sat_thickness
            beta = kd * (rd ** 2)
            rhs = se / (((storage / specific_yield) * beta) + se / 1e9)
            roots = [nth_root(n, rhs) for n in range(n_numbers)]
            for eps in roots:
                xn = (beta * (eps ** 2) + se) ** 0.5
                res[si, ri] += (
                    2
                    * k0(xn)
                    * (
                        np.sin(eps * (1 - (d / sat_thickness)))
                        - np.sin(eps * (1 - (well_depth / sat_thickness)))
                    )
                    * np.cos(eps * (z / sat_thickness))
                ) / (
                    se
                    * ((well_depth - d) / sat_thickness)
                    * (0.5 * eps + 0.25 * np.sin(2 * eps))
                )
    return res * rate / (2 * np.pi * transmissivity)


def neuman_unconfined_fully_penetrating_laplace(
    s,
    rad,
    storage,
    transmissivity,
    rate,
    sat_thickness=49,
    specific_yield=0.26,
    n_numbers=25,
):
    """
    Neuman solution for unconfined aquifers in Laplace space.

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
    rate : :class:`float`, optional
        Pumpingrate at the well.
    sat_thickness : :class:`float`, optional
        Saturated thickness of the aquifer.
    kd : :class:`float`, optional
        Dimensionless parameter for the conductivity.
        Kz/Kr : vertical conductivity divided by horizontal conductivity
    specific_yield: :class:`float`, optional
        Specific yield
    """
    kr = transmissivity / sat_thickness
    kz = kr * 0.001
    s = np.squeeze(s).reshape(-1)
    rad = np.squeeze(rad).reshape(-1)
    res = np.zeros(s.shape + rad.shape)

    for si, se in enumerate(s):
        for ri, re in enumerate(rad):
            if re == np.inf:
                continue
            rd = re / sat_thickness
            beta = kz * (rd ** 2) / kr
            rhs = se / (
                ((storage / specific_yield) * beta)
                + se / ((1e9 * sat_thickness * specific_yield) / kz)
            )
            roots = [nth_root(n, rhs) for n in range(n_numbers)]
            for eps in roots:
                xn = (beta * (eps ** 2) + se) ** 0.5
                res[si, ri] += (2 * k0(xn) * (np.sin(eps) ** 2)) / (
                    se * eps * (0.5 * eps + 0.25 * np.sin(2 * eps))
                )
    return res * rate / (2 * np.pi * transmissivity)


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
    lap_inv = get_lap_inv(
        neuman_unconfined_partially_penetrating_laplace, **kwargs
    )
    # call the laplace inverse function (only at time points > 0)
    res[Input.time_gz, :] = lap_inv(Input.time[Input.time_gz])
    # reshaping results
    res = Input.reshape(res)
    # add the reference head
    res += h_bound
    return res
