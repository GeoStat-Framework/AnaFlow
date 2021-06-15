# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing the Neuman equation for homogeneous aquifer.

.. currentmodule:: anaflow.flow.Neuman

The following functions are provided

.. autosummary::

"""
# pylint: disable=C0103
import numpy as np
from scipy.special import k0, k1
from scipy.integrate import quad as integ

import matplotlib.pyplot as plt

from anaflow.tools.laplace import get_lap_inv
from anaflow.flow.laplace import grf_laplace
from anaflow.tools.special import Shaper, sph_surf

__all__ = []


def neuman(
        s,
        rad,
        storage,
        transmissivity,
        sat_thickness=52.0,
        rate=-1e-4,
        d=0.6,
        l=12.6,
        rw=0.1,
        kz=0.14,
        kr=0.23,
        ss=4.27e-5,
        n_numbers=52,
):
    """
        The Neuman solution.


        Parameters
        ----------
    s : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated in the Laplace space.
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
    d : :class:`float`, optional
        Vertical distance from initial water table to top of
        pumped well screen
    l : :class:`float`, optional
        Vertical distance from initial water table to bottom
        of pumped well screen
    rw : :class:`float`, optional
        Outside radius of the pumped well screen
    kz : :class:`float`, optional
        Hydraulic conductivity in the vertical direction
    kr : :class:`float`, optional
        Hydraulic conductivity in the horizontal direction
    Ss : :class:`float`, optional
        specific storage
    """
    z = float(sat_thickness - l)
    kd = kz / kr
    s = np.squeeze(s).reshape(-1)
    rad = np.squeeze(rad).reshape(-1)
    res = np.zeros(s.shape + rad.shape)

    for si, se in enumerate(s):
        for ri, re in enumerate(rad):
            if re < np.inf:
                for n in range(n_numbers):
                    epsilon_n = np.pi / 2 + n * np.pi
                    betaw = kd * rw ** 2
                    qn = ((epsilon_n ** 2) * betaw + se) ** 0.5
                    rd = re / rw
                    E = k0(qn * rd) * np.cos(epsilon_n * (z / sat_thickness)) / (
                            (
                                    np.sin(epsilon_n * (1 - (d / sat_thickness)))
                                    - np.sin(epsilon_n * (1 - (l / sat_thickness)))
                            )
                            / (qn * k1(qn) * (epsilon_n + 0.5 * np.sin(2 * epsilon_n)))
                        )
                    A = k0(qn) * (
                            (
                                    np.sin(epsilon_n * (1 - (d / sat_thickness)))
                                    - np.sin(epsilon_n * (1 - (l / sat_thickness)))
                            )
                            ** 2
                            / (
                                    epsilon_n
                                    * qn
                                    * k1(qn)
                                    * (epsilon_n + 0.5 * np.sin(2 * epsilon_n))
                            )
                    )
                E = 2 * E
                A = (2 / ((l - d) / sat_thickness)) * A
                sw = storage
                wd = np.pi * rw ** 2 / (2 * np.pi * rw ** 2 * ss * (l - d))
                res[si, ri] = (2 * E) / (se * ((l - d) / sat_thickness) * (1 + wd * se * (A + sw)))
    res *= rate / (2 * np.pi * transmissivity)
    return res


def neuman_from_laplace(
        time,
        rad,
        storage,
        transmissivity,
        rate=-1e-4,
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
    lap_inv = get_lap_inv(neuman, **kwargs)
    # call the laplace inverse function (only at time points > 0)
    res[Input.time_gz, :] = lap_inv(Input.time[Input.time_gz])
    # reshaping results
    res = Input.reshape(res)
    # add the reference head
    res += h_bound
    return res


time = [10, 600, 36000]  # 10s, 10min, 10h
rad = np.geomspace(0.1, 10)
# default parameters
T = 1e-4
S = 1e-4
Q = -1e-4

neu = neuman_from_laplace(time, rad, storage=S, transmissivity=T)
for i in range(len(time)):
    plt.plot(rad, neu[i, :], color="C0", linestyle=":", linewidth=4)
