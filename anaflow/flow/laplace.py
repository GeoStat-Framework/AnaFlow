# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing flow solutions in laplace space.

.. currentmodule:: anaflow.flow.laplace

The following functions are provided

.. autosummary::
   grf_laplace
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
from scipy.special import kv, iv, gamma, erfcx
from pentapy import solve
from anaflow.tools.special import sph_surf

__all__ = ["grf_laplace"]


def constant(s):
    """Constant pumping."""
    return 1.0 / s


def periodic(s, a=0):
    """
    Periodic pumping.

    Q(t) = Q * cos(a * t)
    """
    if np.isclose(a, 0):
        return constant(s)
    return 1.0 / (s + a ** 2 / s)


def slug(s):
    """Slug test."""
    return np.ones_like(s)


def interval(s, a=np.inf):
    """Interval pumping in [0, t]."""
    if np.isposinf(a):
        return constant(s)
    return (1.0 - np.exp(-s * a)) / s


def accruing(s, a=0):
    """Accruing pumping with time scale t."""
    return erfcx((s * a) / 2.0) / s


PUMP_COND = {0: constant, 1: periodic, 2: slug, 3: interval, 4: accruing}


def grf_laplace(
    s,
    rad=None,
    S_part=None,
    K_part=None,
    R_part=None,
    dim=2,
    lat_ext=1.0,
    rate=None,
    K_well=None,
    cut_off_prec=1e-20,
    cond=0,
    cond_kw=None,
):
    """
    The extended GRF-model for transient flow in Laplace-space.

    The General Radial Flow (GRF) Model allowes fractured dimensions for
    transient flow under a pumping condition in a confined aquifer.
    The solutions assumes concentric annuli around the pumpingwell,
    where each annulus has its own conductivity and storativity value.

    Parameters
    ----------
    s : :class:`numpy.ndarray`
        Array with all Laplace-space-points
        where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    S_part : :class:`numpy.ndarray` of length N
        Given storativity values for each disk
    K_part : :class:`numpy.ndarray` of length N
        Given conductivity values for each disk
    R_part : :class:`numpy.ndarray` of length N+1
        Given radii separating the disks as well as starting- and endpoints
    dim : :class:`float`
        Flow dimension. Default: 3
    lat_ext : :class:`float`
        The lateral extend of the flow-domain, used in `L^(3-dim)`. Default: 1
    rate : :class:`float`
        Pumpingrate at the well
    K_well : :class:`float`, optional
        Conductivity at the well. Default: ``K_part[0]``
    cut_off_prec : :class:`float`, optional
        Define a cut-off precision for the calculation to select the disks
        included in the calculation. Default ``1e-20``
    cond : :class:`int`, optional
        Type of the pumping condition:

            * 0 : constant
            * 1 : periodic (needs "w" as cond_kw)
            * 2 : slug (rate will be interpreted as slug-volume)
            * 3 : interval (needs "t" as cond_kw)
            * callable: laplace-transformation of the transient pumping-rate

        Default: 0
    cond_kw : :class:`dict` optional
        Keyword args for the pumping condition. Default: None

    Returns
    -------
    grf_laplace : :class:`numpy.ndarray`
        Array with all values in laplace-space

    Examples
    --------
    >>> grf_laplace([5,10],[1,2,3],[1e-3,1e-3],[1e-3,2e-3],[0,2,10], 2, 1, -1)
    array([[-2.71359196e+00, -1.66671965e-01, -2.82986917e-02],
           [-4.58447458e-01, -1.12056319e-02, -9.85673855e-04]])
    """
    cond_kw = {} if cond_kw is None else cond_kw
    cond = cond if callable(cond) else PUMP_COND[cond]
    # ensure that input is treated as arrays
    s = np.squeeze(s).reshape(-1)
    rad = np.squeeze(rad).reshape(-1)
    S_part = np.squeeze(S_part).reshape(-1)
    K_part = np.squeeze(K_part).reshape(-1)
    R_part = np.squeeze(R_part).reshape(-1)
    # the dimension is used by nu in the literature (See Barker 88)
    dim = float(dim)
    nu = 1.0 - dim / 2.0
    nu1 = nu - 1
    # the lateral extend is a bit subtle in fractured dimension
    lat_ext = float(lat_ext)
    rate = float(rate)
    # get the number of partitions
    parts = len(K_part)
    # set the conductivity at the well
    K_well = K_part[0] if K_well is None else float(K_well)
    # check the input
    if not len(R_part) - 1 == len(S_part) == len(K_part) > 0:
        raise ValueError("R_part, S_part and K_part need matching lengths.")
    if R_part[0] < 0.0:
        raise ValueError("The wellradius needs to be >= 0.")
    if not all([r1 < r2 for r1, r2 in zip(R_part[:-1], R_part[1:])]):
        raise ValueError("The radii values need to be sorted.")
    if not np.min(rad) > R_part[0] or np.max(rad) > R_part[-1]:
        raise ValueError("The given radii need to be in the given range.")
    if not all([con > 0 for con in K_part]):
        raise ValueError("The Conductivity needs to be positiv.")
    if not all([stor > 0 for stor in S_part]):
        raise ValueError("The Storage needs to be positiv.")
    if not dim > 0.0 or dim > 3.0:
        raise ValueError("The dimension needs to be positiv and <= 3.")
    if not lat_ext > 0.0:
        raise ValueError("The lateral extend needs to be positiv.")
    if not K_well > 0:
        raise ValueError("The well conductivity needs to be positiv.")

    # initialize the result
    res = np.zeros(s.shape + rad.shape)
    # the first sqrt of the diffusivity values
    diff_sr0 = np.sqrt(S_part[0] / K_part[0])
    # set the general pumping-condtion depending on the well-radius
    if R_part[0] > 0.0:
        Qs = -s ** (-0.5) / diff_sr0 * R_part[0] ** nu1 * cond(s, **cond_kw)
    else:
        Qs = -(2 / diff_sr0) ** nu * s ** (-nu / 2) * cond(s, **cond_kw)

    # if there is a homgeneouse aquifer, compute the result by hand
    if parts == 1:
        # initialize the equation system
        V = np.zeros(2, dtype=float)
        M = np.array([[-gamma(1 - nu), 2.0 / gamma(nu)], [0, 1]])

        for si, se in enumerate(s):
            Cs = np.sqrt(se) * diff_sr0
            # set the pumping-condition at the well
            V[0] = Qs[si]
            # incorporate the boundary-conditions
            if R_part[0] > 0.0:
                M[0, :] = [-kv(nu1, Cs * R_part[0]), iv(nu1, Cs * R_part[0])]
            if R_part[-1] < np.inf:
                M[1, :] = [kv(nu, Cs * R_part[-1]), iv(nu, Cs * R_part[-1])]
            else:
                M[0, 1] = 0  # Bs is 0 in this case either way
            # solve the equation system
            As, Bs = np.linalg.solve(M, V)

            # calculate the head
            for ri, re in enumerate(rad):
                if re < R_part[-1]:
                    res[si, ri] = re ** nu * (
                        As * kv(nu, Cs * re) + Bs * iv(nu, Cs * re)
                    )

    # if there is more than one partition, create an equation system
    else:
        # initialize LHS and RHS for the linear equation system
        # Mb is the banded matrix for the Eq-System
        V = np.zeros(2 * parts)
        Mb = np.zeros((5, 2 * parts))
        X = np.zeros(2 * parts)
        # set the standard boundary conditions for rwell=0.0 and rinf=np.inf
        Mb[2, 0] = -gamma(1 - nu)
        Mb[1, 1] = 2.0 / gamma(nu)
        Mb[2, -1] = 1.0

        # calculate the consecutive fractions of the conductivities
        Kfrac = K_part[:-1] / K_part[1:]
        # calculate the square-root of the diffusivities
        difsr = np.sqrt(S_part / K_part)
        # calculate a temporal substitution (factor from mass-conservation)
        tmp = Kfrac * difsr[:-1] / difsr[1:]
        # match the radii to the different disks
        pos = np.searchsorted(R_part, rad) - 1

        # iterate over the laplace-variable
        for si, se in enumerate(s):
            Cs = np.sqrt(se) * difsr

            # set the pumping-condition at the well
            # --> implement other pumping conditions
            V[0] = Qs[si]

            # generate the equation system as banded matrix
            for i in range(parts - 1):
                Mb[0, 2 * i + 3] = -iv(nu, Cs[i + 1] * R_part[i + 1])
                Mb[1, 2 * i + 2 : 2 * i + 4] = [
                    -kv(nu, Cs[i + 1] * R_part[i + 1]),
                    -iv(nu1, Cs[i + 1] * R_part[i + 1]),
                ]
                Mb[2, 2 * i + 1 : 2 * i + 3] = [
                    iv(nu, Cs[i] * R_part[i + 1]),
                    kv(nu1, Cs[i + 1] * R_part[i + 1]),
                ]
                Mb[3, 2 * i : 2 * i + 2] = [
                    kv(nu, Cs[i] * R_part[i + 1]),
                    tmp[i] * iv(nu1, Cs[i] * R_part[i + 1]),
                ]
                Mb[4, 2 * i] = -tmp[i] * kv(nu1, Cs[i] * R_part[i + 1])

            # set the boundary-conditions if needed
            if R_part[0] > 0.0:
                Mb[2, 0] = -kv(nu1, Cs[0] * R_part[0])
                Mb[1, 1] = iv(nu1, Cs[0] * R_part[0])
            if R_part[-1] < np.inf:
                Mb[-2, -2] = kv(nu, Cs[-1] * R_part[-1])
                Mb[2, -1] = iv(nu, Cs[-1] * R_part[-1])
            else:  # erase the last row, since X[-1] will be 0
                Mb[0, -1] = 0
                Mb[1, -1] = 0

            # find first disk which has no impact
            Mb_cond = np.max(np.abs(Mb), axis=0)
            Mb_cond_lo = Mb_cond < cut_off_prec
            Mb_cond_hi = Mb_cond > 1 / cut_off_prec
            Mb_cond = np.logical_or(Mb_cond_lo, Mb_cond_hi)
            cond = np.where(Mb_cond)[0]
            found = cond.shape[0] > 0
            first = cond[0] // 2 if found else parts

            # initialize coefficients
            X[2 * first :] = 0.0
            # only the first disk has an impact
            if first <= 1:
                M_sgl = np.eye(2, dtype=float)
                M_sgl[:, 0] = Mb[2:4, 0]
                M_sgl[:, 1] = Mb[1:3, 1]
                # solve the equation system
                try:
                    X[:2] = np.linalg.solve(M_sgl, V[:2])
                except np.linalg.LinAlgError:
                    # set 0 if matrix singular
                    X[:2] = 0
            elif first > 1:
                # shrink the matrix
                M_sgl = Mb[:, : 2 * first]
                if first < parts:
                    M_sgl[-1, -1] = 0
                    M_sgl[-2, -1] = 0
                    M_sgl[-1, -2] = 0
                X[: 2 * first] = solve(
                    M_sgl, V[: 2 * first], is_flat=True, index_row_wise=False
                )
            np.nan_to_num(X, copy=False)

            # calculate the head (ignore small values)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                k0_sub = X[2 * pos] * kv(nu, Cs[pos] * rad)
                k0_sub[np.abs(X[2 * pos]) < cut_off_prec] = 0
                i0_sub = X[2 * pos + 1] * iv(nu, Cs[pos] * rad)
                i0_sub[np.abs(X[2 * pos + 1]) < cut_off_prec] = 0
                res[si, :] = rad ** nu * (k0_sub + i0_sub)

    # set problematic values to 0
    # --> the algorithm tends to violate small values,
    #     therefore this approach is suitable
    np.nan_to_num(res, copy=False)
    # scale to pumpingrate
    res *= rate / (K_well * sph_surf(dim) * lat_ext ** (3.0 - dim))

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
