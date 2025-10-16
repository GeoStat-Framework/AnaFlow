"""
Anaflow subpackage providing flow solutions in laplace space.

.. currentmodule:: anaflow.flow.laplace

The following functions are provided

.. autosummary::
   :toctree:

   grf_laplace
"""

# pylint: disable=C0103,R0915
import numpy as np
from scipy.special import erfcx, gamma

from anaflow.tools.special import sph_surf

try:
    from ._laplace_accel import solve_homogeneous as _solve_homogeneous
    from ._laplace_accel import solve_multilayer as _solve_multilayer
except ImportError as exc:  # pragma: no cover - extension is mandatory
    raise ImportError(
        "anaflow.flow._laplace_accel extension missing. "
        "Please build the Cython module (e.g. `python setup.py build_ext --inplace`)."
    ) from exc

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
    return 1.0 / (s + a**2 / s)


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
    pump_cond = cond if callable(cond) else PUMP_COND[cond]

    # ensure that input is treated as contiguous arrays
    s = np.ascontiguousarray(np.atleast_1d(np.asarray(s, dtype=np.float64)))
    rad = np.ascontiguousarray(np.atleast_1d(np.asarray(rad, dtype=np.float64)))
    S_part = np.ascontiguousarray(np.atleast_1d(np.asarray(S_part, dtype=np.float64)))
    K_part = np.ascontiguousarray(np.atleast_1d(np.asarray(K_part, dtype=np.float64)))
    R_part = np.ascontiguousarray(np.atleast_1d(np.asarray(R_part, dtype=np.float64)))

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
    if not all(r1 < r2 for r1, r2 in zip(R_part[:-1], R_part[1:])):
        raise ValueError("The radii values need to be sorted.")
    if not np.min(rad) > R_part[0] or np.max(rad) > R_part[-1]:
        raise ValueError("The given radii need to be in the given range.")
    if not all(con > 0 for con in K_part):
        raise ValueError("The Conductivity needs to be positiv.")
    if not all(stor > 0 for stor in S_part):
        raise ValueError("The Storage needs to be positiv.")
    if dim <= 0.0 or dim > 3.0:
        raise ValueError("The dimension needs to be positiv and <= 3.")
    if lat_ext <= 0.0:
        raise ValueError("The lateral extend needs to be positiv.")
    if K_well <= 0:
        raise ValueError("The well conductivity needs to be positiv.")

    cut_off_prec = float(cut_off_prec)

    # pre-compute helper arrays
    res = np.zeros((s.size, rad.size), dtype=np.float64)
    diff_sr0 = float(np.sqrt(S_part[0] / K_part[0]))
    cond_vals = np.asarray(pump_cond(s, **cond_kw), dtype=np.float64)
    if cond_vals.shape != s.shape:
        cond_vals = np.broadcast_to(cond_vals, s.shape).astype(np.float64, copy=True)
    cond_vals = np.ascontiguousarray(cond_vals, dtype=np.float64)

    if R_part[0] > 0.0:
        qs = -np.power(s, -0.5) / diff_sr0 * R_part[0] ** nu1 * cond_vals
    else:
        qs = -np.power(2.0 / diff_sr0, nu) * np.power(s, -nu / 2.0) * cond_vals
    qs = np.ascontiguousarray(qs, dtype=np.float64)

    difsr = np.ascontiguousarray(np.sqrt(S_part / K_part), dtype=np.float64)
    rad_pow = np.ascontiguousarray(np.power(rad, nu), dtype=np.float64)
    gamma_1_minus_nu = float(gamma(1.0 - nu))
    two_over_gamma_nu = 2.0 / float(gamma(nu))

    if parts == 1:
        _solve_homogeneous(
            s,
            rad,
            rad_pow,
            float(R_part[0]),
            float(R_part[-1]),
            diff_sr0,
            nu,
            nu1,
            gamma_1_minus_nu,
            two_over_gamma_nu,
            qs,
            cut_off_prec,
            res,
        )
    else:
        tmp = np.ascontiguousarray(
            (K_part[:-1] / K_part[1:]) * (difsr[:-1] / difsr[1:]),
            dtype=np.float64,
        )
        pos = np.ascontiguousarray(
            np.searchsorted(R_part, rad, side="left") - 1, dtype=np.intp
        )
        _solve_multilayer(
            s,
            rad,
            rad_pow,
            R_part,
            difsr,
            tmp,
            pos,
            nu,
            nu1,
            gamma_1_minus_nu,
            two_over_gamma_nu,
            qs,
            cut_off_prec,
            res,
        )

    np.nan_to_num(res, copy=False)
    res *= rate / (K_well * sph_surf(dim) * lat_ext ** (3.0 - dim))
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
