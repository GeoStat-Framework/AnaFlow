# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing special flow solutions.

.. currentmodule:: anaflow.flow.special

The following functions are provided

.. autosummary::
   grf_disk
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from anaflow.tools.laplace import get_lap_inv
from anaflow.flow.laplace import grf_laplace
from anaflow.tools.special import Shaper

__all__ = ["grf_disk"]


def grf_disk(
    time,
    rad,
    K_part,
    S_part,
    R_part,
    Qw,
    dim=2,
    lat_ext=1.0,
    struc_grid=True,
    r_well=0.0,
    r_bound=np.inf,
    head_bound=0.0,
    lap_kwargs=None,
):
    """
    The extended "General radial flow" model for transient flow

    The general radial flow (GRF) model by Barker introduces an arbitrary
    dimension for radial groundwater flow. We introduced the possibility to
    define radial dependet conductivity and storage values.

    This solution is based on the grf model presented in [Barker88]_.

    Parameters
    ----------
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    K_part : :class:`numpy.ndarray`
        Given conductivity values for each disk
    S_part : :class:`numpy.ndarray`
        Given storativity values for each disk
    R_part : :class:`numpy.ndarray`
        Given radii separating the disks
    Qw : :class:`float`
        Pumpingrate at the well
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    r_well : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    r_bound : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    head_bound : :class:`float`, optional
        Reference head at the outer boundary `rinf`. Default: ``0.0``
    lap_kwargs : :class:`dict` or :any:`None` optional
        Dictionary for :any:`get_lap_inv` containing `method` and
        `method_dict`. The default is equivalent to
        ``lap_kwargs = {"method": "stehfest", "method_dict": None}``.
        Default: :any:`None`

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    References
    ----------
    .. [Barker88] Barker, J.
       ''A generalized radial flow model for hydraulic tests
       in fractured rock.'',
       Water Resources Research 24.10, 1796-1804, 1988
    """
    Input = Shaper(time, rad, struc_grid)
    lap_kwargs = {} if lap_kwargs is None else lap_kwargs

    K_part = np.array(K_part).reshape(-1)
    S_part = np.array(S_part).reshape(-1)
    R_part = np.array(R_part).reshape(-1)

    rpart = np.append(np.array([r_well]), R_part)
    rpart = np.append(rpart, np.array([r_bound]))

    # write the paramters in kwargs to use the stehfest-algorithm
    kwargs = {
        "rad": rad,
        "Qw": Qw,
        "rpart": rpart,
        "Spart": S_part,
        "Kpart": K_part,
        "dim": dim,
        "lat_ext": lat_ext,
    }
    kwargs.update(lap_kwargs)

    res = np.zeros((Input.time_no, Input.rad_no))
    # call the grf-model
    lap_inv = get_lap_inv(grf_laplace, **kwargs)
    res[Input.time > 0, :] = lap_inv(Input.time[Input.time > 0])
    res = Input.reshape(res)
    if Qw > 0:
        res = np.maximum(res, 0)
    else:
        res = np.minimum(res, 0)
    # add the reference head
    res += head_bound
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
