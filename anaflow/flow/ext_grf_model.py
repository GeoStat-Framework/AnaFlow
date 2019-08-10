# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing the extended GRF Model.

.. currentmodule:: anaflow.flow.ext_grf_model

The following functions are provided

.. autosummary::
   ext_grf
   ext_grf_steady
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.integrate import quad as integ

from anaflow.tools.laplace import get_lap_inv
from anaflow.flow.laplace import grf_laplace
from anaflow.tools.special import Shaper, sph_surf

__all__ = ["ext_grf", "ext_grf_steady"]


def ext_grf(
    time,
    rad,
    S_part,
    K_part,
    R_part,
    dim=2,
    lat_ext=1.0,
    rate=-1e-4,
    h_bound=0.0,
    K_well=None,
    struc_grid=True,
    lap_kwargs=None,
):
    """
    The extended "General radial flow" model for transient flow.

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
    S_part : :class:`numpy.ndarray`
        Given storativity values for each disk
    K_part : :class:`numpy.ndarray`
        Given conductivity values for each disk
    R_part : :class:`numpy.ndarray`
        Given radii separating the disks (including r_well and r_bound).
    dim : :class:`float`, optional
        Fractional dimension of the aquifer. Default: ``2.0``
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer. Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_bound : :class:`float`, optional
        Reference head at the outer boundary `R_part[-1]`. Default: ``0.0``
    K_well : :class:`float`, optional
        Conductivity at the well. Default: ``K_part[0]``
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
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
    # write the paramters in kwargs to use the stehfest-algorithm
    kwargs = {
        "rad": rad,
        "R_part": R_part,
        "S_part": S_part,
        "K_part": K_part,
        "dim": dim,
        "lat_ext": lat_ext,
        "rate": rate,
        "K_well": K_well,
    }
    kwargs.update(lap_kwargs)

    res = np.zeros((Input.time_no, Input.rad_no))
    # call the grf-model
    lap_inv = get_lap_inv(grf_laplace, **kwargs)
    res[Input.time_gz, :] = lap_inv(Input.time[Input.time_gz])
    res = Input.reshape(res)
    # add the reference head
    res += h_bound
    return res


def ext_grf_steady(
    rad,
    r_ref,
    conductivity,
    dim=2,
    lat_ext=1.0,
    rate=-1e-4,
    h_ref=0.0,
    arg_dict=None,
    **kwargs
):
    """
    The extended "General radial flow" model for steady flow.

    The general radial flow (GRF) model by Barker introduces an arbitrary
    dimension for radial groundwater flow. We introduced the possibility to
    define radial dependet conductivity.

    This solution is based on the grf model presented in [Barker88]_.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    r_ref : :class:`float`
        Radius of the reference head.
    conductivity : :class:`float` or :any:`callable`
        Conductivity. Either callable function taking kwargs or float.
    dim : :class:`float`, optional
        Fractional dimension of the aquifer. Default: ``2.0``
    lat_ext : :class:`float`, optional
        Lateral extend of the aquifer. Default: ``1.0``
    rate : :class:`float`, optional
        Pumpingrate at the well. Default: -1e-4
    h_ref : :class:`float`, optional
        Reference head at the reference-radius `r_ref`. Default: ``0.0``
    arg_dict : :class:`dict` or :any:`None`, optional
        Keyword-arguments given as a dictionary that are forwarded to the
        conductivity function. Will be merged with ``**kwargs``.
        This is designed for overlapping keywords in ``grf_steady`` and
        ``conductivity``.
        Default: ``None``
    **kwargs
        Keyword-arguments that are forwarded to the conductivity function.
        Will be merged with ``arg_dict``

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
    arg_dict = {} if arg_dict is None else arg_dict
    kwargs.update(arg_dict)
    Input = Shaper(rad=rad)
    q_fac = rate / (sph_surf(dim) * lat_ext ** (3.0 - dim))  # pumping factor
    if not r_ref > 0.0:
        raise ValueError("The reference radius needs to be positive.")
    if not Input.rad_min > 0.0:
        raise ValueError("The given radii need to be positive.")
    if not dim > 0.0 or dim > 3.0:
        raise ValueError("The dimension needs to be positiv and <= 3.")
    if not lat_ext > 0.0:
        raise ValueError("The lateral extend needs to be positiv.")

    if callable(conductivity):
        res = np.zeros(Input.rad_no)

        def integrand(val):
            """Integrand."""
            return val ** (1 - dim) / conductivity(val, **kwargs)

        for ri, re in enumerate(Input.rad):
            res[ri] = integ(integrand, re, r_ref)[0]
    else:
        con = float(conductivity)
        if not con > 0:
            raise ValueError("The Conductivity needs to be positive.")
        if np.isclose(dim, 2):
            res = np.log(r_ref / Input.rad) / con
        else:
            res = (
                (r_ref ** (2 - dim) - Input.rad ** (2 - dim)) / (2 - dim) / con
            )

    res = Input.reshape(res)
    # rescale by pumping rate
    res *= q_fac
    # add the reference head
    res += h_ref
    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
