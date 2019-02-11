"""
Anaflow subpackage providing special flow solutions.

.. currentmodule:: anaflow.flow.special

The following functions are provided

.. autosummary::
   diskmodel
   grf_model
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from anaflow.tools.laplace import stehfest as sf
from anaflow.flow.laplace import lap_trans_flow_cyl, grf_laplace

__all__ = ["diskmodel", "grf_model"]


###############################################################################
# solution for a disk-model
###############################################################################


def diskmodel(
    rad,
    time,
    Tpart,
    Spart,
    Rpart,
    Qw,
    struc_grid=True,
    rwell=0.0,
    rinf=np.inf,
    hinf=0.0,
    stehfestn=12,
):
    """
    A diskmodel for transient flow

    A diskmodel for transient flow under a pumping condition
    in a confined aquifer. The solutions assumes concentric disks around the
    pumpingwell, where each disk has its own transmissivity and storativity
    value.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
    Tpart : :class:`numpy.ndarray`
        Given transmissivity values for each disk
    Spart : :class:`numpy.ndarray`
        Given storativity values for each disk
    Rpart : :class:`numpy.ndarray`
        Given radii separating the disks
    Qw : :class:`float`
        Pumpingrate at the well
    struc_grid : :class:`bool`, optional
        If this is set to ``False``, the `rad` and `time` array will be merged
        and interpreted as single, r-t points. In this case they need to have
        the same shapes. Otherwise a structured r-t grid is created.
        Default: ``True``
    rwell : :class:`float`, optional
        Inner radius of the pumping-well. Default: ``0.0``
    rinf : :class:`float`, optional
        Radius of the outer boundary of the aquifer. Default: ``np.inf``
    hinf : :class:`float`, optional
        Reference head at the outer boundary `rinf`. Default: ``0.0``
    stehfestn : :class:`int`, optional
        Since the solution is calculated in Laplace-space, the
        back-transformation is performed with the stehfest-algorithm.
        Here you can specify the number of interations within this
        algorithm. Default: ``12``

    Returns
    -------
    diskmodel : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    The parameters ``rad``, ``time``, ``Tpart`` and ``Spart`` will be checked
    for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Examples
    --------
    >>> diskmodel([1,2,3], [10, 100], [1e-3, 2e-3], [1e-3, 1e-3], [2], -1e-3)
    array([[-0.20312814, -0.09605675, -0.06636862],
           [-0.29785979, -0.18784251, -0.15582597]])
    """

    # ensure that input is treated as arrays
    rad = np.squeeze(rad)
    time = np.array(time).reshape(-1)
    Tpart = np.array(Tpart)
    Spart = np.array(Spart)
    Rpart = np.array(Rpart)

    if not struc_grid:
        grid_shape = rad.shape
        rad = rad.reshape(-1)

    # check the input
    if rwell < 0.0:
        raise ValueError("The wellradius needs to be >= 0")
    if rinf <= rwell:
        raise ValueError(
            "The upper boundary needs to be greater than the wellradius"
        )
    if not all(Rpart[i] < Rpart[i + 1] for i in range(len(Rpart) - 1)):
        raise ValueError("The radii of the zones need to be sorted")
    if np.any(Rpart <= rwell):
        raise ValueError(
            "The radii of the zones need to be greater than the wellradius"
        )
    if np.any(Rpart >= rinf):
        raise ValueError(
            "The radii of the zones need to be less than the outer radius"
        )
    if np.any(rad < rwell) or np.any(rad <= 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius"
        )
    if np.any(time <= 0.0):
        raise ValueError("The given times need to be >= 0")
    if not struc_grid and not rad.shape == time.shape:
        raise ValueError(
            "For unstructured grid the number of time- & radii-pts must equal"
        )
    if np.any(Tpart <= 0.0):
        raise ValueError("The Transmissivities need to be positiv")
    if np.any(Spart <= 0.0):
        raise ValueError("The Storages need to be positiv")
    if not isinstance(stehfestn, int):
        raise ValueError(
            "The boundary for the Stehfest-algorithm needs to be an integer"
        )
    if stehfestn <= 1:
        raise ValueError(
            "The boundary for the Stehfest-algorithm needs to be > 1"
        )
    if stehfestn % 2 != 0:
        raise ValueError(
            "The boundary for the Stehfest-algorithm needs to be even"
        )

    rpart = np.append(np.array([rwell]), Rpart)
    rpart = np.append(rpart, np.array([rinf]))

    # write the paramters in kwargs to use the stehfest-algorithm
    kwargs = {
        "rad": rad,
        "Qw": Qw,
        "rpart": rpart,
        "Spart": Spart,
        "Tpart": Tpart,
    }

    # call the stehfest-algorithm
    res = sf(lap_trans_flow_cyl, time, bound=stehfestn, **kwargs)

    # if the input are unstructured space-time points, return an array
    if not struc_grid and grid_shape:
        res = np.copy(np.diag(res).reshape(grid_shape))

    # add the reference head
    res += hinf

    return res


def grf_model(
    rad,
    time,
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
    stehfest_n=12,
):
    """
    The extended "General radial flow" model for transient flow

    The general radial flow (GRF) model by Barker introduces an arbitrary
    dimension for radial groundwater flow. We introduced the possibility to
    define radial dependet conductivity and storage values.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    time : :class:`numpy.ndarray`
        Array with all time-points where the function should be evaluated
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
    stehfest_n : :class:`int`, optional
        Since the solution is calculated in Laplace-space, the
        back-transformation is performed with the stehfest-algorithm.
        Here you can specify the number of interations within this
        algorithm. Default: ``12``

    Returns
    -------
    :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    The parameters ``rad``, ``time``, ``K_part`` and ``S_part`` will be checked
    for positivity.
    """

    # ensure that input is treated as arrays
    rad = np.squeeze(rad)
    time = np.array(time).reshape(-1)
    K_part = np.array(K_part)
    S_part = np.array(S_part)
    R_part = np.array(R_part)

    if not struc_grid:
        grid_shape = rad.shape
        rad = rad.reshape(-1)

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

    # call the stehfest-algorithm
    res = sf(grf_laplace, time, bound=stehfest_n, **kwargs)

    # if the input are unstructured space-time points, return an array
    if not struc_grid and grid_shape:
        res = np.copy(np.diag(res).reshape(grid_shape))

    # add the reference head
    res += head_bound

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
