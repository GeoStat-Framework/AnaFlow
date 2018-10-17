"""
Anaflow subpackage providing flow solutions in homogeneous aquifers.

.. currentmodule:: anaflow.flow.homogeneous

Functions
---------
The following functions are provided

.. autosummary::
   thiem
   theis
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from anaflow.laplace import stehfest as sf
from anaflow.flow.helper import well_solution, inc_gamma
from anaflow.flow.laplace import lap_trans_flow_cyl, grf_laplace

__all__ = [
    "thiem",
    "theis",
]


###############################################################################
# Thiem-solution
###############################################################################

def thiem(rad, Rref,
          T, Qw,
          href=0.0):
    '''
    The Thiem solution

    The Thiem solution for steady-state flow under a pumping condition
    in a confined and homogeneous aquifer.
    This solution was presented in ''Thiem 1906''[R6]_.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    Rref : :class:`float`
        Reference radius with known head (see `href`)
    T : :class:`float`
        Given transmissivity of the aquifer
    Qw : :class:`float`
        Pumpingrate at the well
    href : :class:`float`, optional
        Reference head at the reference-radius `Rref`. Default: ``0.0``

    Returns
    -------
    thiem : :class:`numpy.ndarray`
        Array with all heads at the given radii.

    References
    ----------
    .. [R6] Thiem, G.,
       ''Hydrologische Methoden, J.M. Gebhardt'', Leipzig, 1906.

    Notes
    -----
    The parameters ``rad``, ``Rref`` and ``T`` will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Example
    -------
    >>> thiem([1,2,3], 10, 0.001, -0.001)
    array([-0.3664678 , -0.25615   , -0.19161822])
    '''

    rad = np.squeeze(rad)

    # check the input
    if Rref <= 0.0:
        raise ValueError(
            "The reference-radius needs to be greater than 0")
    if np.any(rad <= 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius")
    if T <= 0.0:
        raise ValueError(
            "The Transmissivity needs to be positiv")

    return -Qw/(2.0*np.pi*T)*np.log(rad/Rref) + href


###############################################################################
# Theis-solution
###############################################################################

def theis(rad, time,
          T, S, Qw,
          struc_grid=True, rwell=0.0, rinf=np.inf, hinf=0.0,
          stehfestn=12):
    '''
    The Theis solution

    The Theis solution for transient flow under a pumping condition
    in a confined and homogeneous aquifer.
    This solution was presented in ''Theis 1935''[Theis35]_.

    .. [Theis35] Theis, C.,
       ''The relation between the lowering of the piezometric surface and the
       rate and duration of discharge of a well using groundwater storage'',
       Trans. Am. Geophys. Union, 16, 519–524, 1935

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
        If `rwell` or `rinf` are not default, the solution is calculated in
        Laplace-space. The back-transformation is performed with the stehfest-
        algorithm. Here you can specify the number of interations within this
        algorithm. Default: ``12``

    Returns
    -------
    theis : :class:`numpy.ndarray`
        Array with all heads at the given radii and time-points.

    Notes
    -----
    The parameters ``rad``, ``Rref`` and ``T`` will be checked for positivity.
    If you want to use cartesian coordiantes, just use the formula
    ``r = sqrt(x**2 + y**2)``

    Example
    -------
    >>> theis([1,2,3], [10,100], 0.001, 0.001, -0.001)
    array([[-0.24959541, -0.14506368, -0.08971485],
           [-0.43105106, -0.32132823, -0.25778313]])
    '''

    # ensure that 'rad' and 'time' are arrays
    rad = np.squeeze(rad)
    time = np.array(time).reshape(-1)

    if not struc_grid:
        grid_shape = rad.shape
        rad = rad.reshape(-1)

    # check the input
    if rwell < 0.0:
        raise ValueError(
            "The wellradius needs to be >= 0")
    if rinf <= rwell:
        raise ValueError(
            "The upper boundary needs to be greater than the wellradius")
    if np.any(rad < rwell) or np.any(rad <= 0.0):
        raise ValueError(
            "The given radii need to be greater than the wellradius")
    if np.any(time <= 0.0):
        raise ValueError(
            "The given times need to be > 0")
    if not struc_grid and not rad.shape == time.shape:
        raise ValueError(
            "For unstructured grid the number of time- & radii-pts must equal")
    if T <= 0.0:
        raise ValueError(
            "The Transmissivity needs to be positiv")
    if S <= 0.0:
        raise ValueError(
            "The Storage needs to be positiv")
    if not isinstance(stehfestn, int):
        raise ValueError(
            "The boundary for the Stehfest-algorithm needs to be an integer")
    if stehfestn <= 1:
        raise ValueError(
            "The boundary for the Stehfest-algorithm needs to be > 1")
    if stehfestn % 2 != 0:
        raise ValueError(
            "The boundary for the Stehfest-algorithm needs to be even")

    if rwell == 0.0 and rinf == np.inf:
        res = well_solution(rad, time, T, S, Qw)

    else:
        rpart = np.array([rwell, rinf])
        Tpart = np.array([T])
        Spart = np.array([S])

        # write the paramters in kwargs to use the stehfest-algorithm
        kwargs = {"rad": rad,
                  "Qw": Qw,
                  "rpart": rpart,
                  "Spart": Spart,
                  "Tpart": Tpart}

        # call the stehfest-algorithm
        res = sf(lap_trans_flow_cyl, time, bound=stehfestn, **kwargs)

    # if the input are unstructured space-time points, return an array
    if not struc_grid and grid_shape:
        res = np.copy(np.diag(res).reshape(grid_shape))

    # add the reference head
    res += hinf

    return res


def grf_model(rad, time,
              K, S, Qw,
              dim=3, lat_ext=1.):
    # ensure that 'rad' and 'time' are arrays
    rad = np.squeeze(rad)
    time = np.array(time).reshape(-1)

    res = np.zeros(time.shape + rad.shape)

    nu = 1. - dim/2.

    for ti, te in np.ndenumerate(time):
        for ri, re in np.ndenumerate(rad):
            u = S*re**2/(4*K*te)
            res[ti+ri] = Qw/(4.0*np.pi**(1-nu)*K*lat_ext)*re**(2*nu)
            res[ti+ri] *= inc_gamma(-nu, u)
    return res


def grf_lap(rad, time,
            K, S, Qw, rpart=None,
            dim=3, lat_ext=1.):
    # ensure that 'rad' and 'time' are arrays
    rad = np.squeeze(rad)
    time = np.array(time).reshape(-1)
    if rpart is None:
        rpart = np.array([0, np.inf])
    Kpart = np.array(K, ndmin=1)
    Spart = np.array(S, ndmin=1)
    kwargs = {"rad": rad,
              "Qw": Qw,
              "rpart": rpart,
              "Spart": Spart,
              "Kpart": Kpart,
              "dim": dim,
              "lat_ext": lat_ext}

    # call the stehfest-algorithm
    res = sf(grf_laplace, time, bound=12, **kwargs)
    return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    print(theis([1,2,3], [10,100], 1e-4, 1e-4, 1e-4))
    print(grf_model([1,2,3], [10,100], 1e-4, 1e-4, 1e-4, dim=1.2))
    print(grf_lap([1,2,3], [10,100], 1e-4, 1e-4, 1e-4, dim=1.2))

