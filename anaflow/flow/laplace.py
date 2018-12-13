"""
Anaflow subpackage providing flow solutions in laplace space.

.. currentmodule:: anaflow.flow.laplace

Functions
---------
The following functions are provided

.. autosummary::
   lap_transgwflow_cyl
"""

from __future__ import absolute_import, division, print_function

import warnings

import numpy as np
import scipy.sparse as sps
from scipy.sparse.linalg import spsolve
from scipy.special import i0, i1, k0, k1, gamma, kv, iv

# ignore Umpfack warnings for almost singular matrices
try:
    # first look if the umfpack is available
    from scikits.umfpack import UmfpackWarning

    SlvWarn = UmfpackWarning
except ImportError:
    # if umfpack is not present, use scipy.sparse MatrixRankWarning
    SlvWarn = sps.linalg.MatrixRankWarning

__all__ = ["lap_trans_flow_cyl"]


def sph_surf(dim):
    """surface of the sphere"""
    return 2.0 * np.sqrt(np.pi) ** dim / gamma(dim / 2.0)


###############################################################################
# The generic solver of the 2D radial transient groundwaterflow equation
# in Laplace-space with a pumping condition and a fix zero boundary-head
###############################################################################


def lap_trans_flow_cyl(
    s, rad=None, rpart=None, Spart=None, Tpart=None, Qw=None, Twell=None
):
    """
    A diskmodel for transient flow in Laplace-space

    The solution of the diskmodel for transient flow under a pumping condition
    in a confined aquifer in Laplace-space.
    The solutions assumes concentric disks around the pumpingwell,
    where each disk has its own transmissivity and storativity value.

    Parameters
    ----------
    s : :class:`numpy.ndarray`
        Array with all Laplace-space-points
        where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    rpart : :class:`numpy.ndarray`
        Given radii separating the disks as well as starting- and endpoints
    Tpart : :class:`numpy.ndarray`
        Given transmissivity values for each disk
    Spart : :class:`numpy.ndarray`
        Given storativity values for each disk
    Qw : :class:`float`
        Pumpingrate at the well
    Twell : :class:`float`, optional
        Transmissivity at the well. Default: ``Tpart[0]``

    Returns
    -------
    lap_transgwflow_cyl : :class:`numpy.ndarray`
        Array with all values in laplace-space

    Example
    -------
    >>> lap_transgwflow_cyl([5,10],[1,2,3],[0,2,10],[1e-3,1e-3],[1e-3,2e-3],-1)
    array([[ -2.71359196e+00,  -1.66671965e-01,  -2.82986917e-02],
           [ -4.58447458e-01,  -1.12056319e-02,  -9.85673855e-04]])
    """

    # ensure that input is treated as arrays
    s = np.squeeze(s).reshape(-1)
    rad = np.squeeze(rad).reshape(-1)
    rpart = np.squeeze(rpart).reshape(-1)
    Spart = np.squeeze(Spart).reshape(-1)
    Tpart = np.squeeze(Tpart).reshape(-1)

    # get the number of partitions
    parts = len(Tpart)

    # initialize the result
    res = np.zeros(s.shape + rad.shape)

    # set the general pumping-condtion
    if Twell is None:
        Twell = Tpart[0]
    Q = Qw / (2.0 * np.pi * Twell)

    # if there is a homgeneouse aquifer, compute the result by hand
    if parts == 1:
        # calculate the square-root of the diffusivities
        difsr = np.sqrt(Spart[0] / Tpart[0])

        for si, se in np.ndenumerate(s):
            Cs = np.sqrt(se) * difsr

            # set the pumping-condition at the well
            Qs = Q / se

            # incorporate the boundary-conditions
            if rpart[0] == 0.0:
                Bs = Qs
                if rpart[-1] == np.inf:
                    As = 0.0
                else:
                    As = -Qs * k0(Cs * rpart[-1]) / i0(Cs * rpart[-1])

            else:
                if rpart[-1] == np.inf:
                    As = 0.0
                    Bs = Qs / (Cs * rpart[0] * k1(Cs * rpart[0]))
                else:
                    det = i1(Cs * rpart[0]) * k0(Cs * rpart[-1]) + k1(
                        Cs * rpart[0]
                    ) * i0(Cs * rpart[-1])
                    As = -Qs / (Cs * rpart[0]) * k0(Cs * rpart[-1]) / det
                    Bs = Qs / (Cs * rpart[0]) * i0(Cs * rpart[-1]) / det

            # calculate the head
            for ri, re in np.ndenumerate(rad):
                if re < rpart[-1]:
                    res[si + ri] = As * i0(Cs * re) + Bs * k0(Cs * re)

    # if there is more than one partition, create an equation system
    else:
        # initialize LHS and RHS for the linear equation system
        # Mb is the banded matrix for the Eq-System
        V = np.zeros(2 * (parts))
        Mb = np.zeros((5, 2 * (parts)))
        # the positions of the diagonals of the matrix set in Mb
        diagpos = [2, 1, 0, -1, -2]
        # set the standard boundary conditions for rwell=0.0 and rinf=np.inf
        Mb[2, 0] = 1.0
        Mb[-3, -1] = 1.0

        # calculate the consecutive fractions of the transmissivities
        Tfrac = Tpart[:-1] / Tpart[1:]

        # calculate the square-root of the diffusivities
        difsr = np.sqrt(Spart / Tpart)

        # calculate a temporal substitution
        tmp = Tfrac * difsr[:-1] / difsr[1:]

        # match the radii to the different disks
        pos = np.searchsorted(rpart, rad) - 1

        # iterate over the laplace-variable
        for si, se in enumerate(s):
            Cs = np.sqrt(se) * difsr

            # set the pumping-condition at the well
            # --> implement other pumping conditions
            V[0] = Q / se

            # set the boundary-conditions if needed
            if rpart[0] > 0.0:
                Mb[2, 0] = Cs[0] * rpart[0] * k1(Cs[0] * rpart[0])
                Mb[1, 1] = -Cs[0] * rpart[0] * i1(Cs[0] * rpart[0])
            if rpart[-1] < np.inf:
                Mb[-2, -2] = k0(Cs[-1] * rpart[-1])
                Mb[-3, -1] = i0(Cs[-1] * rpart[-1])

            # generate the equation system as banded matrix
            for i in range(parts - 1):
                Mb[0, 2 * i + 3] = -i0(Cs[i + 1] * rpart[i + 1])
                Mb[1, 2 * i + 2 : 2 * i + 4] = [
                    -k0(Cs[i + 1] * rpart[i + 1]),
                    -i1(Cs[i + 1] * rpart[i + 1]),
                ]
                Mb[2, 2 * i + 1 : 2 * i + 3] = [
                    i0(Cs[i] * rpart[i + 1]),
                    k1(Cs[i + 1] * rpart[i + 1]),
                ]
                Mb[3, 2 * i : 2 * i + 2] = [
                    k0(Cs[i] * rpart[i + 1]),
                    tmp[i] * i1(Cs[i] * rpart[i + 1]),
                ]
                Mb[4, 2 * i] = -tmp[i] * k1(Cs[i] * rpart[i + 1])

            # genearate the cooeficient matrix as a spare matrix
            M = sps.spdiags(Mb, diagpos, 2 * parts, 2 * parts, format="csc")

            # solve the Eq-Sys and ignore errors from the umf-pack
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore")
                warnings.simplefilter("ignore", SlvWarn)
                warnings.simplefilter("ignore", RuntimeWarning)
                X = spsolve(M, V, use_umfpack=True)

            # to suppress numerical errors, set NAN values to 0
            X[np.logical_not(np.isfinite(X))] = 0.0

            # calculate the head
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res[si, :] = X[2 * pos] * k0(Cs[pos] * rad) + X[
                    2 * pos + 1
                ] * i0(Cs[pos] * rad)

        # set problematic values to 0
        # --> the algorithm tends to violate small values,
        #     therefore this approachu is suitable
        res[np.logical_not(np.isfinite(res))] = 0.0

    return res


###############################################################################
# The generic solver of the 2D radial transient groundwaterflow equation
# in Laplace-space with a pumping condition and a fix zero boundary-head
###############################################################################


def grf_laplace(
    s,
    rad=None,
    dim=3,
    lat_ext=1.0,
    rpart=None,
    Spart=None,
    Kpart=None,
    Qw=None,
    Kwell=None,
):
    """
    A modified GRF-model for transient flow in Laplace-space

    The General Radial Flow (GRF) Model allowes fractured dimensions for
    transient flow under a pumping condition
    in a confined aquifer in Laplace-space.
    The solutions assumes concentric disks around the pumpingwell,
    where each disk has its own conductivity and storativity value.

    Parameters
    ----------
    s : :class:`numpy.ndarray`
        Array with all Laplace-space-points
        where the function should be evaluated
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    dim : :class:`float`
        Flow dimension. Default: 3
    lat_ext : :class:`float`
        The lateral extend of the flow-domain in `L^(3-dim)`. Default: 1
    rpart : :class:`numpy.ndarray` of length N+1
        Given radii separating the disks as well as starting- and endpoints
    Kpart : :class:`numpy.ndarray` of length N
        Given conductivity values for each disk
    Spart : :class:`numpy.ndarray`of length N
        Given storativity values for each disk
    Qw : :class:`float`
        Pumpingrate at the well
    Twell : :class:`float`, optional
        Transmissivity at the well. Default: ``Tpart[0]``

    Returns
    -------
    grf_laplace : :class:`numpy.ndarray`
        Array with all values in laplace-space

    Example
    -------
    >>> grf_laplace([5,10],[1,2,3], 2, 1, [0,2,10],[1e-3,1e-3],[1e-3,2e-3],-1)
    array([[ -2.71359196e+00,  -1.66671965e-01,  -2.82986917e-02],
           [ -4.58447458e-01,  -1.12056319e-02,  -9.85673855e-04]])
    """

    # ensure that input is treated as arrays
    s = np.squeeze(s).reshape(-1)
    rad = np.squeeze(rad).reshape(-1)
    rpart = np.squeeze(rpart).reshape(-1)
    Spart = np.squeeze(Spart).reshape(-1)
    Kpart = np.squeeze(Kpart).reshape(-1)

    # the dimension is used by nu in the literature
    dim = float(dim)
    nu = 1.0 - dim / 2.0
    # the lateral extend is a bit subtle in fractured dimension
    lat_ext = float(lat_ext)
    Qw = float(Qw)

    # get the number of partitions
    parts = len(Kpart)

    # initialize the result
    res = np.zeros(s.shape + rad.shape)

    # set the general pumping-condtion
    if Kwell is None:
        Kwell = Kpart[0]

    # the first sqrt of the diffusivity values
    diff_sr0 = np.sqrt(Spart[0] / Kpart[0])
    # the pumping-condition
    Q = Qw / (Kwell * sph_surf(dim) * lat_ext ** (3.0 - dim))

    if rpart[0] > 0.0:
        Q /= -diff_sr0 * rpart[0] ** (1 - nu)
        Qs = Q * s ** (-1.5)
    else:
        Q *= (2 / diff_sr0) ** nu / gamma(1 - nu)
        Qs = Q * s ** (-nu / 2 - 1)

    # if there is a homgeneouse aquifer, compute the result by hand
    if parts == 1:
        # calculate the square-root of the diffusivities
        difsr = np.sqrt(Spart[0] / Kpart[0])

        # initialize the equation system
        V = np.zeros(2, dtype=float)
        M = np.eye(2, dtype=float)

        for si, se in enumerate(s):
            Cs = np.sqrt(se) * difsr

            # set the pumping-condition at the well
            V[0] = Qs[si]

            # incorporate the boundary-conditions
            if rpart[0] > 0.0:
                M[0, :] = [
                    -kv(nu - 1, Cs * rpart[0]),
                    iv(nu - 1, Cs * rpart[0]),
                ]
            if rpart[-1] < np.inf:
                M[1, :] = [kv(nu, Cs * rpart[-1]), iv(nu, Cs * rpart[-1])]

            # solve the equation system
            As, Bs = np.linalg.solve(M, V)

            # calculate the head
            for ri, re in enumerate(rad):
                if re < rpart[-1]:
                    res[si, ri] = re ** nu * (
                        As * kv(nu, Cs * re) + Bs * iv(nu, Cs * re)
                    )

    # if there is more than one partition, create an equation system
    else:
        # initialize LHS and RHS for the linear equation system
        # Mb is the banded matrix for the Eq-System
        V = np.zeros(2 * (parts))
        Mb = np.zeros((5, 2 * (parts)))
        # the positions of the diagonals of the matrix set in Mb
        diagpos = [2, 1, 0, -1, -2]
        # set the standard boundary conditions for rwell=0.0 and rinf=np.inf
        Mb[2, 0] = 1.0
        Mb[-3, -1] = 1.0

        # calculate the consecutive fractions of the conductivities
        Kfrac = Kpart[:-1] / Kpart[1:]

        # calculate the square-root of the diffusivities
        difsr = np.sqrt(Spart / Kpart)

        # calculate a temporal substitution
        tmp = Kfrac * difsr[:-1] / difsr[1:]

        # match the radii to the different disks
        pos = np.searchsorted(rpart, rad) - 1

        # iterate over the laplace-variable
        for si, se in enumerate(s):
            Cs = np.sqrt(se) * difsr

            # set the pumping-condition at the well
            # --> implement other pumping conditions
            V[0] = Qs[si]

            # set the boundary-conditions if needed
            if rpart[0] > 0.0:
                Mb[2, 0] = -kv(nu - 1, Cs[0] * rpart[0])
                Mb[1, 1] = iv(nu - 1, Cs[0] * rpart[0])
            if rpart[-1] < np.inf:
                Mb[-2, -2] = kv(nu, Cs[-1] * rpart[-1])
                Mb[-3, -1] = iv(nu, Cs[-1] * rpart[-1])

            # generate the equation system as banded matrix
            for i in range(parts - 1):
                Mb[0, 2 * i + 3] = -iv(nu, Cs[i + 1] * rpart[i + 1])
                Mb[1, 2 * i + 2 : 2 * i + 4] = [
                    -kv(nu, Cs[i + 1] * rpart[i + 1]),
                    -iv(nu - 1, Cs[i + 1] * rpart[i + 1]),
                ]
                Mb[2, 2 * i + 1 : 2 * i + 3] = [
                    iv(nu, Cs[i] * rpart[i + 1]),
                    kv(nu - 1, Cs[i + 1] * rpart[i + 1]),
                ]
                Mb[3, 2 * i : 2 * i + 2] = [
                    kv(nu, Cs[i] * rpart[i + 1]),
                    tmp[i] * iv(nu - 1, Cs[i] * rpart[i + 1]),
                ]
                Mb[4, 2 * i] = -tmp[i] * kv(nu - 1, Cs[i] * rpart[i + 1])

            # genearate the cooeficient matrix as a spare matrix
            M = sps.spdiags(Mb, diagpos, 2 * parts, 2 * parts, format="csc")

            # solve the Eq-Sys and ignore errors from the umf-pack
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore")
                warnings.simplefilter("ignore", SlvWarn)
                warnings.simplefilter("ignore", RuntimeWarning)
                X = spsolve(M, V, use_umfpack=True)

            # to suppress numerical errors, set NAN values to 0
            # --> the algorithm tends to violate small values,
            #     therefore this approachu is suitable
            X[np.logical_not(np.isfinite(X))] = 0.0

            # calculate the head
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res[si, :] = rad ** nu * (
                    X[2 * pos] * kv(nu, Cs[pos] * rad)
                    + X[2 * pos + 1] * iv(nu, Cs[pos] * rad)
                )

        # set problematic values to 0
        # --> the algorithm tends to violate small values,
        #     therefore this approachu is suitable
        res[np.logical_not(np.isfinite(res))] = 0.0

    return res


if __name__ == "__main__":
    import doctest

    doctest.testmod()
