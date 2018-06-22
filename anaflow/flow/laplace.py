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
from scipy.special import (i0, i1, k0, k1)

# ignore Umpfack warnings for almost singular matrices
try:
    # first look if the umfpack is available
    from scikits.umfpack import UmfpackWarning
    SlvWarn = UmfpackWarning
except ImportError:
    # if umfpack is not present, use standard-warning to catch
    SlvWarn = UserWarning

__all__ = [
    "lap_trans_flow_cyl",
]


###############################################################################
# The generic solver of the 2D radial transient groundwaterflow equation
# in Laplace-space with a pumping condition and a fix zero boundary-head
###############################################################################

def lap_trans_flow_cyl(s, rad=None, rpart=None,
                       Spart=None, Tpart=None, Qw=None, Twell=None):
    '''
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
    '''

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
    Q = Qw/(2.0*np.pi*Twell)

    # if there is a homgeneouse aquifer, compute the result by hand
    if parts == 1:
        # calculate the square-root of the diffusivities
        difsr = np.sqrt(Spart[0]/Tpart[0])

        for si, se in np.ndenumerate(s):
            Cs = np.sqrt(se)*difsr

            # set the pumping-condition at the well
            Qs = Q/se

            # incorporate the boundary-conditions
            if rpart[0] == 0.0:
                Bs = Qs
                if rpart[-1] == np.inf:
                    As = 0.0
                else:
                    As = -Qs*k0(Cs*rpart[-1])/i0(Cs*rpart[-1])

            else:
                if rpart[-1] == np.inf:
                    As = 0.0
                    Bs = Qs/(Cs*rpart[0]*k1(Cs*rpart[0]))
                else:
                    det = i1(Cs*rpart[0])*k0(Cs*rpart[-1]) \
                        + k1(Cs*rpart[0])*i0(Cs*rpart[-1])
                    As = -Qs/(Cs*rpart[0])*k0(Cs*rpart[-1])/det
                    Bs = Qs/(Cs*rpart[0])*i0(Cs*rpart[-1])/det

            # calculate the head
            for ri, re in np.ndenumerate(rad):
                if re < rpart[-1]:
                    res[si+ri] = As*i0(Cs*re) + Bs*k0(Cs*re)

    # if there is more than one partition, create an equation system
    else:
        # initialize LHS and RHS for the linear equation system
        # Mb is the banded matrix for the Eq-System
        V = np.zeros(2*(parts))
        Mb = np.zeros((5, 2*(parts)))
        # the positions of the diagonals of the matrix set in Mb
        diagpos = [2, 1, 0, -1, -2]
        # set the standard boundary conditions for rwell=0.0 and rinf=np.inf
        Mb[1, 1] = 1.0
        Mb[-2, -2] = 1.0

        # calculate the consecutive fractions of the transmissivities
        Tfrac = Tpart[:-1]/Tpart[1:]

        # calculate the square-root of the diffusivities
        difsr = np.sqrt(Spart/Tpart)

        # calculate a temporal substitution
        tmp = Tfrac*difsr[:-1]/difsr[1:]

        # match the radii to the different disks
        pos = np.searchsorted(rpart, rad) - 1

        # iterate over the laplace-variable
        for si, se in enumerate(s):
            Cs = np.sqrt(se)*difsr

            # set the pumping-condition at the well
            # --> implement other pumping conditions
            V[0] = Q/se

            # set the boundary-conditions if needed
            if rpart[0] > 0.0:
                Mb[1, 1] = Cs[0]*rpart[0]*k1(Cs[0]*rpart[0])
                Mb[0, 2] = -Cs[0]*rpart[0]*i1(Cs[0]*rpart[0])
            if rpart[-1] < np.inf:
                Mb[-3, -1] = k0(Cs[-1]*rpart[-1])
                Mb[-2, -2] = i0(Cs[-1]*rpart[-1])

            # generate the equation system as banded matrix
            for i in range(parts-1):
                Mb[0, 2*i+3] = -k0(Cs[i+1]*rpart[i+1])
                Mb[1, 2*i+2:2*i+4] = [-i0(Cs[i+1]*rpart[i+1]),
                                      k1(Cs[i+1]*rpart[i+1])]
                Mb[2, 2*i+1:2*i+3] = [k0(Cs[i]*rpart[i+1]),
                                      -i1(Cs[i+1]*rpart[i+1])]
                Mb[3, 2*i:2*i+2] = [i0(Cs[i]*rpart[i+1]),
                                    -tmp[i]*k1(Cs[i]*rpart[i+1])]
                Mb[4, 2*i] = tmp[i]*i1(Cs[i]*rpart[i+1])

            # genearate the cooeficient matrix as a spare matrix
            M = sps.spdiags(Mb, diagpos, 2*parts, 2*parts, format="csc")

            # solve the Eq-Sys and ignore errors from the umf-pack
            with warnings.catch_warnings():
                # warnings.simplefilter("ignore")
                warnings.simplefilter("ignore", SlvWarn)
                warnings.simplefilter("ignore", RuntimeWarning)
                X = sps.linalg.spsolve(M, V, use_umfpack=True)

            # to suppress numerical errors, set NAN values to 0
            X[np.logical_not(np.isfinite(X))] = 0.0

            # calculate the head
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                res[si, :] = (X[2*pos]*i0(Cs[pos]*rad) +
                              X[2*pos+1]*k0(Cs[pos]*rad))

        # set problematic values to 0
        # --> the algorithm tends to violate small values,
        #     therefore this approachu is suitable
        res[np.logical_not(np.isfinite(res))] = 0.0

    return res


if __name__ == "__main__":
    import doctest
    doctest.testmod()
