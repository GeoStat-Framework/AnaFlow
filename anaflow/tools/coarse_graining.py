# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing several helper functions related to coarse graining.

.. currentmodule:: anaflow.tools.coarse_graining

The following functions are provided

.. autosummary::

   T_CG
   T_CG_inverse
   T_CG_error
   K_CG
   K_CG_inverse
   K_CG_error
"""

from __future__ import absolute_import, division, print_function

import numpy as np

from anaflow.tools.special import aniso

__all__ = [
    "T_CG",
    "T_CG_inverse",
    "T_CG_error",
    "K_CG",
    "K_CG_inverse",
    "K_CG_error",
]


def T_CG(rad, TG, sig2, corr, prop=1.6, Twell=None):
    """
    The coarse-graining Transmissivity

    This solution was presented in ''Schneider & Attinger 2008''[R3]_.

    This functions gives an effective
    transmissivity for 2D pumpingtests in heterogenous aquifers, where the
    transmissivity is following a log-normal distribution and a gaussian
    correlation function.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    TG : :class:`float`
        Geometric-mean of the transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``

    Returns
    -------
    T_CG : :class:`numpy.ndarray`
        Array containing the effective transmissivity values.

    References
    ----------
    .. [R3] Schneider, C. and Attinger, S.,
       ''Beyond thiem: A new method for interpreting large scale
       pumping tests in heterogeneous aquifers.''
       Water resources research, 44(4), 2008

    Examples
    --------
    >>> T_CG([1,2,3], 0.001, 1, 10, 2)
    array([ 0.00061831,  0.00064984,  0.00069236])
    """

    rad = np.squeeze(rad)

    if Twell is not None:
        chi = np.log(Twell) - np.log(TG)
    else:
        chi = -sig2 / 2.0

    return TG * np.exp(chi / (1.0 + (prop * rad / corr) ** 2))


def T_CG_inverse(T, TG, sig2, corr, prop=1.6, Twell=None):
    """
    The inverse coarse-graining Transmissivity

    See: :func:`T_CG`

    Parameters
    ----------
    T : :class:`numpy.ndarray`
        Array with all transmissivity values
        where the function should be evaluated
    TG : :class:`float`
        Geometric-mean of the transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``

    Returns
    -------
    rad : :class:`numpy.ndarray`
        Array containing the radii belonging to the given transmissivity values

    Examples
    --------
    >>> T_CG_inverse([7e-4,8e-4,9e-4], 0.001, 1, 10, 2)
    array([ 3.16952925,  5.56935826,  9.67679026])
    """

    T = np.squeeze(T)

    if Twell is not None:
        chi = np.log(Twell) - np.log(TG)
    else:
        chi = -sig2 / 2.0

    return (corr / prop) * np.sqrt(chi / np.log(T / TG) - 1.0)


def T_CG_error(err, TG, sig2, corr, prop=1.6, Twell=None):
    """
    Calculating the radial-point for given error

    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance.
    See: :func:`T_CG`

    Parameters
    ----------
    err : :class:`float`
        Given relative error for the farfield transmissivity
    TG : :class:`float`
        Geometric-mean of the transmissivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the transmissivity-distribution
    corr : :class:`float`
        corralation-length of transmissivity-distribution
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Twell : :class:`float`, optional
        Explicit transmissivity value at the well. Default: ``None``

    Returns
    -------
    rad : :class:`float`
        Radial point, where the relative error is less than the given one.

    Examples
    --------
    >>> T_CG_error(0.01, 0.001, 1, 10, 2)
    34.910450167790387
    """

    if Twell is not None:
        chi = np.log(Twell) - np.log(TG)
    else:
        chi = -sig2 / 2.0

    if chi > 0.0:
        if chi / np.log(1.0 + err) >= 1.0:
            return (corr / prop) * np.sqrt(chi / np.log(1.0 + err) - 1.0)
        # standard value if the error is less then the variation
        return 1.0
    else:
        if chi / np.log(1.0 - err) >= 1.0:
            return (corr / prop) * np.sqrt(chi / np.log(1.0 - err) - 1.0)
        # standard value if the error is less then the variation
        return 1.0


def K_CG(rad, KG, sig2, corr, e, prop=1.6, Kwell="KH"):
    """
    The coarse-graining conductivity

    This solution was presented in ''Zech 2013''[R8]_.

    This functions gives an effective
    conductivity for 3D pumpingtests in heterogenous aquifers, where the
    conductivity is following a log-normal distribution and a gaussian
    correlation function and taking vertical anisotropy into account.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    KG : :class:`float`
        Geometric-mean conductivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the conductivity-distribution
    corr : :class:`float`
        corralation-length of conductivity-distribution
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Kwell :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``),
        the arithmetic mean (``"KA"``) or an arbitrary float
        value. Default: ``"KH"``

    Returns
    -------
    K_CG : :class:`numpy.ndarray`
        Array containing the effective conductivity values.

    References
    ----------
    .. [R8] Zech, A.
       ''Impact of Aqifer Heterogeneity on Subsurface Flow and Salt
       Transport at Different Scales: from a method determine parameters
       of heterogeneous permeability at local scale to a large-scale model
       for the sedimentary basin of Thuringia.''
       PhD thesis, Friedrich-Schiller-Universität Jena, 2013

    Examples
    --------
    >>> K_CG([1,2,3], 0.001, 1, 10, 1, 2)
    array([ 0.00063008,  0.00069285,  0.00077595])
    """

    rad = np.squeeze(rad)

    Kefu = KG * np.exp(sig2 * (0.5 - aniso(e)))
    if Kwell == "KH":
        chi = sig2 * (aniso(e) - 1.0)
    elif Kwell == "KA":
        chi = sig2 * aniso(e)
    else:
        chi = np.log(Kwell) - np.log(Kefu)

    return Kefu * np.exp(
        chi / np.sqrt(1.0 + (prop * rad / (corr * e ** (1.0 / 3.0))) ** 2) ** 3
    )


def K_CG_inverse(K, KG, sig2, corr, e, prop=1.6, Kwell="KH"):
    """
    The inverse coarse-graining conductivity

    See: :func:`K_CG`

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        Array with all conductivity values
        where the function should be evaluated
    KG : :class:`float`
        Geometric-mean conductivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the conductivity-distribution
    corr : :class:`float`
        corralation-length of conductivity-distribution
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Kwell :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean ("KH"), the arithmetic mean ("KA") or an arbitrary float
        value. Default: ``"KH"``

    Returns
    -------
    rad : :class:`numpy.ndarray`
        Array containing the radii belonging to the given conductivity values

    Examples
    --------
    >>> K_CG_inverse([7e-4,8e-4,9e-4], 0.001, 1, 10, 1, 2)
    array([ 2.09236867,  3.27914996,  4.52143956])
    """

    K = np.squeeze(K)

    Kefu = KG * np.exp(sig2 * (0.5 - aniso(e)))
    if Kwell == "KH":
        chi = sig2 * (aniso(e) - 1.0)
    elif Kwell == "KA":
        chi = sig2 * aniso(e)
    else:
        chi = np.log(Kwell) - np.log(Kefu)

    return (
        corr
        * e ** (1.0 / 3.0)
        / prop
        * np.sqrt((chi / np.log(K / Kefu)) ** (2.0 / 3.0) - 1.0)
    )


def K_CG_error(err, KG, sig2, corr, e, prop=1.6, Kwell="KH"):
    """
    Calculating the radial-point for given error

    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance.
    See: :func:`K_CG`

    Parameters
    ----------
    err : :class:`float`
        Given relative error for the farfield conductivity
    KG : :class:`float`
        Geometric-mean conductivity-distribution
    sig2 : :class:`float`
        log-normal-variance of the conductivity-distribution
    corr : :class:`float`
        corralation-length of conductivity-distribution
    e : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``
    Kwell :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean ("KH"), the arithmetic mean ("KA") or an arbitrary float
        value. Default: ``"KH"``

    Returns
    -------
    rad : :class:`float`
        Radial point, where the relative error is less than the given one.

    Examples
    --------
    >>> K_CG_error(0.01, 0.001, 1, 10, 1, 2)
    19.612796453639845
    """

    Kefu = KG * np.exp(sig2 * (0.5 - aniso(e)))
    if Kwell == "KH":
        chi = sig2 * (aniso(e) - 1.0)
    elif Kwell == "KA":
        chi = sig2 * aniso(e)
    else:
        chi = np.log(Kwell) - np.log(Kefu)

    coef = corr * e ** (1.0 / 3.0) / prop

    if chi > 0.0:
        if chi / np.log(1.0 + err) >= 1.0:
            return coef * np.sqrt(
                (chi / np.log(1.0 + err)) ** (2.0 / 3.0) - 1.0
            )
        # standard value if the error is less then the variation
        return 1.0
    else:
        if chi / np.log(1.0 - err) >= 1.0:
            return coef * np.sqrt(
                (chi / np.log(1.0 - err)) ** (2.0 / 3.0) - 1.0
            )
        # standard value if the error is less then the variation
        return 1.0


if __name__ == "__main__":
    import doctest

    doctest.testmod()
