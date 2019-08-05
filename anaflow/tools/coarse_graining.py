# -*- coding: utf-8 -*-
"""
Anaflow subpackage providing helper functions related to coarse graining.

.. currentmodule:: anaflow.tools.coarse_graining

The following functions are provided

.. autosummary::

   T_CG
   T_CG_inverse
   T_CG_error
   K_CG
   K_CG_inverse
   K_CG_error
   TPL_CG
   TPL_CG_error
"""
# pylint: disable=C0103
from __future__ import absolute_import, division, print_function

import numpy as np
from scipy.optimize import root

from anaflow.tools.special import aniso, tpl_hyp

__all__ = [
    "T_CG",
    "T_CG_inverse",
    "T_CG_error",
    "K_CG",
    "K_CG_inverse",
    "K_CG_error",
    "TPL_CG",
    "TPL_CG_error",
]


def T_CG(rad, trans_gmean, var, len_scale, T_well=None, prop=1.6):
    """
    The coarse-graining Transmissivity.

    This solution was presented in ''Schneider & Attinger 2008''[R3]_.

    This functions gives an effective
    transmissivity for 2D pumpingtests in heterogenous aquifers, where the
    transmissivity is following a log-normal distribution and a gaussian
    correlation function.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
    T_well : :class:`float`, optional
        Explicit transmissivity value at the well. Harmonic mean by default.
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

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
    array([0.00061831, 0.00064984, 0.00069236])
    """
    chi = -var / 2.0 if T_well is None else np.log(T_well / trans_gmean)
    return trans_gmean * np.exp(chi / (1.0 + (prop * rad / len_scale) ** 2))


def T_CG_inverse(T, trans_gmean, var, len_scale, T_well=None, prop=1.6):
    """
    The inverse coarse-graining Transmissivity.

    See: :func:`T_CG`

    Parameters
    ----------
    T : :class:`numpy.ndarray`
        Array with all transmissivity values
        where the function should be evaluated
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
    T_well : :class:`float`, optional
        Explicit transmissivity value at the well. Harmonic mean by default.
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    rad : :class:`numpy.ndarray`
        Array containing the radii belonging to the given transmissivity values

    Examples
    --------
    >>> T_CG_inverse([7e-4,8e-4,9e-4], 0.001, 1, 10, 2)
    array([3.16952925, 5.56935826, 9.67679026])
    """
    chi = -var / 2.0 if T_well is None else np.log(T_well / trans_gmean)
    return (len_scale / prop) * np.sqrt(chi / np.log(T / trans_gmean) - 1.0)


def T_CG_error(err, trans_gmean, var, len_scale, T_well=None, prop=1.6):
    """
    Calculating the radial-point for given error.

    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance.
    See: :func:`T_CG`

    Parameters
    ----------
    err : :class:`float`
        Given relative error for the farfield transmissivity
    trans_gmean : :class:`float`
        Geometric-mean transmissivity.
    var : :class:`float`
        Variance of log-transmissivity.
    len_scale : :class:`float`
        Correlation-length of log-transmissivity.
    T_well : :class:`float`, optional
        Explicit transmissivity value at the well. Harmonic mean by default.
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    rad : :class:`float`
        Radial point, where the relative error is less than the given one.

    Examples
    --------
    >>> T_CG_error(0.01, 0.001, 1, 10, 2)
    34.91045016779039
    """
    chi = -var / 2.0 if T_well is None else np.log(T_well / trans_gmean)
    if chi > 0.0:
        if chi / np.log(1.0 + err) >= 1.0:
            return (len_scale / prop) * np.sqrt(chi / np.log(1.0 + err) - 1.0)
        # standard value if the error is less then the variation
        return 0
    if chi / np.log(1.0 - err) >= 1.0:
        return (len_scale / prop) * np.sqrt(chi / np.log(1.0 - err) - 1.0)
    # standard value if the error is less then the variation
    return 0


def K_CG(rad, cond_gmean, var, len_scale, anis, K_well="KH", prop=1.6):
    """
    The coarse-graining conductivity.

    This solution was presented in ''Zech 2013''[R8]_.

    This functions gives an effective
    conductivity for 3D pumpingtests in heterogenous aquifers, where the
    conductivity is following a log-normal distribution and a gaussian
    correlation function and taking vertical anisotropy into account.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
    var : :class:`float`
        Variance of the log-conductivity.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    anis : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
    K_well : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

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
    array([0.00063008, 0.00069285, 0.00077595])
    """
    K_efu = cond_gmean * np.exp(var * (0.5 - aniso(anis)))
    if K_well == "KH":
        chi = var * (aniso(anis) - 1.0)
    elif K_well == "KA":
        chi = var * aniso(anis)
    else:
        chi = np.log(K_well / K_efu)

    return K_efu * np.exp(
        chi
        / np.sqrt(1.0 + (prop * rad / (len_scale * anis ** (1.0 / 3.0))) ** 2)
        ** 3
    )


def K_CG_inverse(K, cond_gmean, var, len_scale, anis, K_well="KH", prop=1.6):
    """
    The inverse coarse-graining conductivity.

    See: :func:`K_CG`

    Parameters
    ----------
    K : :class:`numpy.ndarray`
        Array with all conductivity values
        where the function should be evaluated
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
    var : :class:`float`
        Variance of the log-conductivity.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    anis : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
    K_well : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    rad : :class:`numpy.ndarray`
        Array containing the radii belonging to the given conductivity values

    Examples
    --------
    >>> K_CG_inverse([7e-4,8e-4,9e-4], 0.001, 1, 10, 1, 2)
    array([2.09236867, 3.27914996, 4.52143956])
    """
    K_efu = cond_gmean * np.exp(var * (0.5 - aniso(anis)))
    if K_well == "KH":
        chi = var * (aniso(anis) - 1.0)
    elif K_well == "KA":
        chi = var * aniso(anis)
    else:
        chi = np.log(K_well / K_efu)

    return (
        len_scale
        * anis ** (1.0 / 3.0)
        / prop
        * np.sqrt((chi / np.log(K / K_efu)) ** (2.0 / 3.0) - 1.0)
    )


def K_CG_error(err, cond_gmean, var, len_scale, anis, K_well="KH", prop=1.6):
    """
    Calculating the radial-point for given error.

    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance.
    See: :func:`K_CG`

    Parameters
    ----------
    err : :class:`float`
        Given relative error for the farfield conductivity
    cond_gmean : :class:`float`
        Geometric-mean conductivity.
    var : :class:`float`
        Variance of the log-conductivity.
    len_scale : :class:`float`
        Corralation-length of log-conductivity.
    anis : :class:`float`
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
    K_well : string/float, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``), the arithmetic mean (``"KA"``) or an
        arbitrary float value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    rad : :class:`float`
        Radial point, where the relative error is less than the given one.

    Examples
    --------
    >>> K_CG_error(0.01, 0.001, 1, 10, 1, 2)
    19.612796453639845
    """
    K_efu = cond_gmean * np.exp(var * (0.5 - aniso(anis)))
    if K_well == "KH":
        chi = var * (aniso(anis) - 1.0)
    elif K_well == "KA":
        chi = var * aniso(anis)
    else:
        chi = np.log(K_well / K_efu)

    coef = len_scale * anis ** (1.0 / 3.0) / prop

    if chi > 0.0:
        if chi / np.log(1.0 + err) >= 1.0:
            return coef * np.sqrt(
                (chi / np.log(1.0 + err)) ** (2.0 / 3.0) - 1.0
            )
        # standard value if the error is less then the variation
        return 0

    if chi / np.log(1.0 - err) >= 1.0:
        return coef * np.sqrt((chi / np.log(1.0 - err)) ** (2.0 / 3.0) - 1.0)
    # standard value if the error is less then the variation
    return 0


def TPL_CG(
    rad,
    cond_gmean,
    len_scale,
    hurst,
    var=None,
    c=1.0,
    anis=1,
    dim=2.0,
    K_well="KH",
    prop=1.6,
):
    """
    The gaussian truncated power-law coarse-graining conductivity.

    Parameters
    ----------
    rad : :class:`numpy.ndarray`
        Array with all radii where the function should be evaluated
    cond_gmean : :class:`float`
        Geometric-mean conductivity
    len_scale : :class:`float`
        upper bound of the corralation-length of conductivity-distribution
    hurst: :class:`float`
        Hurst coefficient of the TPL model. Should be in (0, 1).
    var : :class:`float` or :any:`None`, optional
        Variance of log-conductivity
        If given, c will be calculated accordingly.
        Default: :any:`None`
    c : :class:`float`, optional
        Intensity of variation in the TPL model.
        Is overwritten if var is given.
        Default: ``1.0``
    anis : :class:`float`, optional
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
        This is only applied in 3 dimensions.
        Default: 1.0
    dim: :class:`float`, optional
        Dimension of space.
        Default: ``2.0``
    K_well :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``),
        the arithmetic mean (``"KA"``) or an arbitrary float
        value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    TPL_CG : :class:`numpy.ndarray`
        Array containing the effective conductivity values.
    """
    # handle special case in 3D with anisotropy
    anis = 1.0 if not np.isclose(dim, 3) else anis
    ani = aniso(anis) if np.isclose(dim, 3) else 1.0 / dim
    var = c * len_scale ** (2 * hurst) / (2 * hurst) if var is None else var
    K_efu = cond_gmean * np.exp(var * (0.5 - ani))
    if K_well == "KH":
        chi = var * (ani - 1.0)
    elif K_well == "KA":
        chi = var * ani
    else:
        chi = np.log(K_well / K_efu)

    return K_efu * np.exp(
        (chi * 2.0 * hurst / (dim + 2.0 * hurst))
        * tpl_hyp(rad, dim, hurst, len_scale * anis ** (1 / 3.0), prop)
    )


def TPL_CG_error(
    err,
    cond_gmean,
    len_scale,
    hurst,
    var=None,
    c=1.0,
    anis=1,
    dim=2.0,
    K_well="KH",
    prop=1.6,
):
    """
    Calculating the radial-point for given error.

    Calculating the radial-point where the relative error of the farfield
    value is less than the given tollerance.
    See: :func:`TPL_CG`

    Parameters
    ----------
    err : :class:`float`
        Given relative error for the farfield conductivity
    cond_gmean : :class:`float`
        Geometric-mean conductivity
    len_scale : :class:`float`
        upper bound of the corralation-length of conductivity-distribution
    hurst: :class:`float`
        Hurst coefficient of the TPL model. Should be in (0, 1).
    var : :class:`float` or :any:`None`, optional
        Variance of log-conductivity
        If given, c will be calculated accordingly.
        Default: :any:`None`
    c : :class:`float`, optional
        Intensity of variation in the TPL model.
        Is overwritten if var is given.
        Default: ``1.0``
    anis : :class:`float`, optional
        Anisotropy-ratio of the vertical and horizontal corralation-lengths.
        This is only applied in 3 dimensions.
        Default: 1.0
    dim: :class:`float`, optional
        Dimension of space.
        Default: ``2.0``
    K_well :  :class:`str` or  :class:`float`, optional
        Explicit conductivity value at the well. One can choose between the
        harmonic mean (``"KH"``),
        the arithmetic mean (``"KA"``) or an arbitrary float
        value. Default: ``"KH"``
    prop: :class:`float`, optional
        Proportionality factor used within the upscaling procedure.
        Default: ``1.6``

    Returns
    -------
    rad : :class:`float`
        Radial point, where the relative error is less than the given one.
    """
    # handle special case in 3D with anisotropy
    anis = 1.0 if not np.isclose(dim, 3) else anis
    ani = aniso(anis) if np.isclose(dim, 3) else 1.0 / dim
    var = c * len_scale ** (2 * hurst) / (2 * hurst) if var is None else var
    K_efu = cond_gmean * np.exp(var * (0.5 - ani))
    if K_well == "KH":
        chi = var * (ani - 1.0)
    elif K_well == "KA":
        chi = var * ani
    else:
        chi = np.log(K_well / K_efu)
    Kw = np.exp(chi + np.log(K_efu))

    # define a curve, that has its root at the wanted percentile
    if chi > 0:
        per = (1 + err) * K_efu
        if not per < Kw:
            return 0
    elif chi < 0:
        per = (1 - err) * K_efu
        if not per > Kw:
            return 0
    else:
        return 0

    def curve(x):
        """Curve for fitting."""
        return (
            TPL_CG(
                x,
                cond_gmean=cond_gmean,
                len_scale=len_scale,
                hurst=hurst,
                var=var,
                c=c,
                anis=anis,
                dim=dim,
                K_well=K_well,
                prop=prop,
            )
            - per
        )

    return root(curve, 2 * len_scale)["x"][0]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
